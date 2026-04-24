"""
Bivariate survival simulation: IPCW, doubly robust, and PPI-Old (R-style) with
LimiX missing-value imputation of X2 from (X1, X2) instead of R ranger.

Run in conda env: limix_test
  pip install lifelines
  python examples/rmst_ppi_cox_bivariate_limiX.py --m 5

or:
  conda run -n limix_test pip install lifelines
  conda run -n limix_test python examples/rmst_ppi_cox_bivariate_limiX.py --m 1 --n-pre 500 --n-inf 10000
  (默认 LimiX 用 GPU 若 ``torch.cuda.is_available()``，否则 CPU；可显式 ``--device cpu``)

Paths: run with cwd = LimiX/ so ``config/reg_default_noretrieval_MVI.json`` resolves; the script chdirs to LimiX when possible.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

try:
    from lifelines import CoxPHFitter
except ImportError as e:
    raise SystemExit("Please install lifelines, e.g.: pip install lifelines") from e

# LimiX package root (this file: LimiX/examples/...)
_LIMIX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _LIMIX_ROOT not in sys.path:
    sys.path.insert(0, _LIMIX_ROOT)
from inference.predictor import LimiXPredictor  # noqa: E402


# --- DGP: matches R (MASS mvrnorm + Weibull, truncation at tau) -----------------


def generate_data(
    n: int,
    rng: np.random.Generator,
    rho: float = 0.7,
    tau: float = 2.0,
) -> pd.DataFrame:
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    x1, x2 = rng.multivariate_normal(np.zeros(2), cov, size=n).T
    z1 = 1.5 * x1 + 1.5 * x2
    lam_t = np.exp(z1)
    t_true = (1.0 / np.sqrt(lam_t)) * (-np.log(rng.random(n))) ** 0.5
    t_trunc = np.minimum(t_true, tau)

    z2 = 0.4 * x1 + 0.8 * x2
    lam_c = np.exp(z2)
    c = (1.5 / lam_c) * (-np.log(rng.random(n))) ** (1.0 / 1.5)

    y = np.minimum(t_trunc, c)
    delta = (t_trunc <= c).astype(np.float64)

    return pd.DataFrame(
        {
            "X1": x1,
            "X2": x2,
            "T_true": t_trunc,
            "Y": y,
            "Delta": delta,
        }
    )


def truth_theta0_rmst(rng: np.random.Generator, n_big: int, tau: float) -> float:
    """E[T_true] = E[min(T, tau)] under DGP. Same role as R big_data mean."""
    d = generate_data(n_big, rng, tau=tau)
    return float(d["T_true"].mean())


def _cumulative_baseline_on_grid(
    cph: CoxPHFitter, grid_t: np.ndarray
) -> np.ndarray:
    base = cph.baseline_cumulative_hazard_.iloc[:, 0]
    t_bh = base.index.to_numpy(dtype=np.float64)
    h_bh = base.to_numpy(dtype=np.float64)
    if len(t_bh) == 0:
        return np.zeros_like(grid_t, dtype=np.float64)
    # Left-continuous evaluation: H0(t) = max cum haz at event times <= t
    j = np.searchsorted(t_bh, grid_t, side="right") - 1
    h0 = np.empty_like(grid_t, dtype=np.float64)
    neg = j < 0
    h0[neg] = 0.0
    h0[~neg] = h_bh[j[~neg]]
    return h0


def _match_r_find_interval(
    y: np.ndarray, grid_t: np.ndarray, k: int
) -> np.ndarray:
    """0-based col index, matching R: i <- findInterval(newdata$Y, grid_t); pmin(pmax(i,1),K)."""
    g = np.sort(np.asarray(grid_t, dtype=np.float64))
    # R findInterval( x, vec, all.inside=TRUE is not the default; use 1..K-1 and clamp
    j = np.searchsorted(g, y, side="right")
    j = np.clip(j, 1, k)
    return (j - 1).astype(np.int64)  # 0-based


def compute_scores(
    newdata: pd.DataFrame,
    cph_t: CoxPHFitter,
    cph_c: CoxPHFitter,
    tau: float,
    x_cols: List[str],
) -> dict:
    n_obs = len(newdata)
    y = newdata["Y"].to_numpy(dtype=np.float64)
    delta = newdata["Delta"].to_numpy(dtype=np.float64)

    bht = cph_t.baseline_cumulative_hazard_.index.to_numpy(dtype=np.float64)
    bhc = cph_c.baseline_cumulative_hazard_.index.to_numpy(dtype=np.float64)
    all_times = np.sort(np.unique(np.concatenate([np.array([0.0]), bht, bhc, [tau]])))
    grid_t = all_times[all_times <= tau]
    k = len(grid_t)
    if k < 2:
        raise ValueError("time grid has fewer than 2 points")

    dt = np.append(np.diff(grid_t), 0.0)

    h0_t = _cumulative_baseline_on_grid(cph_t, grid_t)
    h0_c = _cumulative_baseline_on_grid(cph_c, grid_t)
    dh0_c = np.empty(k, dtype=np.float64)
    dh0_c[0] = 0.0
    dh0_c[1:] = h0_c[1:] - h0_c[:-1]

    r_t = cph_t.predict_partial_hazard(newdata[x_cols].astype(np.float64))
    r_c = cph_c.predict_partial_hazard(newdata[x_cols].astype(np.float64))
    risk_t = r_t.to_numpy().ravel()
    risk_c = r_c.to_numpy().ravel()

    s_t_mat = np.exp(-np.outer(risk_t, h0_t))
    s_c_mat = np.exp(-np.outer(risk_c, h0_c))
    s_c_mat = np.maximum(s_c_mat, 0.05)
    s_t_mat = np.maximum(s_t_mat, 1e-5)

    area_s_t = s_t_mat * dt[None, :]
    
    # ⚡ 极致向量化优化 1：逆向累加使用翻转 numpy 操作替代 for 循环
    # 相当于 R 语言中的向后 cumsum
    int_s_t = np.cumsum(area_s_t[:, ::-1], axis=1)[:, ::-1]
    q_mat = (int_s_t / s_t_mat) + grid_t[None, :]

    integrand_mat = (q_mat / s_c_mat) * (risk_c[:, None]) * dh0_c[None, :]
    
    # ⚡ 极致向量化优化 2：正向累加直接使用 axis=1
    integral_mat = np.cumsum(integrand_mat, axis=1)

    idx = _match_r_find_interval(y, grid_t, k)
    row = np.arange(n_obs)
    
    sc_y = s_c_mat[row, idx]
    q_y = q_mat[row, idx]
    int_y = integral_mat[row, idx]

    term1 = (y * delta) / sc_y
    term2 = (q_y * (1.0 - delta)) / sc_y
    term3 = int_y
    
    ipcw = term1
    dr = term1 + term2 - term3
    
    return {"ipcw": ipcw, "dr": dr}


def fit_cox_t(frame: pd.DataFrame, x_cols: List[str]) -> CoxPHFitter:
    d = frame[["Y", *x_cols]].copy()
    d["Delta"] = frame["Delta"].values.astype(int)

    cph = CoxPHFitter(penalizer=0.0) 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(d, duration_col="Y", event_col="Delta", show_progress=False)
    return cph

def fit_cox_c(frame: pd.DataFrame, x_cols: List[str]) -> CoxPHFitter:
    d = frame[["Y", *x_cols]].copy()
    d["eventC"] = (1.0 - frame["Delta"].to_numpy() > 0.5)

    cph = CoxPHFitter(penalizer=0.0) 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(d, duration_col="Y", event_col="eventC", show_progress=False)
    return cph


def _rep_then_shuffle(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """R: sample(rep(1:K, length.out = n), ...). Returns 1..K per row."""
    base = (np.arange(n) % k) + 1
    return rng.permutation(base)


def limix_impute_x2(
    pre: pd.DataFrame, inf: pd.DataFrame, predictor: LimiXPredictor
) -> np.ndarray:
    """LimiX MVI: train on [X1,X2] with y=X2, infer X2|X1 with all X2 missing + anchor row."""
    x_tr = pre[["X1", "X2"]].to_numpy(np.float32)
    y_tr = pre["X2"].to_numpy(dtype=np.float32, copy=True).ravel()
    n_inf = len(inf)
    x_te = np.array(inf[["X1", "X2"]], dtype=np.float32, copy=True)
    x_te[:, 1] = np.nan
    anc = pre[["X1", "X2"]].to_numpy(np.float32)[:1]
    x_aug = np.vstack([x_te, anc])
    _, mask_pred = predictor.predict(x_tr, y_tr, x_aug, task_type="Regression")
    # test block: first n_inf rows of augmented (anchor is last)
    n_tr = x_tr.shape[0]
    x2_hat = mask_pred[n_tr : n_tr + n_inf, 1]
    return np.asarray(x2_hat, dtype=np.float64)


def load_or_fetch_ckpt(ckpt: Optional[str], root: str) -> str:
    if ckpt and os.path.isfile(ckpt):
        return os.path.abspath(ckpt)
    def_dir = os.path.join(root, "cache", "LimiX-16M.ckpt")
    if os.path.isfile(def_dir):
        return def_dir
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit("Install huggingface_hub and download the checkpoint, or pass --model_path.") from e
    p = hf_hub_download(
        repo_id="stableai-org/LimiX-16M",
        filename="LimiX-16M.ckpt",
        local_dir=os.path.join(root, "cache"),
    )
    return p


def run_single_sim(
    seed: int,
    n_pre: int,
    n_inf: int,
    p_label: float,
    tau: float,
    k_folds: int,
    predictor: LimiXPredictor,
) -> np.ndarray:
    """Returns length-8: IPCW_O, C, I, P, DR_O, C, I, P."""
    rng = np.random.default_rng(seed)
    pre = generate_data(n_pre, rng, tau=tau)
    inf0 = generate_data(n_inf, rng, tau=tau)
    r = rng.binomial(1, p_label, n_inf).astype(np.float64)
    # LimiX impute on full infer
    x2_hat = limix_impute_x2(pre, inf0, predictor)

    dt_inf = inf0.copy()
    dt_inf["Xhat2"] = x2_hat
    dt_inf["R"] = r

    idx_r1 = np.where(r == 1.0)[0]
    n_r1 = int(len(idx_r1))
    n_preinf = n_inf

    for _c in [
        "ipcw_oracle",
        "dr_oracle",
        "ipcw_naive",
        "dr_naive",
        "ipcw_class",
        "dr_class",
    ]:
        dt_inf[_c] = np.nan

    folds = _rep_then_shuffle(n_preinf, k_folds, rng)
    x_cols_ora = ["X1", "X2"]

    for kk in range(1, k_folds + 1):
        tr = np.where(folds != kk)[0]
        te = np.where(folds == kk)[0]
        ftr = dt_inf.iloc[tr]
        fte = dt_inf.iloc[te]
        cox_t_ora = fit_cox_t(ftr, x_cols_ora)
        cox_c_ora = fit_cox_c(ftr, x_cols_ora)
        o = compute_scores(fte, cox_t_ora, cox_c_ora, tau, x_cols_ora)
        dt_inf.loc[te, "ipcw_oracle"] = o["ipcw"]
        dt_inf.loc[te, "dr_oracle"] = o["dr"]

    for kk in range(1, k_folds + 1):
        tr = np.where(folds != kk)[0]
        te = np.where(folds == kk)[0]
        ftr = dt_inf.iloc[tr].copy()
        ftr["X2"] = ftr["Xhat2"]
        fte = dt_inf.iloc[te].copy()
        fte["X2"] = fte["Xhat2"]
        cox_t_nv = fit_cox_t(ftr, x_cols_ora)
        cox_c_nv = fit_cox_c(ftr, x_cols_ora)
        n = compute_scores(fte, cox_t_nv, cox_c_nv, tau, x_cols_ora)
        dt_inf.loc[te, "ipcw_naive"] = n["ipcw"]
        dt_inf.loc[te, "dr_naive"] = n["dr"]

    folds_n = _rep_then_shuffle(n_r1, k_folds, rng) if n_r1 > 0 else np.array([])

    if n_r1 > 0:
        for kk in range(1, k_folds + 1):
            in_k = np.where(folds_n != kk)[0]
            out_k = np.where(folds_n == kk)[0]
            tr_sub = idx_r1[in_k]
            te_sub = idx_r1[out_k]
            ftr = dt_inf.loc[tr_sub, :].copy()
            cox_t_cl = fit_cox_t(ftr, x_cols_ora)
            cox_c_cl = fit_cox_c(ftr, x_cols_ora)
            fte = dt_inf.loc[te_sub, :]
            c = compute_scores(fte, cox_t_cl, cox_c_cl, tau, x_cols_ora)
            dt_inf.loc[te_sub, "ipcw_class"] = c["ipcw"]
            dt_inf.loc[te_sub, "dr_class"] = c["dr"]

    est_ora = (
        float(dt_inf["ipcw_oracle"].mean()),
        float(dt_inf["dr_oracle"].mean()),
    )
    est_nai = (
        float(dt_inf["ipcw_naive"].mean()),
        float(dt_inf["dr_naive"].mean()),
    )
    est_cl = (
        float(dt_inf.loc[dt_inf["R"] == 1.0, "ipcw_class"].mean()) if n_r1 > 0 else np.nan,
        float(dt_inf.loc[dt_inf["R"] == 1.0, "dr_class"].mean()) if n_r1 > 0 else np.nan,
    )
    t1i = float(dt_inf.loc[dt_inf["R"] == 0.0, "ipcw_naive"].mean())
    t2i = float(dt_inf.loc[dt_inf["R"] == 1.0, "ipcw_naive"].mean()) if n_r1 > 0 else 0.0
    t1d = float(dt_inf.loc[dt_inf["R"] == 0.0, "dr_naive"].mean())
    t2d = float(dt_inf.loc[dt_inf["R"] == 1.0, "dr_naive"].mean()) if n_r1 > 0 else 0.0
    ppi_ipcw = t1i - t2i + est_cl[0]
    ppi_dr = t1d - t2d + est_cl[1]

    return np.array(
        [
            est_ora[0],
            est_cl[0],
            est_nai[0],
            ppi_ipcw,
            est_ora[1],
            est_cl[1],
            est_nai[1],
            ppi_dr,
        ],
        dtype=np.float64,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bivariate survival IPCW/DR/PPI with LimiX imputing X2|X1",
    )
    p.add_argument("--m", type=int, default=500, help="Monte Carlo replicates")
    p.add_argument(
        "--seed-gold", type=int, default=999, help="Random seed for theta0 reference"
    )
    p.add_argument(
        "--theta-n", type=int, default=1_000_000, help="N for E[T_true] reference"
    )
    p.add_argument("--n-pre", type=int, default=500)
    p.add_argument("--n-inf", type=int, default=10_000)
    p.add_argument("--p-label", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("-K", "--k-folds", type=int, default=5, dest="k_folds")
    p.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to LimiX-16M.ckpt (else cache or HF download)",
    )
    p.add_argument(
        "--inference_config",
        type=str,
        default="",
        help="Path to reg_default_noretrieval_MVI.json",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="LimiX only: auto=cuda:0 if torch.cuda.is_available() else cpu; or e.g. cuda:0, cpu",
    )
    p.add_argument(
        "--predictor-seed", type=int, default=0, help="LimiX preprocessor seed"
    )
    p.add_argument("--base-seed", type=int, default=2024, help="sim seed: base + m")
    return p.parse_args()


def resolve_limiX_device(device_arg: str) -> torch.device:
    """
    Default LimiX to GPU (cuda:0) when available; use ``cpu`` only if no CUDA.
    ``auto`` or empty string means the same.
    """
    s = (device_arg or "auto").strip().lower()
    if s in ("", "auto", "default"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    root = _LIMIX_ROOT
    try:
        os.chdir(root)
    except OSError:
        pass
    rng_g = np.random.default_rng(args.seed_gold)
    theta0 = truth_theta0_rmst(rng_g, args.theta_n, args.tau)
    print("theta0 (E[T_true]):", theta0, flush=True)

    ck = load_or_fetch_ckpt(
        (args.model_path or None) if args.model_path else None, root
    )
    print("Checkpoint:", ck, flush=True)
    icfg = (
        args.inference_config
        or os.path.join(root, "config", "reg_default_noretrieval_MVI.json")
    )
    if not os.path.isfile(icfg):
        raise SystemExit(
            f"Inference config not found: {icfg}. Chdir to LimiX/ or set --inference_config."
        )

    dev = resolve_limiX_device(args.device)
    print("LimiX device:", dev, flush=True)
    pred = LimiXPredictor(
        device=dev,
        model_path=ck,
        mask_prediction=True,
        inference_config=icfg,
        seed=args.predictor_seed,
    )

    M = max(1, int(args.m))
    names = [
        "IPCW_Oracle",
        "IPCW_Classical",
        "IPCW_Impute",
        "IPCW_PPI",
        "DR_Oracle",
        "DR_Classical",
        "DR_Impute",
        "DR_PPI",
    ]
    res = np.zeros((M, len(names)), dtype=np.float64)
    for m2 in range(1, M + 1):
        print(f"replicate {m2}/{M} ...", flush=True)
        res[m2 - 1, :] = run_single_sim(
            int(args.base_seed) + m2,
            n_pre=args.n_pre,
            n_inf=args.n_inf,
            p_label=args.p_label,
            tau=args.tau,
            k_folds=args.k_folds,
            predictor=pred,
        )

    bias = res.mean(axis=0) - theta0
    ese = res.std(axis=0, ddof=0)
    rmse = np.sqrt(bias**2 + ese**2)
    out = pd.DataFrame(
        {
            "Method": [f"{i+1}. {n}" for i, n in enumerate(names)],
            "Bias": np.round(bias, 6),
            "ESE": np.round(ese, 6),
            "RMSE": np.round(rmse, 6),
        }
    )
    print(f"\n--- Final Unified Simulation Results (M = {M}) ---\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
