"""
Bivariate survival simulation: IPCW, doubly robust, and PPI-Old with
2-Fold Cross-Fitting & STRICT Single Nuisance Model Logic.
LimiX missing-value imputation of X2 from (X1, X2) instead of R ranger.

Run in conda env: limix_test
  pip install lifelines
  python examples/rmst_ppi_cox_bivariate_limiX.py --m 5

or:
  conda run -n limix_test pip install lifelines
  conda run -n limix_test python examples/rmst_ppi_cox_bivariate_limiX.py --m 1 --n-pre 500 --n-inf 10000
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


# ==============================================================================
# 1. DGP (Data Generating Process) matches R
# ==============================================================================
def generate_data(
    n: int,
    rng: np.random.Generator,
    rho: float = 0.7,
    tau: float = 2.0,
    c_base_hazard: float = 1.0,
) -> pd.DataFrame:
    """
    Explicit Cox PH DGP using Inverse Probability Integral Transform.
    """
    # 1. 生成协变量
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    x1, x2 = rng.multivariate_normal(np.zeros(2), cov, size=n).T

    # 2. 显式 Cox PH 生成真实事件时间 T
    beta_t = np.array([1.5, 1.5], dtype=np.float64)
    risk_score_t = np.exp(beta_t[0] * x1 + beta_t[1] * x2)
    
    nu_t = 2.0        
    lambda_t = 1.0    
    u_t = rng.random(n)
    
    t_true = (-np.log(u_t) / (lambda_t * risk_score_t)) ** (1.0 / nu_t)
    t_trunc = np.minimum(t_true, tau)

    # 3. 显式 Cox PH 生成删失时间 C
    beta_c = np.array([0.4, 0.8], dtype=np.float64)
    risk_score_c = np.exp(beta_c[0] * x1 + beta_c[1] * x2)
    
    nu_c = 1.5
    lambda_c = c_base_hazard
    u_c = rng.random(n)
    
    c = (-np.log(u_c) / (lambda_c * risk_score_c)) ** (1.0 / nu_c)

    # 4. 组装观测数据
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

def truth_theta0_rmst(
    rng: np.random.Generator, n_big: int, tau: float
) -> tuple[float, float]:
    d = generate_data(n_big, rng, tau=tau)
    censor_rate = 1 - float(d["Delta"].mean())
    return float(d["T_true"].mean()), censor_rate

# ==============================================================================
# 2. Survival Estimators & IPCW Scores
# ==============================================================================
def _cumulative_baseline_on_grid(cph: CoxPHFitter, grid_t: np.ndarray) -> np.ndarray:
    base = cph.baseline_cumulative_hazard_.iloc[:, 0]
    t_bh = base.index.to_numpy(dtype=np.float64)
    h_bh = base.to_numpy(dtype=np.float64)
    if len(t_bh) == 0:
        return np.zeros_like(grid_t, dtype=np.float64)
    j = np.searchsorted(t_bh, grid_t, side="right") - 1
    h0 = np.empty_like(grid_t, dtype=np.float64)
    neg = j < 0
    h0[neg] = 0.0
    h0[~neg] = h_bh[j[~neg]]
    return h0

def _match_r_find_interval(y: np.ndarray, grid_t: np.ndarray, k: int) -> np.ndarray:
    g = np.sort(np.asarray(grid_t, dtype=np.float64))
    j = np.searchsorted(g, y, side="right")
    j = np.clip(j, 1, k)
    return (j - 1).astype(np.int64) 

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
    s_c_mat = np.maximum(s_c_mat, 1e-5)
    s_t_mat = np.maximum(s_t_mat, 1e-5)

    area_s_t = s_t_mat * dt[None, :]
    int_s_t = np.cumsum(area_s_t[:, ::-1], axis=1)[:, ::-1]
    q_mat = (int_s_t / s_t_mat) + grid_t[None, :]

    integrand_mat = (q_mat / s_c_mat) * (risk_c[:, None]) * dh0_c[None, :]
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

def limix_impute_x2(pre: pd.DataFrame, inf: pd.DataFrame, predictor: LimiXPredictor) -> np.ndarray:
    x_tr = pre[["X1", "X2"]].to_numpy(np.float32)
    y_tr = pre["X2"].to_numpy(dtype=np.float32, copy=True).ravel()
    n_inf = len(inf)
    x_te = np.array(inf[["X1", "X2"]], dtype=np.float32, copy=True)
    x_te[:, 1] = np.nan
    anc = pre[["X1", "X2"]].to_numpy(np.float32)[:1]
    x_aug = np.vstack([x_te, anc])
    _, mask_pred = predictor.predict(x_tr, y_tr, x_aug, task_type="Regression")
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


# ==============================================================================
# 3. Core Compute Logic for Single Fold (Cross-Fitting Engine)
# ==============================================================================
def compute_fold_theta(dt_train: pd.DataFrame, dt_target: pd.DataFrame, tau: float) -> dict:
    """Trains strictly on `dt_train`, predicts strictly on `dt_target`."""
    x_cols = ["X1", "X2"]
    
    idx_R1_train = (dt_train["R"] == 1.0)
    
    idx_R1_target = (dt_target["R"] == 1.0)
    idx_R0_target = (dt_target["R"] == 0.0)

    # -------------------------------------------------------------------------
    # Baseline 1: Oracle (Train on all true target data)
    # -------------------------------------------------------------------------
    cox_t_ora = fit_cox_t(dt_train, x_cols)
    cox_c_ora = fit_cox_c(dt_train, x_cols)
    res_ora = compute_scores(dt_target, cox_t_ora, cox_c_ora, tau, x_cols)
    ora_ipcw = float(res_ora["ipcw"].mean())
    ora_dr   = float(res_ora["dr"].mean())

    # -------------------------------------------------------------------------
    # Baseline 2: Naive Impute (Train on all train with Xhat2)
    # -------------------------------------------------------------------------
    dt_train_naive = dt_train.copy()
    dt_train_naive["X2"] = dt_train_naive["Xhat2"]
    
    dt_target_naive = dt_target.copy()
    dt_target_naive["X2"] = dt_target_naive["Xhat2"]

    cox_t_naive = fit_cox_t(dt_train_naive, x_cols)
    cox_c_naive = fit_cox_c(dt_train_naive, x_cols)
    res_naive = compute_scores(dt_target_naive, cox_t_naive, cox_c_naive, tau, x_cols)
    naive_ipcw = float(res_naive["ipcw"].mean())
    naive_dr   = float(res_naive["dr"].mean())

    # =========================================================================
    # 核心逻辑：训练唯一的 Nuisance 模型 (仅使用 dt_train 中 R=1 的真实 X2)
    # =========================================================================
    dt_train_R1 = dt_train[idx_R1_train].copy()
    cox_t_cla = fit_cox_t(dt_train_R1, x_cols)
    cox_c_cla = fit_cox_c(dt_train_R1, x_cols)

    # --- Term 3 & Classical: target 集中 R=1 样本代入真实 X2 ---
    dt_target_R1 = dt_target[idx_R1_target].copy()
    if len(dt_target_R1) > 0:
        res_cla = compute_scores(dt_target_R1, cox_t_cla, cox_c_cla, tau, x_cols)
        class_ipcw = float(res_cla["ipcw"].mean())
        class_dr   = float(res_cla["dr"].mean())
    else:
        class_ipcw, class_dr = 0.0, 0.0

    # --- Term 1 & Term 2: target 集全体代入预测值 Xhat2 ---
    # 【技巧】：在 target 集上构造代理，强制将真实 X2 列替换为 Xhat2
    dt_target_proxy = dt_target.copy()
    dt_target_proxy["X2"] = dt_target_proxy["Xhat2"]
    res_proxy = compute_scores(dt_target_proxy, cox_t_cla, cox_c_cla, tau, x_cols)

    # 分别在 R=0 和 R=1 的群体中求均值 (使用 target 集)
    ppi_t1_ipcw = float(res_proxy["ipcw"][idx_R0_target.values].mean()) if idx_R0_target.any() else 0.0
    ppi_t2_ipcw = float(res_proxy["ipcw"][idx_R1_target.values].mean()) if idx_R1_target.any() else 0.0

    ppi_t1_dr = float(res_proxy["dr"][idx_R0_target.values].mean()) if idx_R0_target.any() else 0.0
    ppi_t2_dr = float(res_proxy["dr"][idx_R1_target.values].mean()) if idx_R1_target.any() else 0.0

    # --- 组装 PPI 估计量 ---
    ppi_ipcw = ppi_t1_ipcw - ppi_t2_ipcw + class_ipcw
    ppi_dr   = ppi_t1_dr - ppi_t2_dr + class_dr

    return {
        "ora": np.array([ora_ipcw, ora_dr]),
        "naive": np.array([naive_ipcw, naive_dr]),
        "class": np.array([class_ipcw, class_dr]),
        "ppi": np.array([ppi_ipcw, ppi_dr]),
    }


# ==============================================================================
# 4. Main Simulation Loop with 2-Fold Split
# ==============================================================================
def run_single_sim(
    seed: int,
    n_pre: int,
    n_inf: int,
    p_label: float,
    tau: float,
    predictor: LimiXPredictor,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    
    # --- Step 1: 独立模型与插补 ---
    pre = generate_data(n_pre, rng, tau=tau)
    inf0 = generate_data(n_inf, rng, tau=tau)
    r = rng.binomial(1, p_label, n_inf).astype(np.float64)
    
    x2_hat = limix_impute_x2(pre, inf0, predictor)

    dt_inf = inf0.copy()
    dt_inf["Xhat2"] = x2_hat
    dt_inf["R"] = r

    # --- Step 2: 2-Fold 均等划分 ---
    idx_A = rng.choice(n_inf, size=n_inf // 2, replace=False)
    mask_A = np.zeros(n_inf, dtype=bool)
    mask_A[idx_A] = True

    dt_A = dt_inf[mask_A].reset_index(drop=True)
    dt_B = dt_inf[~mask_A].reset_index(drop=True)

    # --- Step 3: Cross-Fitting 计算 (互为推断集) ---
    res_B = compute_fold_theta(dt_train=dt_A, dt_target=dt_B, tau=tau)
    res_A = compute_fold_theta(dt_train=dt_B, dt_target=dt_A, tau=tau)

    # --- Step 4: 加权融合 ---
    w_A = len(dt_A) / n_inf
    w_B = len(dt_B) / n_inf

    est_ora   = w_A * res_A["ora"]   + w_B * res_B["ora"]
    est_naive = w_A * res_A["naive"] + w_B * res_B["naive"]
    est_class = w_A * res_A["class"] + w_B * res_B["class"]
    est_ppi   = w_A * res_A["ppi"]   + w_B * res_B["ppi"]

    return np.array(
        [
            est_ora[0], est_class[0], est_naive[0], est_ppi[0],
            est_ora[1], est_class[1], est_naive[1], est_ppi[1],
        ],
        dtype=np.float64,
    )

# ==============================================================================
# 5. Execution Pipeline
# ==============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2-Fold Bivariate survival IPCW/DR/PPI with LimiX imputing X2|X1",
    )
    p.add_argument("--m", type=int, default=20, help="Monte Carlo replicates")
    p.add_argument("--seed-gold", type=int, default=999, help="Random seed for theta0")
    p.add_argument("--theta-n", type=int, default=10_000_000, help="N for E[T_true]")
    p.add_argument("--n-pre", type=int, default=500)
    p.add_argument("--n-inf", type=int, default=10000)
    p.add_argument("--p-label", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=2)
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--inference_config", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--predictor-seed", type=int, default=0)
    p.add_argument("--base-seed", type=int, default=2024)
    return p.parse_args()


def resolve_limiX_device(device_arg: str) -> torch.device:
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
    theta0, censor_rate = truth_theta0_rmst(rng_g, args.theta_n, args.tau)
    print("theta0 (E[T_true]):", theta0, "Censor rate:", censor_rate, flush=True)

    ck = load_or_fetch_ckpt((args.model_path or None) if args.model_path else None, root)
    print("Checkpoint:", ck, flush=True)
    icfg = args.inference_config or os.path.join(root, "config", "reg_default_noretrieval_MVI.json")
    
    if not os.path.isfile(icfg):
        raise SystemExit(f"Inference config not found: {icfg}. Chdir to LimiX/ or set --inference_config.")

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
        "IPCW_Oracle", "IPCW_Classical", "IPCW_Naive", "IPCW_PPI",
        "DR_Oracle", "DR_Classical", "DR_Naive", "DR_PPI",
    ]
    res = np.zeros((M, len(names)), dtype=np.float64)
    
    print("\nRunning 2-Fold Limix + STRICT Single Model Simulation...\n", flush=True)
    
    for m2 in range(1, M + 1):
        print(f"Replicate {m2}/{M} ...", flush=True)
        res[m2 - 1, :] = run_single_sim(
            seed=int(args.base_seed) + m2,
            n_pre=args.n_pre,
            n_inf=args.n_inf,
            p_label=args.p_label,
            tau=args.tau,
            predictor=pred,
        )

    bias = res.mean(axis=0) - theta0
    esd = res.std(axis=0, ddof=0)
    rmse = np.sqrt(bias**2 + esd**2)
    
    out = pd.DataFrame(
        {
            "Method": [f"{i+1}. {n}" for i, n in enumerate(names)],
            "Bias": np.round(bias, 6),
            "ESD": np.round(esd, 6),
            "RMSE": np.round(rmse, 6),
        }
    )
    print(f"\n--- Final Unified Simulation Results (M = {M}) ---\n")
    print(f"Missing rate: {1 - args.p_label}")
    print(f"Censor rate: {censor_rate}")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()