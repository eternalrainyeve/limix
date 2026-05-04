"""
Bivariate survival simulation: IPCW, doubly robust, and PPI-Old (Ablation Test)
2-Fold Cross-Fitting + SEPARATE NUISANCE MODELS FOR EVERY TERM

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
from lifelines import CoxPHFitter

import numpy as np
import pandas as pd
import torch


# LimiX package root (this file: LimiX/examples/...)
_LIMIX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _LIMIX_ROOT not in sys.path:
    sys.path.insert(0, _LIMIX_ROOT)
from inference.predictor import LimiXPredictor  # noqa: E402


# ==============================================================================
# 1. 数据生成函数 (DGP)
# ==============================================================================
def generate_data(
    n: int,
    rng: np.random.Generator,
    rho: float = 0.7,
    tau: float = 2,
    c_base_hazard: float = 1.35
) -> pd.DataFrame:
    """Explicit Cox PH DGP using Inverse Probability Integral Transform"""
    # 1. 生成协变量
    cov = np.array([[1.0, rho], [rho, 1.0]])
    x1, x2 = rng.multivariate_normal(np.zeros(2), cov, size=n).T

    # 2. 显式 Cox PH 生成真实事件时间 T
    beta_t = np.array([-0.5, 1.5])
    risk_score_t = np.exp(beta_t[0] * x1 + beta_t[1] * x2)
    
    nu_t = 2.0  # shape     
    lambda_t = 1.0  # scale
    u_t = rng.random(n) 
    
    t_true = (-np.log(u_t) / (lambda_t * risk_score_t)) ** (1.0 / nu_t)
    t_trunc = np.minimum(t_true, tau)
    # 算并打印 85% 分位数（基于 t_true）
    q85 = np.quantile(t_true, 0.85)
    print(f"85th percentile of t_true: {q85:.4f}")

    # 3. 显式 Cox PH 生成删失时间 C
    beta_c = np.array([0.2, 0.4])
    risk_score_c = np.exp(beta_c[0] * x1 + beta_c[1] * x2)
    
    nu_c = 1.0
    lambda_c = c_base_hazard  
    u_c = rng.random(n)
    
    c = (-np.log(u_c) / (lambda_c * risk_score_c)) ** (1.0 / nu_c)
    q85_c = np.quantile(c, 0.85)
    print(f"85th percentile of c: {q85_c:.4f}")

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
# 2. 核心辅助函数：极速计算 IPCW 得分与 DR-CUT 得分
# ==============================================================================
def _cumulative_baseline_on_grid(
    cph: CoxPHFitter, grid_t: np.ndarray
) -> np.ndarray:
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

def _match_r_find_interval(
    y: np.ndarray, grid_t: np.ndarray, k: int
) -> np.ndarray:
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
    
    return {"ipcw": term1, "dr": term1 + term2 - term3}

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

def limix_impute_x2(
    pre: pd.DataFrame, inf: pd.DataFrame, predictor: LimiXPredictor
) -> np.ndarray:
    x1_tr = pre[["X1"]].to_numpy(np.float64)
    x2_tr = pre[["X2"]].to_numpy(np.float64).ravel()
    x1_te = inf[["X1"]].to_numpy(np.float64)
    x2_hat = predictor.predict(x1_tr, x2_tr, x1_te, task_type="Regression")
    return x2_hat.to('cpu').numpy().ravel()


# ==============================================================================
# 3. 核心单向推断函数 (Cross-Fitting 的单折逻辑：训练与分离)
# ==============================================================================
def compute_fold_theta_separate(
    dt_train: pd.DataFrame, dt_target: pd.DataFrame, tau: float
) -> dict:
    
    x_cols = ["X1", "X2"]
    
    # 布尔索引
    mask_r1_train = dt_train["R"] == 1.0
    mask_r0_train = dt_train["R"] == 0.0
    
    mask_r1_target = dt_target["R"] == 1.0
    mask_r0_target = dt_target["R"] == 0.0
    
    n_target = len(dt_target)
    n_r0_target = mask_r0_target.sum()
    
    # -------------------------------------------------------------------------
    # Baseline 1: Oracle
    # -------------------------------------------------------------------------
    cox_t_ora = fit_cox_t(dt_train, x_cols)
    cox_c_ora = fit_cox_c(dt_train, x_cols)
    res_ora = compute_scores(dt_target, cox_t_ora, cox_c_ora, tau, x_cols)
    theta_ipcw_ora = res_ora["ipcw"].mean()
    theta_dr_ora = res_ora["dr"].mean()
    
    # -------------------------------------------------------------------------
    # Baseline 2: Naive Impute
    # -------------------------------------------------------------------------
    # 使用 Proxy 技巧：将真实 X2 替换为 Xhat2
    dt_train_proxy = dt_train.copy()
    dt_train_proxy["X2"] = dt_train["Xhat2"]
    dt_target_proxy = dt_target.copy()
    dt_target_proxy["X2"] = dt_target["Xhat2"]
    
    cox_t_naive = fit_cox_t(dt_train_proxy, x_cols)
    cox_c_naive = fit_cox_c(dt_train_proxy, x_cols)
    res_naive = compute_scores(dt_target_proxy, cox_t_naive, cox_c_naive, tau, x_cols)
    theta_ipcw_naive = res_naive["ipcw"].mean()
    theta_dr_naive = res_naive["dr"].mean()
    
    # =========================================================================
    # 核心逻辑：为 PPI 的三项分别拟合完全独立的 Nuisance 模型
    # =========================================================================
    
    # --- Component 1: R=0 样本 ---
    if mask_r0_train.sum() > 0 and mask_r0_target.sum() > 0:
        cox_t_r0 = fit_cox_t(dt_train_proxy[mask_r0_train], x_cols)
        cox_c_r0 = fit_cox_c(dt_train_proxy[mask_r0_train], x_cols)
        res_t1 = compute_scores(dt_target_proxy[mask_r0_target], cox_t_r0, cox_c_r0, tau, x_cols)
        ppi_t1_ipcw = res_t1["ipcw"].mean()
        ppi_t1_dr = res_t1["dr"].mean()
    else:
        ppi_t1_ipcw, ppi_t1_dr = 0.0, 0.0
        
    # --- Component 2: R=1 带预测值 Xhat2 ---
    if mask_r1_train.sum() > 0 and mask_r1_target.sum() > 0:
        cox_t_r1_hat = fit_cox_t(dt_train_proxy[mask_r1_train], x_cols)
        cox_c_r1_hat = fit_cox_c(dt_train_proxy[mask_r1_train], x_cols)
        res_t2 = compute_scores(dt_target_proxy[mask_r1_target], cox_t_r1_hat, cox_c_r1_hat, tau, x_cols)
        ppi_t2_ipcw = res_t2["ipcw"].mean()
        ppi_t2_dr = res_t2["dr"].mean()
    else:
        ppi_t2_ipcw, ppi_t2_dr = 0.0, 0.0
        
    # --- Component 3: R=1 带真实值 X2 (Classical) ---
    if mask_r1_train.sum() > 0 and mask_r1_target.sum() > 0:
        cox_t_cla = fit_cox_t(dt_train[mask_r1_train], x_cols)
        cox_c_cla = fit_cox_c(dt_train[mask_r1_train], x_cols)
        res_cla = compute_scores(dt_target[mask_r1_target], cox_t_cla, cox_c_cla, tau, x_cols)
        theta_ipcw_class = res_cla["ipcw"].mean()
        theta_dr_class = res_cla["dr"].mean()
    else:
        theta_ipcw_class, theta_dr_class = 0.0, 0.0
        
    # -------------------------------------------------------------------------
    # 组装 PPI 估计量 (在目标集 target 上进行加权)
    # -------------------------------------------------------------------------
    w_0 = n_r0_target / n_target if n_target > 0 else 0.0
    
    theta_ipcw_ppi = w_0 * (ppi_t1_ipcw - ppi_t2_ipcw) + theta_ipcw_class
    theta_dr_ppi = w_0 * (ppi_t1_dr - ppi_t2_dr) + theta_dr_class
    
    return {
        "ora": np.array([theta_ipcw_ora, theta_dr_ora]),
        "naive": np.array([theta_ipcw_naive, theta_dr_naive]),
        "class_": np.array([theta_ipcw_class, theta_dr_class]),
        "ppi": np.array([theta_ipcw_ppi, theta_dr_ppi]),
    }

# ==============================================================================
# 4. 单次模拟函数 (包含 2-Fold 划分与组合)
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
    
    # --- Step 1: ML 预测模型 ---
    pre = generate_data(n_pre, rng, tau=tau)
    dt_inf = generate_data(n_inf, rng, tau=tau)
    r = rng.binomial(1, p_label, n_inf).astype(np.float64)
    
    x2_hat = limix_impute_x2(pre, dt_inf, predictor)

    dt_inf["Xhat2"] = x2_hat
    dt_inf["R"] = r

    # --- Step 2: 将推断集随机均匀分为 A 和 B 两折 ---
    idx_perm = rng.permutation(n_inf)
    split_point = n_inf // 2
    idx_A = idx_perm[:split_point]
    idx_B = idx_perm[split_point:]

    dt_A = dt_inf.iloc[idx_A].copy()
    dt_B = dt_inf.iloc[idx_B].copy()

    n_A = len(dt_A)
    n_B = len(dt_B)

    # --- Step 3: 交叉拟合计算 ---
    # 方向 1: 用 A 作为训练集建立三个独立模型，对 B 进行对应预测和 PPI 聚合
    res_B = compute_fold_theta_separate(dt_A, dt_B, tau)

    # 方向 2: 用 B 作为训练集建立三个独立模型，对 A 进行对应预测和 PPI 聚合
    res_A = compute_fold_theta_separate(dt_B, dt_A, tau)

    # --- Step 4: 加权平均融合结果 ---
    w_A = n_A / n_inf
    w_B = n_B / n_inf

    est_ora = w_A * res_A["ora"] + w_B * res_B["ora"]
    est_naive = w_A * res_A["naive"] + w_B * res_B["naive"]
    est_class = w_A * res_A["class_"] + w_B * res_B["class_"]
    est_ppi = w_A * res_A["ppi"] + w_B * res_B["ppi"]

    return np.array(
        [
            est_ora[0],   # IPCW_Oracle
            est_class[0], # IPCW_Classical
            est_naive[0], # IPCW_Naive
            est_ppi[0],   # IPCW_PPI
            est_ora[1],   # DR_Oracle
            est_class[1], # DR_Classical
            est_naive[1], # DR_Naive
            est_ppi[1],   # DR_PPI
        ],
        dtype=np.float64,
    )

# ==============================================================================
# 环境及模型加载
# ==============================================================================
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

def resolve_limiX_device(device_arg: str) -> torch.device:
    s = (device_arg or "auto").strip().lower()
    if s in ("", "auto", "default"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bivariate survival 2-Fold CF + SEPARATE Models")
    p.add_argument("--m", type=int, default=30, help="Monte Carlo replicates")
    p.add_argument("--seed-gold", type=int, default=999)
    p.add_argument("--theta-n", type=int, default=10_000_000)
    p.add_argument("--n-pre", type=int, default=500)
    p.add_argument("--n-inf", type=int, default=10000)
    p.add_argument("--p-label", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=1.3)
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--inference_config", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--predictor-seed", type=int, default=0)
    p.add_argument("--base-seed", type=int, default=2024)
    return p.parse_args()

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
    icfg = args.inference_config or os.path.join(root, "config", "reg_default_noretrieval_MVI.json")
    
    dev = resolve_limiX_device(args.device)
    pred = LimiXPredictor(
        device=dev,
        model_path=ck,
        inference_config=icfg,
        seed=args.predictor_seed,
    )

    M = max(1, int(args.m))
    names = [
        "IPCW_Oracle",
        "IPCW_Classical",
        "IPCW_Naive",
        "IPCW_PPI",
        "DR_Oracle",
        "DR_Classical",
        "DR_Naive",
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
            predictor=pred,
        )

    bias = res.mean(axis=0) - theta0
    sd = res.std(axis=0, ddof=0)
    rmse = np.sqrt(bias**2 + sd**2)
    
    out = pd.DataFrame({
        "Method": [f"{i+1}. {n}" for i, n in enumerate(names)],
        "Bias": np.round(bias, 6),
        "SD": np.round(sd, 6),
        "RMSE": np.round(rmse, 6),
    })
    
    print(f"\n--- Final Results (2-Fold Limix + Separate Models(cox)) (M = {M}) ---\n")
    print(f"missing rate: {1 - args.p_label}")
    print(f"censor rate: {censor_rate}")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()