from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from limix_imputer import LimixX2Imputer
from ppi_ipcw import estimate_theta_ppi_ipcw
from sc_estimation import estimate_sc_stratified_crossfit
from simulate_data import generate_full_data, split_pretrain_inference

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "cache", "LimiX-16M.ckpt"))


@dataclass
class MethodSummary:
    method: str
    mse: float
    sd: float
    bias: float


def _compute_true_rmst(seed: int = 233, n_true: int = 50000) -> float:
    data_true = generate_full_data(n=n_true, seed=seed)
    return float(data_true["T"].mean())


def run_simulation(
    b: int = 100,
    n_pretrain: int = 500,
    n_infer: int = 10000,
    obs_rate: float = 0.1,
    n_splits_full: int = 2,
    n_splits_obs: int = 2,
    use_sample_pi: bool = True,
    fixed_pi: float = 0.1,
    random_state: int = 0,
    model_path: str | None = None,
    print_imputation_preview: bool = False,
    preview_rows: int = 5,
) -> Dict[str, object]:
    print("Computing true RMST...")
    true_rmst = _compute_true_rmst(seed=233, n_true=50000)

    imputer = LimixX2Imputer(model_path=model_path)

    methods = ["rsf"]
    theta_store: Dict[str, List[float]] = {m: [] for m in methods}
    censor_rate_list: List[float] = []
    obs_rate_list: List[float] = []

    for iter_idx in range(1, b + 1):
        print(f"Iteration: {iter_idx}/{b}")
        seed_iter = random_state + 1000 + iter_idx
        full_data = generate_full_data(n=n_pretrain + n_infer, seed=seed_iter)
        split = split_pretrain_inference(
            full_data=full_data,
            n_pretrain=n_pretrain,
            n_infer=n_infer,
            obs_rate=obs_rate,
            seed=seed_iter + 77,
        )

        infer_df = split.inferencing
        censor_rate_list.append(float(1.0 - infer_df["Delta"].mean()))
        obs_rate_list.append(float(infer_df["R"].mean()))

        impute_result = imputer.impute(pretraining=split.pretraining, inferencing=infer_df)
        infer_imputed = impute_result.inferencing_with_imputation

        if print_imputation_preview:
            before_cols = ["X1", "X2", "X3", "X4", "X5", "R", "X4_input", "X5_input", "Y", "Delta"]
            after_cols = ["X1", "X2", "X3", "X4", "X5", "X4_hat", "X5_hat", "R", "Y", "Delta"]
            print(f"\n[Iteration {iter_idx}] Inferencing before LimiX imputation (head={preview_rows})")
            print(infer_df[before_cols].head(preview_rows).to_string(index=False))
            print(f"\n[Iteration {iter_idx}] Inferencing after LimiX imputation (head={preview_rows})")
            print(infer_imputed[after_cols].head(preview_rows).to_string(index=False))
            print("")

        for method in methods:
            sc_result = estimate_sc_stratified_crossfit(
                inferencing_df=infer_imputed,
                model_type=method,  # type: ignore[arg-type]
                n_splits_full=n_splits_full,
                n_splits_obs=n_splits_obs,
                random_state=seed_iter + (11 if method == "rsf" else 29),
            )
            theta_hat = estimate_theta_ppi_ipcw(
                inferencing_df=infer_imputed,
                sc_imputed_full=sc_result.sc_imputed_full,
                sc_observed_complete=sc_result.sc_observed_complete,
                use_sample_pi=use_sample_pi,
                fixed_pi=fixed_pi,
            )
            theta_store[method].append(theta_hat)

    summary_list: List[MethodSummary] = []
    for method in methods:
        arr = np.asarray(theta_store[method], dtype=float)
        summary_list.append(
            MethodSummary(
                method=method,
                mse=float(np.mean((arr - true_rmst) ** 2)),
                sd=float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0),
                bias=float(np.mean(arr - true_rmst)),
            )
        )

    print("\n===== Simulation Results =====")
    print(f"n_pretrain: {n_pretrain}")
    print(f"n_infer: {n_infer}")
    print(f"B: {b}")
    print(f"True RMST: {true_rmst:.6f}")
    print(f"Avg Censor Rate: {np.mean(censor_rate_list):.6f}")
    print(f"Avg Observed Rate (R=1): {np.mean(obs_rate_list):.6f}")
    for s in summary_list:
        print(f"[{s.method}] MSE={s.mse:.6f}, SD={s.sd:.6f}, Bias={s.bias:.6f}")

    out = {
        "true_rmst": true_rmst,
        "theta_estimates": theta_store,
        "avg_censor_rate": float(np.mean(censor_rate_list)),
        "avg_observed_rate": float(np.mean(obs_rate_list)),
        "summary": [s.__dict__ for s in summary_list],
    }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPI-IPCW simulation with Limix imputation")
    parser.add_argument("--B", type=int, default=100)
    parser.add_argument("--n_pretrain", type=int, default=250)
    parser.add_argument("--n_infer", type=int, default=5000)
    parser.add_argument("--obs_rate", type=float, default=0.1)
    parser.add_argument("--n_splits_full", type=int, default=2)
    parser.add_argument("--n_splits_obs", type=int, default=2)
    parser.add_argument("--fixed_pi", type=float, default=0.1)
    parser.add_argument("--use_sample_pi", action="store_true")
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument("--print_imputation_preview", action="store_true")
    parser.add_argument("--preview_rows", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_simulation(
        b=args.B,
        n_pretrain=args.n_pretrain,
        n_infer=args.n_infer,
        obs_rate=args.obs_rate,
        n_splits_full=args.n_splits_full,
        n_splits_obs=args.n_splits_obs,
        use_sample_pi=args.use_sample_pi,
        fixed_pi=args.fixed_pi,
        random_state=args.random_state,
        model_path=args.model_path,
        print_imputation_preview=args.print_imputation_preview,
        preview_rows=args.preview_rows,
    )
    if args.save_json:
        out_path = os.path.abspath(args.save_json)
        pd.Series(result).to_json(out_path, force_ascii=False, indent=2)
        print(f"Saved results to: {out_path}")

