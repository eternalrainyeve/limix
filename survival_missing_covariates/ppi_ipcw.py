from __future__ import annotations

import numpy as np
import pandas as pd


def psi_theta(theta: float, y: np.ndarray, delta: np.ndarray, sc: np.ndarray) -> np.ndarray:
    return delta * y / sc - theta


def estimate_theta_ppi_ipcw(
    inferencing_df: pd.DataFrame,
    sc_imputed_full: np.ndarray,
    sc_observed_complete: np.ndarray,
    use_sample_pi: bool = True,
    fixed_pi: float = 0.1,
) -> float:
    """
    Closed-form PPI-IPCW estimator:
      (1/n) sum [ (1-R)/(1-pi_n) * Delta*Y/Sc_hat(X1,X2hat)
                - R/pi_n * (Delta*Y/Sc_hat(X1,X2hat) - Delta*Y/Sc_hat(X1,X2)) ]
    """
    y = inferencing_df["Y"].to_numpy(dtype=float)
    delta = inferencing_df["Delta"].to_numpy(dtype=float)
    r = inferencing_df["R"].to_numpy(dtype=float)
    n = len(inferencing_df)

    pi_n = float(np.mean(r)) if use_sample_pi else float(fixed_pi)
    if not (0.0 < pi_n < 1.0):
        raise ValueError(f"pi_n must lie in (0,1), got {pi_n}")

    term_imputed = delta * y / sc_imputed_full

    term_observed = np.full(n, 0.0, dtype=float)
    obs_mask = r == 1.0
    if np.any(obs_mask):
        if np.any(np.isnan(sc_observed_complete[obs_mask])):
            raise ValueError("sc_observed_complete has NaN on observed-complete (R=1) samples.")
        term_observed[obs_mask] = delta[obs_mask] * y[obs_mask] / sc_observed_complete[obs_mask]

    first = (1.0 - r) / (1.0 - pi_n) * term_imputed
    second = r / pi_n * (term_imputed - term_observed)
    theta_hat = float(np.mean(first - second))
    return theta_hat

