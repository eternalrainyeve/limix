from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


FEATURE_COLS = ["X1", "X2", "X3", "X4", "X5"]
X1_COLS = ["X1", "X2", "X3"]
X2_COLS = ["X4", "X5"]


def generate_full_data(n: int, seed: Optional[int] = None) -> pd.DataFrame:
    """Reproduce the R data-generating process for survival simulation."""
    rng = np.random.default_rng(seed)

    x = rng.uniform(0.0, 1.0, size=(n, 5))

    epsilon = rng.normal(0.0, 1.0, size=n)
    log_t = -1.6 + x[:, 0] - 0.5 * x[:, 1] + x[:, 2] - x[:, 3] + 2.0 * x[:, 4] + epsilon
    t = np.exp(log_t)
    tau = 2.5
    t = np.minimum(t, tau)

    eta = -1.0 + x[:, 0] + 2.0 * x[:, 1] - x[:, 2] + 0.5 * x[:, 3] - x[:, 4]
    u = rng.uniform(0.0, 1.0, size=n)
    c = np.sqrt(-np.log(u) / np.exp(eta))

    y = np.minimum(t, c)
    delta = (t <= c).astype(int)

    return pd.DataFrame(
        {
            "X1": x[:, 0],
            "X2": x[:, 1],
            "X3": x[:, 2],
            "X4": x[:, 3],
            "X5": x[:, 4],
            "T": t,
            "C": c,
            "Y": y,
            "Delta": delta,
        }
    )


@dataclass
class SimulationSplit:
    pretraining: pd.DataFrame
    inferencing: pd.DataFrame


def split_pretrain_inference(
    full_data: pd.DataFrame,
    n_pretrain: int,
    n_infer: int,
    obs_rate: float = 0.1,
    seed: Optional[int] = None,
) -> SimulationSplit:
    """Split full data into pretraining and inferencing blocks and create R indicator."""
    if len(full_data) < n_pretrain + n_infer:
        raise ValueError("full_data does not contain enough rows for requested split sizes.")

    rng = np.random.default_rng(seed)
    data = full_data.sample(n=n_pretrain + n_infer, replace=False, random_state=seed).reset_index(drop=True)

    pretraining = data.iloc[:n_pretrain].copy().reset_index(drop=True)
    inferencing = data.iloc[n_pretrain : n_pretrain + n_infer].copy().reset_index(drop=True)

    r = np.zeros(n_infer, dtype=int)
    n_observed = max(1, int(round(obs_rate * n_infer)))
    observed_idx = rng.choice(n_infer, size=n_observed, replace=False)
    r[observed_idx] = 1
    inferencing["R"] = r

    # For PPI-IPCW with predicted X2 in all inferencing samples, always mask X4/X5 inputs.
    # True X4/X5 are kept in original columns for the R=1 correction term only.
    inferencing["X4_input"] = np.nan
    inferencing["X5_input"] = np.nan

    return SimulationSplit(pretraining=pretraining, inferencing=inferencing)

