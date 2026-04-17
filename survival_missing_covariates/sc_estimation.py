from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from scipy.linalg import LinAlgError


ModelType = Literal["rsf", "cox"]


def _build_surv_target(df: pd.DataFrame) -> np.ndarray:
    # Censoring survival target uses event = 1 - Delta.
    event = (1 - df["Delta"].to_numpy(dtype=int)).astype(bool)
    time = df["Y"].to_numpy(dtype=float)
    return Surv.from_arrays(event=event, time=time)


def _fit_model(model_type: ModelType, x_train: np.ndarray, y_train: np.ndarray, random_state: int) -> object:
    if model_type == "rsf":
        model = RandomSurvivalForest(
            n_estimators=500,
            min_samples_leaf=15,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        return model
    elif model_type == "cox":
        # Small folds can be nearly singular; progressively increase ridge penalty.
        for alpha in (0.0, 1e-6, 1e-4, 1e-2):
            model = CoxPHSurvivalAnalysis(alpha=alpha)
            try:
                model.fit(x_train, y_train)
                return model
            except LinAlgError:
                continue
        # Last attempt with stronger regularization to avoid fold-level crashes.
        model = CoxPHSurvivalAnalysis(alpha=1e-1)
        model.fit(x_train, y_train)
        return model
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def _predict_sc_at_y(model: object, x_test: np.ndarray, y_times: np.ndarray) -> np.ndarray:
    survival_functions = model.predict_survival_function(x_test)
    sc = np.empty(len(y_times), dtype=float)
    for i, (sf, t_i) in enumerate(zip(survival_functions, y_times)):
        x_axis = sf.x
        y_axis = sf.y
        if t_i < x_axis[0]:
            sc[i] = 1.0
        elif t_i > x_axis[-1]:
            sc[i] = float(y_axis[-1])
        else:
            sc[i] = float(sf(t_i))
    return sc


def cross_fit_sc_at_y(
    df: pd.DataFrame,
    covariates: Iterable[str],
    model_type: ModelType,
    n_splits: int = 2,
    random_state: int = 0,
    min_sc: float = 1e-5,
) -> np.ndarray:
    """Cross-fitted S^C(Y|X) for all rows in df."""
    x = df[list(covariates)].to_numpy(dtype=float)
    y_surv = _build_surv_target(df)
    y_time = df["Y"].to_numpy(dtype=float)

    sc_hat = np.empty(len(df), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_id, (train_idx, test_idx) in enumerate(kf.split(x)):
        model = _fit_model(
            model_type=model_type,
            x_train=x[train_idx],
            y_train=y_surv[train_idx],
            random_state=random_state + fold_id + 1,
        )
        sc_hat[test_idx] = _predict_sc_at_y(model, x[test_idx], y_time[test_idx])
    return np.clip(sc_hat, min_sc, 1.0)


@dataclass
class StratifiedSCResult:
    sc_imputed_full: np.ndarray
    sc_observed_complete: np.ndarray
    observed_indices: np.ndarray


def estimate_sc_stratified_crossfit(
    inferencing_df: pd.DataFrame,
    model_type: ModelType,
    n_splits_full: int = 2,
    n_splits_obs: int = 2,
    random_state: int = 0,
    min_sc: float = 1e-5,
) -> StratifiedSCResult:
    """
    Two-channel cross-fitting:
    1) imputed_full channel on all inferencing samples for S^C(Y|X1,X2_hat)
    2) observed_complete channel on R=1 subset for S^C(Y|X1,X2)
    """
    cov_imputed = ["X1", "X2", "X3", "X4_hat", "X5_hat"]
    sc_imputed_full = cross_fit_sc_at_y(
        inferencing_df,
        covariates=cov_imputed,
        model_type=model_type,
        n_splits=n_splits_full,
        random_state=random_state,
        min_sc=min_sc,
    )

    observed_mask = inferencing_df["R"].to_numpy(dtype=int) == 1
    observed_indices = np.where(observed_mask)[0]
    sc_observed_complete = np.full(len(inferencing_df), np.nan, dtype=float)

    if len(observed_indices) < 2:
        raise ValueError("Not enough complete observed samples (R=1) for cross-fitting.")

    obs_df = inferencing_df.iloc[observed_indices].reset_index(drop=True)
    cov_observed = ["X1", "X2", "X3", "X4", "X5"]
    n_splits_obs_eff = min(n_splits_obs, len(obs_df))
    if n_splits_obs_eff < 2:
        n_splits_obs_eff = 2
    sc_obs_local = cross_fit_sc_at_y(
        obs_df,
        covariates=cov_observed,
        model_type=model_type,
        n_splits=n_splits_obs_eff,
        random_state=random_state + 1000,
        min_sc=min_sc,
    )
    sc_observed_complete[observed_indices] = sc_obs_local

    return StratifiedSCResult(
        sc_imputed_full=sc_imputed_full,
        sc_observed_complete=sc_observed_complete,
        observed_indices=observed_indices,
    )

