from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.predictor import LimiXPredictor  # noqa: E402


FEATURE_COLS = ["X1", "X2", "X3", "X4", "X5"]  # pre-training features
INPUT_FEATURE_COLS = ["X1", "X2", "X3", "X4_input", "X5_input"]  # inferencing features


@dataclass
class LimixImputeResult:
    inferencing_with_imputation: pd.DataFrame
    reconstructed_all: np.ndarray


class LimixX2Imputer:
    """Use Limix mask prediction to impute X4/X5 on inferencing samples."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        inference_config_path: str = "config/reg_default_noretrieval_MVI.json",
        device: Optional[str] = None,
    ) -> None:
        local_default_model = os.path.join(PROJECT_ROOT, "cache", "LimiX-16M.ckpt")
        if model_path is None:
            if os.path.isfile(local_default_model):
                model_path = local_default_model
            else:
                model_path = hf_hub_download(
                    repo_id="stableai-org/LimiX-16M",
                    filename="LimiX-16M.ckpt",
                    local_dir=os.path.join(PROJECT_ROOT, "cache"),
                )
        else:
            model_path = os.path.abspath(model_path)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.predictor = LimiXPredictor(
            device=torch.device(device),
            model_path=model_path,
            mask_prediction=True,
            inference_config=os.path.join(PROJECT_ROOT, inference_config_path),
        )

    def impute(self, pretraining: pd.DataFrame, inferencing: pd.DataFrame) -> LimixImputeResult:
        x_train = pretraining[FEATURE_COLS].to_numpy(dtype=np.float32)
        # Use constant placeholder labels to avoid injecting outcome information into imputation.
        y_train = np.zeros(len(pretraining), dtype=np.float32)

        x_test = inferencing[INPUT_FEATURE_COLS].copy()
        x_test = x_test.rename(columns={"X4_input": "X4", "X5_input": "X5"})
        x_test_np = x_test.to_numpy(dtype=np.float32)

        _, reconstructed = self.predictor.predict(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test_np,
            task_type="Regression",
        )

        reconstructed_infer = reconstructed[-len(inferencing) :]  # limix输入和输出的行数都是pre-training的行数+inference的行数；这里取inference的部分

        out = inferencing.copy()  # 默认深拷贝，对于pd中的数据，copy是深拷贝，对于np中dataframe
        out["X4_hat"] = reconstructed_infer[:, 3]
        out["X5_hat"] = reconstructed_infer[:, 4]

        return LimixImputeResult(inferencing_with_imputation=out, reconstructed_all=reconstructed)

