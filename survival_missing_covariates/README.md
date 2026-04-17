# survival_missing_covariates

该目录实现了你给出的 R 版模拟流程在 Python 下的迁移版本，并接入 LimiX 进行缺失协变量填补，再通过 PPI-IPCW 估计 RMST（`E[T]`）。

## 实现内容

- 数据生成：复刻原始 AFT + Cox 删失机制（`simulate_data.py`）。
- 缺失填补：使用 LimiX 对 inferencing 全体样本预测 `X4_hat, X5_hat`（`limix_imputer.py`）。
- nuisance 函数估计：`RSF` 和 `CoxPH` 两种模型（`sc_estimation.py`）。
- 分层 cross-fitting：
  - `S^C(Y|X1, X2_hat)` 在 inferencing 全样本（10000）上 cross-fitting；
  - `S^C(Y|X1, X2)` 仅在 `R=1` 完整样本（约1000）上 cross-fitting。
- 估计器：按闭式公式实现 PPI-IPCW（`ppi_ipcw.py`）。
- Monte Carlo 主程序：输出 MSE / SD / Bias / 删失率（`run_simulation.py`）。

## 目录文件

- `simulate_data.py`: 生成完整数据与 pretrain/infer 切分，构造 `R` 和输入缺失列。
- `limix_imputer.py`: 调用 `LimiXPredictor(mask_prediction=True)` 做 `X4/X5` 填补。
- `sc_estimation.py`: RSF/Cox 的 `S^C` 估计与双通道 cross-fitting。
- `ppi_ipcw.py`: PPI-IPCW 估计函数。
- `run_simulation.py`: 端到端实验运行入口。

## 安装依赖

```bash
pip install -r survival_missing_covariates/requirements.txt
```

## 运行示例

先做冒烟测试：

```bash
python survival_missing_covariates/run_simulation.py --B 2 --n_pretrain 500 --n_infer 1000 --obs_rate 0.1 --use_sample_pi
```

完整设定（耗时较长）：

```bash
python survival_missing_covariates/run_simulation.py --B 100 --n_pretrain 500 --n_infer 10000 --obs_rate 0.1 --use_sample_pi
```

## 参数说明

- `--B`: Monte Carlo 重复次数。
- `--n_pretrain`: LimiX 训练样本量（默认 500）。
- `--n_infer`: inferencing 样本量（默认 10000）。
- `--obs_rate`: inferencing 中 `R=1` 完整观测比例（默认 0.1）。
- `--n_splits_full`: 全样本通道 cross-fitting 折数。
- `--n_splits_obs`: 完整样本通道 cross-fitting 折数。
- `--use_sample_pi`: 使用样本内 `pi_n=mean(R)`；不加该参数时用 `--fixed_pi`。

## 备注

- 当前 `sc_estimation.py` 中 RSF 参数与 R 脚本近似对齐：`n_estimators=500`, `min_samples_leaf=15`。
- LimiX 模型默认从 HuggingFace 下载 `LimiX-16M.ckpt`，也可通过 `--model_path` 指定本地权重。

