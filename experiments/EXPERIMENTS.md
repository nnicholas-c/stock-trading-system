# Legacy Experiments

These scripts were moved here from the repository root during the honest rewrite. They are retained as research history, not as validated production training pipelines.

## Inventory

| File | Role | Main Caveat |
| --- | --- | --- |
| `train_v3.py` | LSTM/XGB/LGB/PPO ensemble experiment | Full-series scaling before splits and several model families tried. |
| `train_v4.py` | Larger supervised ensemble | Full-series scaling, full-sample quantiles, and stacked predictions that contaminate validation. |
| `train_v5_quant.py` | Factor/news/self-improvement experiment | In-sample retrain MAE and repeated adaptive checks. |
| `train_v6_micro.py` | Microstructure simulation/research | Useful research code, not a broad validated headline. |
| `train_v7_deep.py` | Literature-inspired synthetic/deep quant system | Synthetic OHLCV and literature claims mixed with model outputs. |
| `train_v8_finetune.py` | Iterative supervised fine-tuning | Reuses validation outcomes to adjust weights and rerun predictions. |
| `train_v9_xgb.py` | XGB/LGB monthly walk-forward ensemble | Iterative loop selects the best run on the same validation story. |
| `train_drl_v1.py` | DRL/XGB high-confidence signal gating | Skips HOLD predictions and optimizes high-confidence accuracy. |
| `train_drl_v2.py` | Later DRL refresh flow | Historical experiment; use only with explicit audit of data and costs. |
| `train_pltr_deep.py` | PLTR-specific deep model | Single-ticker selection and local artifact dependencies. |
| `train_pltr_ultra.py` | PLTR-only multi-agent DRL | Schedule search and selected-signal reporting. |
| `self_improve.py` | Incremental retrain helper | Fits scaler on full data before holdout split. |

## How To Treat These Files

- Do not cite their headline accuracy, Sharpe, or return numbers as validated performance.
- Do use them as examples of iterative research, model exploration, and failure modes.
- If a script is rerun, record the exact command, data source, date range, universe, configuration count, and generated artifact.
- Prefer `honest_backtest/run.py` for any public-facing performance claim.

## Multiple Testing Note

The old experiment suite contains many model families, thresholds, schedules, and retraining loops. A conservative constructor scan found 41 explicit model constructors; a broader scan found 142 configuration/search-related lines. This is enough trial volume that unadjusted best-run metrics are expected to be optimistic.
