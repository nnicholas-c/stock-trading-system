# AXIOM - Quant Research Backtesting Portfolio

AXIOM is a quantitative research portfolio project focused on equity signal design, backtest integrity, and model risk controls. The project documents an end-to-end research workflow: hypothesis formation, feature construction, walk-forward validation, leakage auditing, baseline comparison, cost modeling, and honest reporting of weak or negative results.

The central research question is simple:

> Can daily price and volume features produce an economically meaningful out-of-sample equity signal after execution costs and multiple-testing adjustment?

Current answer: not convincingly. The fixed validation harness finds a small statistical signal, but it is not economically attractive after costs, baselines, and deflated Sharpe adjustment. That is the point of the project: it shows research discipline rather than a polished performance claim.

## Quant Research Skills Demonstrated

- Leakage-aware feature engineering and target construction.
- Purged and embargoed walk-forward cross-validation.
- Train-fold-only preprocessing with sklearn pipelines.
- Explicit execution lag, transaction costs, and slippage.
- Buy-hold and random baselines.
- Information coefficient, t-stat, hit rate, turnover, out-of-sample Sharpe, drawdown, and deflated Sharpe reporting.
- Post-research audit of model-selection bias, survivorship bias, and overfit experimental history.
- Reproducible artifacts and command-line research workflow.

## Research Design

The current headline result comes from `honest_backtest/run.py`, a deliberately fixed-specification backtest:

- Universe: 60 median tradable names from a current S&P 500 yfinance download.
- Features: causal daily OHLCV-derived returns, volatility, moving-average distances, RSI, volume z-score, and intraday range.
- Label/execution: decide at `close[t]`, enter at `open[t+1]`, exit at `close[t+1]`.
- Model: `StandardScaler` plus `LogisticRegression`, fit only inside each training fold.
- Validation: walk-forward folds with 504-day minimum train window, 63-day validation windows, 5-day purge, and 5-day embargo.
- Costs: 5 bps transaction cost plus 5 bps slippage per side.
- Baselines: equal-weight buy-hold and a zero-skill random strategy with comparable active rate.
- Multiple testing: deflated Sharpe adjustment using a lower-bound count of 41 prior model/configuration trials found in the repository.

Run:

```bash
.venv/bin/python -m honest_backtest.run --max-tickers 60 --min-tickers 50
```

## Headline Result

Latest reproducible output: `honest_backtest/results/summary_metrics.csv`

| Method | Total return | Ann. Sharpe | Max drawdown | Notes |
| --- | ---: | ---: | ---: | --- |
| Fixed logistic signal, net | +1.84% | 0.097 | -10.21% | 90,713 predictions, 60 median tickers |
| Equal-weight buy-hold | +64.20% | 0.646 | -14.95% | Same universe and dates |
| Zero-skill random, net | -15.74% | -3.499 | -16.42% | Same active rate and cost model |

Signal diagnostics:

| Metric | Value |
| --- | ---: |
| Spearman IC | 0.00728 |
| IC t-stat | 2.19 |
| Active-signal hit rate | 53.47% |
| Average active fraction | 5.36% |
| Average daily turnover | 10.72% |
| Config trials lower bound | 41 |
| Deflated Sharpe threshold | 0.805 |
| Deflated Sharpe | -0.707 |
| DSR probability | 1.79e-235 |

Interpretation: the signal has a tiny positive IC, but the realized strategy is not competitive with the passive baseline and does not survive a multiple-testing-aware Sharpe adjustment. This is a useful research result because it prevents overstating an overfit strategy.

## Repository Guide

```text
honest_backtest/
  run.py                    Fixed-specification validation harness
  results/                  Reproducible result CSVs and figures

LEAKAGE_AUDIT.md            Detailed model-risk and leakage audit
SECURITY_FINDINGS.md        Secret-scan and repository hygiene notes

experiments/
  train_v3.py ... train_v9_xgb.py
  train_drl_v1.py, train_drl_v2.py
  train_pltr_deep.py, train_pltr_ultra.py
  self_improve.py
  EXPERIMENTS.md            Historical experiment inventory

research/
  pipeline.py               Earlier research pipeline with stronger controls than the first experiments

backend/, docs/, ios/       Supporting app surfaces retained for project context
```

## Historical Experiments

The `experiments/` directory contains prior supervised learning, deep learning, reinforcement learning, and PLTR-specific research attempts. Those files are kept because they show the research path and the types of model risk that can enter an iterative project.

They should not be read as validated performance evidence. The detailed audit in `LEAKAGE_AUDIT.md` documents full-series scaling, same-bar execution assumptions, narrow ticker selection, confidence-gated metrics, and repeated configuration search.

## Run Locally

```bash
make setup
.venv/bin/python -m honest_backtest.run --max-tickers 60 --min-tickers 50
python -m pytest -q
```

Backtest outputs:

- `honest_backtest/results/summary_metrics.csv`
- `honest_backtest/results/fold_predictions.csv`
- `honest_backtest/results/daily_returns.csv`
- `honest_backtest/results/universe.csv`
- `honest_backtest/results/equity_curves.png`
- `honest_backtest/results/score_vs_return.png`
- `honest_backtest/results/run_config.json`

## Limitations

- Current S&P 500 membership is used, so survivorship bias is reduced relative to four hand-picked tickers but not eliminated.
- yfinance OHLCV is transparent and reproducible, but it is not institutional point-in-time market data.
- The strategy uses daily bars only; it does not model intraday queue position, borrow, financing, tax, or capacity constraints.
- The deflated Sharpe calculation is an approximation using a conservative lower-bound trial count.
- This is a research portfolio project, not a production trading system.

## Disclaimer

This repository is for education and research only. It is not financial advice, not an investment recommendation, and not a live trading system.
