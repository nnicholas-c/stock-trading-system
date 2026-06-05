# AXIOM - Equity Signal Research Audit

AXIOM is now framed as a signal-research and backtest-audit portfolio project, not a production trading system and not a hedge-fund-grade product.

The original repository contained several impressive-looking ML/RL training runs for PLTR, AAPL, NVDA, and TSLA. A self-audit found that many of those headline numbers were not reliable because of full-series preprocessing, narrow universe selection, repeated configuration search, same-bar execution assumptions, and post-hoc iteration on the same validation data. See `LEAKAGE_AUDIT.md` for the detailed map.

## Honest Headline Result

The reproducible headline result is the fixed-specification harness in `honest_backtest/`. It uses a broader yfinance universe, causal price-only features, fold-local scaling, purged/embargoed walk-forward validation, next-day execution, explicit transaction cost plus slippage, and buy-hold/random baselines.

Latest checked run:

```bash
.venv/bin/python -m honest_backtest.run --max-tickers 60 --min-tickers 50
```

| Method | Total return | Ann. Sharpe | Max drawdown | Notes |
| --- | ---: | ---: | ---: | --- |
| Fixed logistic signal, net | +1.84% | 0.097 | -10.21% | 90,713 predictions, 60 median tickers |
| Equal-weight buy-hold | +64.20% | 0.646 | -14.95% | Same downloaded universe and dates |
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

That is the point: the honest version finds a tiny signal that is not economically compelling after costs and multiple-testing adjustment. The older larger claims are treated as experimental history.

## Repository Layout

```text
honest_backtest/
  run.py                    Fixed-specification backtest harness
  results/                  Reproducible CSV/PNG outputs from the latest run

experiments/
  train_v3.py ... train_v9_xgb.py
  train_drl_v1.py, train_drl_v2.py
  train_pltr_deep.py, train_pltr_ultra.py
  self_improve.py
  EXPERIMENTS.md            Inventory and caveats for old training sprawl

research/
  pipeline.py               More careful legacy research pipeline

backend/
  app/routers/backtest.py   Backtest endpoint now marks old metrics invalidated
```

## Run The Honest Backtest

```bash
make setup
.venv/bin/python -m honest_backtest.run --max-tickers 60 --min-tickers 50
```

Outputs are written to `honest_backtest/results/`:

- `summary_metrics.csv`
- `fold_predictions.csv`
- `daily_returns.csv`
- `universe.csv`
- `equity_curves.png`
- `score_vs_return.png`
- `run_config.json`

## Run Tests

```bash
python -m pytest -q
```

## What Changed In This Rewrite

- Removed the production-grade and hedge-fund-grade framing.
- Moved exploratory training scripts into `experiments/`.
- Added `LEAKAGE_AUDIT.md` and `SECURITY_FINDINGS.md`.
- Added a reproducible honest backtest with explicit baselines and costs.
- Stopped the backend backtest router from serving old hard-coded or artifact-embedded performance claims as live validation.
- Updated `.gitignore` so future local data, model binaries, and caches are not added accidentally.

## Limitations

- The S&P 500 universe is loaded from current Wikipedia constituents, so survivorship bias is reduced versus four hand-picked tickers but not eliminated.
- yfinance OHLCV is convenient and reproducible enough for a portfolio project, but it is not institutional point-in-time data.
- The deflated Sharpe calculation is an approximation using a conservative lower-bound count of model/configuration trials found in the repo.
- Existing committed legacy artifacts remain as historical outputs. They should not be used as production performance evidence.

## Disclaimer

This repository is for education and research only. It is not financial advice, not an investment recommendation, and not a live trading system.
