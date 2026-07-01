# AXIOM

A quant research project about backtest integrity: building a stock predictor, auditing my own data leakage, and rebuilding it honestly.

This started as an attempt to build a stock-prediction model that worked, and turned into something more useful — a study of how easy it is to convince yourself one does.

The question I kept coming back to is narrow: can daily price and volume data predict next-day equity returns well enough to beat just holding the market, once you pay realistic costs and account for how many things you tried? After building the naive version, auditing it, and rebuilding it carefully, my answer is no — at least not with the features and universe here. The signal that survives a clean evaluation is real but tiny, loses to buy-and-hold, and doesn't clear a multiple-testing-adjusted bar.

I've kept the whole arc in this repo on purpose, including the early models that looked good and were wrong, because the useful part isn't the result — it's why the first version lied.

## Start here

- `honest_backtest/run.py` — the current, fixed-specification backtest. This is the only performance number I stand behind.
- `LEAKAGE_AUDIT.md` — a line-by-line account of the data leakage in the earlier code, and how each issue inflated the results.
- `experiments/` — the earlier attempts, kept as documented anti-examples rather than results. Every problem in them is catalogued in the audit.

## What the honest backtest does

I stripped the project down to one fixed specification so there was nothing left to tune:

- **Universe:** 60 of the more liquid current S&P 500 names (via yfinance).
- **Features:** next-day-safe daily signals only — returns, volatility, distance from moving averages, RSI, a volume z-score, and intraday range.
- **Execution:** decide at today's close, enter at tomorrow's open, exit at tomorrow's close. Nothing trades on information it wouldn't have had at decision time.
- **Model:** a single logistic regression with standardization, fit only inside each training fold.
- **Validation:** walk-forward folds with a 504-day minimum training window and 63-day test windows, with a 5-day purge and 5-day embargo between them.
- **Costs:** 5 bps transaction plus 5 bps slippage per side.
- **Baselines:** equal-weight buy-and-hold, and a zero-skill random strategy trading at the same rate.
- **Multiple testing:** a deflated Sharpe ratio using a conservative count of 41 model/config variations found in the earlier code.

Run it:

```bash
make setup
.venv/bin/python -m honest_backtest.run --max-tickers 60 --min-tickers 50
python -m pytest -q
```

## What it found

Numbers below come straight from `honest_backtest/results/summary_metrics.csv`.

| Method | Total return | Ann. Sharpe | Max drawdown |
| --- | ---: | ---: | ---: |
| Fixed logistic signal, net | +1.84% | 0.097 | -10.21% |
| Equal-weight buy-and-hold | +64.20% | 0.646 | -14.95% |
| Zero-skill random, net | -15.74% | -3.499 | -16.42% |

Across 90,713 predictions, the signal has a small positive information coefficient (0.00728, t = 2.19) and a hit rate just over half (53.47%), so it isn't pure noise. But it returns almost nothing after costs, loses badly to simply holding the universe, and its deflated Sharpe is -0.707 against a 0.805 threshold. Once you account for how many configurations were tried, there is no edge left to claim.

That is the point of the project. A tiny real signal that doesn't beat buy-and-hold is the honest outcome, and reaching it without talking myself into a trade is the thing I was actually practicing.

## Why the first version was wrong

The early code in `experiments/` reported much better numbers, and all of it was leakage. The full breakdown is in `LEAKAGE_AUDIT.md`; the short version:

- It hand-picked four winners — PLTR, AAPL, NVDA, TSLA — so the universe already knew which names would do well.
- It scaled features and set label thresholds on the full dataset before splitting, letting the test period leak into training.
- It made decisions and filled trades on the same bar, using the day's own candle.
- It reported accuracy only on the high-confidence predictions, and kept the best of many training runs.

Each of these makes a backtest look better than the strategy really is. The honest harness exists to remove them one at a time and see what is left.

## Repository guide

```
honest_backtest/      the fixed evaluation harness and its result files
LEAKAGE_AUDIT.md      what was wrong with the earlier code, cited line by line
SECURITY_FINDINGS.md  secret-scan and repo-hygiene notes
experiments/          earlier supervised / RL / PLTR-specific attempts (anti-examples)
research/pipeline.py  an intermediate pipeline with stronger controls than the first scripts
backend/ docs/ ios/   supporting app surfaces kept for context
```

The `experiments/` scripts (`train_v3` through `train_v9`, the DRL runs, the PLTR-specific runs, `self_improve.py`) are earlier research, not validated results. `LEAKAGE_AUDIT.md` documents exactly what is wrong with each.

## Limitations

I would rather state these than have them found:

- The universe is today's S&P 500, so there is still survivorship bias.
- yfinance OHLCV is reproducible but not true point-in-time market data.
- Daily bars only — no intraday fills, borrow, financing, or capacity constraints.
- The deflated Sharpe uses an approximate, conservative trial count.

This is a research project, not a trading system, and nothing here is financial advice.
