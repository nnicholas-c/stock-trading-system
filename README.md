# AXIOM

I started this project to answer one question honestly: can daily price and volume
data predict next-day equity returns well enough to beat buying and holding, once you
pay realistic costs and stop fooling yourself with leakage and repeated trials?

My answer, so far, is no. A fixed logistic model finds a statistically detectable but
tiny signal. It loses badly to equal-weight buy-and-hold and does not survive a
multiple-testing-aware Sharpe adjustment. I'm keeping the repository as a record of how
I got to that answer, including the earlier attempts that looked good only because they
were leaking.

The two things worth reading first are:

- `honest_backtest/run.py` — the current, leakage-aware evaluation. Every number below
  comes from it.
- `LEAKAGE_AUDIT.md` — a line-by-line account of what was wrong with the earlier work
  and why its metrics can't be trusted.

## The honest result

I run one fixed model specification and report net-of-cost, out-of-sample metrics. No
configuration search, no cherry-picking. To reproduce:

```bash
.venv/bin/python -m honest_backtest.run --max-tickers 60 --min-tickers 50
```

That writes `honest_backtest/results/summary_metrics.csv`, which is the source of these
numbers:

| Method | Total return | Ann. Sharpe | Max drawdown |
| --- | ---: | ---: | ---: |
| Fixed logistic signal, net | +1.84% | 0.097 | -10.21% |
| Equal-weight buy-hold | +64.20% | 0.646 | -14.95% |
| Zero-skill random, net | -15.74% | -3.499 | -16.42% |

All three run over the same universe and dates, on 90,713 predictions across a median of
60 tickers. The signal makes money in absolute terms, but far less than just holding the
same names, and a random strategy with the same trade frequency and cost model loses money,
which is the sanity check I wanted.

The signal diagnostics say the same thing more precisely:

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

There is a real, positive information coefficient — the IC t-stat clears 2 — but it's
economically trivial. Once I deflate the Sharpe for the 41 model/configuration trials I
can count in this repository, the strategy sits well below the threshold. That's the point:
with enough trials something will look good, so the honest number is the deflated one, and
it's negative.

## How the harness is built

`honest_backtest/run.py` is deliberately plain. The choices that keep it honest:

- Universe: 60 median tradable names from a current S&P 500 download (yfinance), with a
  static fallback list if the Wikipedia table is unavailable.
- Features: causal daily OHLCV features only — returns over several horizons, realized
  volatility, moving-average distances, RSI, a volume z-score, and intraday range.
- Timing: decide at `close[t]`, enter at `open[t+1]`, exit at `close[t+1]`. Nothing is
  decided using a bar it then trades into.
- Model: `StandardScaler` plus `LogisticRegression`, fit inside each training fold and
  nowhere else.
- Validation: walk-forward folds, 504-day minimum train window, 63-day validation windows,
  with a 5-day purge and 5-day embargo between them.
- Costs: 5 bps transaction cost plus 5 bps slippage per side.
- Baselines: equal-weight buy-hold, and a zero-skill random strategy matched to the
  signal's active rate and cost model.
- Multiple testing: a deflated Sharpe adjustment using a lower-bound count of 41 prior
  model/configuration trials found in the repository.

`honest_backtest/README.md` has the same method statement alongside its assumptions and
limitations.

## The experiments/ directory is not results

`experiments/` holds the earlier supervised, deep-learning, reinforcement-learning, and
PLTR-specific attempts. I'm keeping them on purpose, but as documented anti-examples, not
as evidence. Each of those files carries a header pointing back to the audit, and the
specific flaws are catalogued in `LEAKAGE_AUDIT.md`: hand-picked tickers, scalers fit on
the full series before the time split, same-bar execution, accuracy reported only on
high-confidence signals, and repeated configuration search. `ml_trading_system.py` at the
repository root is the oldest of these and has the most problems.

`research/pipeline.py` is the one older component that mostly holds up — it uses sklearn
pipelines, walk-forward validation, and fold-local fitting — but it still searches across
candidate model families on a small universe, so I treat it as research rather than a
performance claim. The audit's "What Survives" section says exactly how far I trust it.

## Layout

```text
honest_backtest/
  run.py                 Fixed-specification validation harness (the headline)
  results/               Reproducible result CSVs and figures

LEAKAGE_AUDIT.md         Line-by-line leakage and model-risk audit
SECURITY_FINDINGS.md     Secret-scan and repository hygiene notes

experiments/             Earlier attempts, kept as documented anti-examples
  train_v3.py ... train_v9_xgb.py, train_drl_*.py, train_pltr_*.py, self_improve.py
  EXPERIMENTS.md         Historical experiment inventory

research/pipeline.py     Earlier pipeline with stronger controls than the experiments
ml_trading_system.py     Oldest attempt; see the audit before reading it

backend/, docs/, ios/    Supporting app surfaces kept for context
```

## Running it locally

```bash
make setup
.venv/bin/python -m honest_backtest.run --max-tickers 60 --min-tickers 50
python -m pytest -q
```

The backtest writes these to `honest_backtest/results/`:

- `summary_metrics.csv`, `fold_predictions.csv`, `daily_returns.csv`, `universe.csv`
- `equity_curves.png`, `score_vs_return.png`, `run_config.json`

Because the harness pulls a current S&P 500 sample and live yfinance history, a fresh run
extends the window to the current date and will not reproduce the committed numbers to the
last basis point. The committed CSVs in `honest_backtest/results/` are the snapshot the
tables above quote.

## What this doesn't do

- It uses current S&P 500 membership, so survivorship bias is reduced relative to a few
  hand-picked names but not removed.
- yfinance OHLCV is transparent and reproducible, but it is not point-in-time institutional
  data.
- It uses daily bars only. There is no modeling of intraday queue position, borrow,
  financing, tax, or capacity.
- The deflated Sharpe is an approximation using a conservative lower-bound trial count.
- This is a research project, not a trading system.

## Disclaimer

For education and research only. Not financial advice, not an investment recommendation,
and not a live trading system.
