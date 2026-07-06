# AXIOM

This project is really about not fooling myself. I wanted to know a simple thing: can daily
price and volume data predict next-day stock moves well enough to beat just holding the market,
once you pay realistic costs and account for how many things you tried? I built the naive
version first, caught it cheating, and rebuilt it carefully. The answer is no — the signal
that survives a clean test is real but tiny, loses to buy-and-hold, and doesn't clear a
multiple-testing bar.

I kept the whole thing, including the early models that looked great and were wrong, because the
interesting part isn't the result. It's why the first version lied.

## Start here

- `honest_backtest/run.py` — the current backtest with one fixed setup and no knobs left to turn.
  It's the only performance number I'd actually defend.
- `LEAKAGE_AUDIT.md` — where I go through the old code line by line and show exactly how it leaked
  and how each leak flattered the results.
- `experiments/` — the earlier attempts. I keep them as examples of what not to do, not as results.
  Every problem in them is written up in the audit.

## What the backtest does

I stripped it down to one fixed spec so there was nothing left to tune:

- **Universe:** 60 of the more liquid current S&P 500 names, pulled from yfinance.
- **Features:** only things I'd actually know by the decision time — returns, volatility, distance
  from moving averages, RSI, a volume z-score, and the intraday range.
- **Timing:** decide at today's close, buy at tomorrow's open, sell at tomorrow's close. Nothing
  trades on a bar it hasn't seen yet.
- **Model:** one logistic regression with standardization, fit inside each training fold only.
- **Validation:** walk-forward folds, 504-day minimum train window, 63-day test windows, with a
  5-day purge and 5-day embargo in between.
- **Costs:** 5 bps to trade plus 5 bps slippage, each side.
- **Baselines:** equal-weight buy-and-hold, and a random strategy that trades at the same rate.
- **Multiple testing:** a deflated Sharpe using a conservative count of 41 model/config variations
  I could find in the old code.

Run it:

```bash
make setup
.venv/bin/python -m honest_backtest.run --max-tickers 60 --min-tickers 50
python -m pytest -q
```

## What it found

These come straight out of `honest_backtest/results/summary_metrics.csv`.

| Method | Total return | Ann. Sharpe | Max drawdown |
| --- | ---: | ---: | ---: |
| Fixed logistic signal, net | +1.84% | 0.097 | -10.21% |
| Equal-weight buy-and-hold | +64.20% | 0.646 | -14.95% |
| Zero-skill random, net | -15.74% | -3.499 | -16.42% |

Over 90,713 predictions the signal has a small positive information coefficient (0.00728,
t = 2.19) and a hit rate a hair over half (53.47%), so it's not just noise. But after costs it
barely makes anything, it gets crushed by just holding the same names, and its deflated Sharpe is
-0.707 against a 0.805 threshold. Once you account for how many configs got tried, there's no edge
left to claim.

That's the whole point. A tiny real signal that can't beat buy-and-hold is the answer, and
getting there without talking myself into a trade was the actual exercise.

## Why the first version was wrong

The old code in `experiments/` reported much better numbers, and it was all leakage. The full
write-up is in `LEAKAGE_AUDIT.md`; the short version:

- It hand-picked four winners — PLTR, AAPL, NVDA, TSLA — so the universe already knew who'd win.
- It scaled features and set label thresholds on the whole dataset before splitting, so the test
  period leaked into training.
- It decided and filled trades on the same bar, using that bar's own candle.
- It scored accuracy only on the high-confidence calls, and kept the best of many training runs.

Each of those makes a backtest look better than the strategy is. The fixed harness exists to strip
them out one at a time and see what's left.

## Repository guide

```
honest_backtest/      the fixed backtest and its result files
LEAKAGE_AUDIT.md      what was wrong with the old code, cited line by line
SECURITY_FINDINGS.md  secret-scan and repo-hygiene notes
experiments/          earlier supervised / RL / PLTR-specific attempts (what not to do)
research/pipeline.py  an in-between pipeline with tighter controls than the first scripts
backend/ docs/ ios/   supporting app surfaces, kept for context
```

The `experiments/` scripts (`train_v3` through `train_v9`, the DRL runs, the PLTR-specific runs,
`self_improve.py`) are earlier research, not validated results. `LEAKAGE_AUDIT.md` says exactly
what's wrong with each one.

## What this doesn't do

I'd rather list these than have someone find them:

- The universe is today's S&P 500, so there's still survivorship bias.
- yfinance OHLCV is reproducible but not true point-in-time data.
- Daily bars only — no intraday fills, borrow, financing, or capacity limits.
- The deflated Sharpe uses an approximate, conservative trial count.

It's a research project, not a trading system, and none of it is financial advice.
