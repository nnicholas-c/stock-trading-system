# LEAKAGE AUDIT

Audit date: 2026-06-05

Scope: legacy training scripts, backend backtest reporting, and the new `honest_backtest/` harness. The result is blunt: the old headline metrics should not be presented as validated trading performance. The honest harness is the only current reproducible performance headline.

## Executive Summary

- Old headline results were materially overstated.
- Main causes: hand-picked ticker universe, full-data scaling before time splits, same-bar execution, repeated trial selection, confidence-gated accuracy reporting, and hard-coded/static backtest outputs.
- The more recent `research/pipeline.py` is more careful than the early scripts: it uses sklearn pipelines, walk-forward validation, purge/embargo logic, and fold-local fitting. It still searches candidate families on a tiny universe and should not be used as a broad production claim.
- The new `honest_backtest/run.py` fixes the headline evaluation path with a fixed model, train-only transforms, purged/embargoed walk-forward folds, next-day execution, explicit costs/slippage, a 50+ ticker universe, and buy-hold/random baselines.

## Validated Headline Replacement

Command:

```bash
.venv/bin/python -m honest_backtest.run --max-tickers 60 --min-tickers 50
```

Current summary from `honest_backtest/results/summary_metrics.csv`:

| Method | Total return | Ann. Sharpe | Max drawdown |
| --- | ---: | ---: | ---: |
| Fixed logistic signal, net | +1.84% | 0.097 | -10.21% |
| Equal-weight buy-hold | +64.20% | 0.646 | -14.95% |
| Zero-skill random, net | -15.74% | -3.499 | -16.42% |

Signal diagnostics: IC 0.00728, IC t-stat 2.19, active hit rate 53.47%, average daily turnover 10.72%, deflated Sharpe -0.707, DSR probability 1.79e-235.

## Findings

### 1. Survivorship And Selection Bias

`ml_trading_system.py:41` hard-codes four successful/attention-heavy tickers:

```python
TICKERS = ['PLTR', 'AAPL', 'NVDA', 'TSLA']
```

The later experiments repeatedly keep this narrow universe or move to PLTR-only runs:

- `experiments/train_drl_v1.py:16-17` targets 80% high-confidence accuracy on four tickers.
- `experiments/train_pltr_ultra.py:28-31` is PLTR-only and explicitly targets 80% directional accuracy.
- `experiments/train_v7_deep.py:96-98` keeps `["NVDA", "AAPL", "PLTR", "TSLA"]`.

Impact: results cannot be read as broad-market evidence. The honest harness uses at least 50 current S&P 500 names, which still has survivorship bias but is far less cherry-picked.

### 2. Time-Invariant And Non-Point-In-Time Features

`ml_trading_system.py:49-90` embeds static fundamentals, analyst targets, and sentiment values. `ml_trading_system.py:191-199` adds them as constants to every historical row:

```python
df['analyst_upside'] = (fund['analyst_target'] / fund['price'] - 1) * 100
df['bull_pct'] = fund['bull_pct']
```

Impact: later information is made available to earlier historical rows. This is a point-in-time violation unless every value is timestamped and lagged to its publication date.

### 3. Full-Data Preprocessing Before Splits

`ml_trading_system.py:254-259` says time-series CV/no lookahead, but fits the scaler on the full dataset before the train/test split:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

More examples:

- `experiments/self_improve.py:204-208` fits `StandardScaler` on all rows before the 80/20 split.
- `experiments/train_v3.py:473-494` fits the LSTM scaler on the full feature matrix before sequence splits.
- `experiments/train_v3.py:555-568` refits a full-data scaler for the latest forecast and saves it.
- `experiments/train_v3.py:590-592` scales the full classifier dataset.
- `experiments/train_v3.py:653-655` scales the full LightGBM regression dataset.
- `experiments/train_v3.py:693-697` scales the full RL environment feature matrix.
- `experiments/train_v4.py:680-686` scales full LSTM inputs before sequence split.
- `experiments/train_v4.py:814-819` scales full XGB inputs and computes class quantiles on full forward returns.
- `experiments/train_v4.py:873-935` scales full LGB/RF inputs before validation.

Impact: validation rows influence feature normalization and target thresholds. This can materially inflate apparent out-of-sample skill.

### 4. Same-Bar Execution And Missing Execution Lag

The original ML backtest predicts from row `i`, then buys or sells at `row['close']` before valuing at the next close:

- Prediction and signal: `ml_trading_system.py:627-636`
- Immediate fill at same close: `ml_trading_system.py:637-654`
- Next-row valuation: `ml_trading_system.py:656-657`

The RL environment uses the current close in the observation and then executes at that same close:

- Observation price: `ml_trading_system.py:374-382`
- Execution price: `ml_trading_system.py:388-406`

Impact: if the model uses same-day candle features like body/wick/direction (`ml_trading_system.py:185-189`) and trades at the same close, it gets information that was not available at the fill.

The honest harness makes the decision at `close[t]`, enters at `open[t+1]`, exits at `close[t+1]`, and pays transaction cost plus slippage.

### 5. Train/Test Contamination Through Stacking And Iteration

`experiments/train_v4.py:963-989` builds meta-model features from base-model predictions over all rows, then fits the meta-model on the first 80%. The base models and preprocessing have already seen validation context:

```python
xgb_p = xgb_res["clf"].predict_proba(X_sc)
rf_p = rf_res["clf"].predict_proba(rf_res["scaler"].transform(X_raw))
meta_clf.fit(meta_X[:n_tr], y[:n_tr])
```

`experiments/train_v8_finetune.py:1037-1048` describes a loop that compares to ground truth, adjusts weights, retrains, and re-predicts. `experiments/train_v8_finetune.py:1161-1244` then reruns predictions on the same result frame after failure analysis.

`experiments/train_v9_xgb.py:848-893` iterates up to 15 times and selects the best metrics; `experiments/train_v9_xgb.py:968-970` runs a five-iteration loop in main.

Impact: repeated adjustment on the same validation target turns validation into training.

### 6. High-Confidence-Only Accuracy Reporting

`experiments/train_drl_v1.py:550-620` computes forward labels and records only emitted BUY/SELL predictions. HOLDs are skipped:

```python
if pred_dir == -1:
    continue
```

`experiments/train_drl_v1.py:693-723` tries several DRL schedules and horizons, selecting the best high-confidence accuracy. `experiments/train_pltr_ultra.py:823-836` reports confidence-tier accuracy and Sharpe from selected signals; `experiments/train_pltr_ultra.py:925-959` tries three large PPO schedules and keeps the best.

Impact: accuracy on a filtered subset is not the same as full strategy performance, especially without a costed portfolio simulation.

### 7. Multiple Testing And Configuration Search

A conservative constructor scan found 41 explicit model constructors across the legacy experiment scripts, root ML scripts, and `research/pipeline.py`. A broader pattern scan found 142 lines involving model families, schedules, thresholds, and config paths.

Examples:

- v3/v4 train LSTM, XGB, LightGBM, RF, meta models, and PPO variants.
- DRL v1 tries multiple PPO timestep/horizon configs (`experiments/train_drl_v1.py:693-723`).
- PLTR Ultra tries 150k/200k/250k schedules (`experiments/train_pltr_ultra.py:925-959`).
- v9 iterates and keeps the best run (`experiments/train_v9_xgb.py:848-893`).

Impact: unadjusted Sharpe/accuracy numbers are expected to look good after enough trials. The honest harness reports a deflated Sharpe adjustment using the 41-trial lower bound.

### 8. Synthetic Or Literature-Style Claims Mixed With Real-Data Claims

`experiments/train_v7_deep.py:6-57` cites current academic/literature claims and vendor-style data. The market data loader says it uses realistic synthetic OHLCV (`experiments/train_v7_deep.py:245-250`) and generates correlated returns (`experiments/train_v7_deep.py:279-317`).

It also computes alpha IC using next-day returns inside the transformer approximation (`experiments/train_v7_deep.py:507-525`) and later evaluates against generated forward returns (`experiments/train_v7_deep.py:1463-1473`).

Impact: useful as a simulation/idea notebook, not evidence of real-market predictive performance.

### 9. Cost And Slippage Realism

The old ML backtest includes a fixed 0.1% cost (`ml_trading_system.py:623`) but no spread, slippage, next-bar fill, or borrow/short constraints. Several old scripts report accuracy or selected-signal Sharpe rather than net portfolio returns.

The honest harness uses round-trip cost equal to two sides of transaction cost plus slippage:

```python
round_trip_cost = 2.0 * (config.transaction_cost_bps + config.slippage_bps) / 10_000.0
```

Default: 5 bps transaction cost plus 5 bps slippage per side.

### 10. Backend Static Performance Claims

The pre-rewrite backend had a hard-coded `BACKTEST_CACHE` with unreproducible PLTR/AAPL/NVDA/TSLA returns and Sharpe values, and it passed artifact-embedded backtest payloads through as live API output. That was invalidated in this branch: `/backtest` now reports that old ticker-level backtests are not validated and points readers to `honest_backtest/results/summary_metrics.csv`.

## What Survives

- `research/pipeline.py:654-692` builds sklearn `Pipeline` estimators where scalers are inside the estimator.
- `research/pipeline.py:811-860` fits estimators only on train slices and uses purge/embargo logic.
- `research/pipeline.py:1797-1801` still compares candidate families, so it should be treated as research, not a single untouched hypothesis.
- `research/pipeline.py:2048-2078` promotes artifacts based on recent Brier/log-loss comparisons, which is useful for monitoring but not enough for a broad performance claim.

## Required Standard Going Forward

- Every performance number must name its data source, universe, dates, exact command, and output artifact.
- All transforms must be fit inside the training fold only.
- Execution must include at least one-bar lag and explicit costs/slippage.
- Any model/config search must be reported with trial count and a multiple-testing adjustment.
- Old experiment metrics can be discussed only as historical experiments, not validated production results.
