# Honest Backtest Harness

This is the clean re-run behind the AXIOM audit — the one backtest here I'd actually defend.

Run:

```bash
python -m honest_backtest.run
```

Method:

- Uses a broad S&P 500 sample fetched from Wikipedia plus yfinance. A static S&P 500 fallback list is used only if the Wikipedia table is unavailable.
- Builds strictly causal daily features from each ticker's own OHLCV history.
- Decides after `close[t]`, enters at `open[t+1]`, exits at `close[t+1]`.
- Fits all preprocessing and the logistic model inside each walk-forward fold.
- Uses chronological train/validation folds with a purge plus embargo.
- Applies explicit round-trip transaction cost and slippage.
- Compares the signal to equal-weight buy-and-hold and a zero-skill random baseline.
- Reports IC, IC t-stat, hit rate, turnover, out-of-sample Sharpe, and a deflated Sharpe adjustment using a lower-bound count of prior configurations in the repo.

Assumptions and limitations:

- yfinance data is not point-in-time fundamental data. It is suitable for a transparent price-only audit, not for validating the original fundamental/news claims.
- Current S&P 500 membership is used, so the universe still has survivorship bias relative to historical S&P 500 membership.
- The model is intentionally simple and fixed before evaluation. It is not optimized for performance.
- The deflated Sharpe calculation is an approximation, not a full Lopez de Prado implementation with all higher-moment corrections and dependence adjustments.
- The strategy uses daily round trips. That is conservative for trading costs and makes execution lag explicit.
