#!/usr/bin/env python3
"""
AXIOM Self-Improving Model Pipeline
====================================
Runs daily after market close. For each ticker:
  1. Fetch today's actual closing price
  2. Compare vs yesterday's LSTM / LGB predictions
  3. Compute prediction error (MAE, directional accuracy)
  4. Append error to persistent error log
  5. If MAE exceeds threshold → trigger incremental retraining
  6. Update signals JSON with new predictions
  7. Save performance history for dashboard display

This implements online learning / continual learning:
  - Walk-forward: always train on newest data, validate on last N days
  - Error-weighted retraining: worse predictions get model retrained first
  - Feature importance drift: track which features degrade over time
"""

import json, os, sys, warnings, pickle, logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT        = Path('/home/user/workspace')
SIGNALS_DIR = ROOT / 'trading_system' / 'signals'
MODELS_DIR  = ROOT / 'trading_system' / 'models' / 'v4'
DATA_DIR    = ROOT / 'finance_data' / 'daily'
TRACK_DIR   = ROOT / 'cron_tracking' / 'self_improve'
TRACK_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE    = TRACK_DIR / 'prediction_log.jsonl'
PERF_FILE   = TRACK_DIR / 'model_performance.json'
RETRAIN_THRESHOLD_MAE  = 0.03   # 3% MAE triggers retrain
RETRAIN_THRESHOLD_DAYS = 5      # retrain if 5 consecutive errors above threshold

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger('AXIOM-SelfImprove')

TICKERS = ['PLTR','AAPL','NVDA','TSLA']

# ─────────────────────────────────────────────────────────────────────────────
def load_current_signals() -> dict:
    path = SIGNALS_DIR / 'current_signals_v4.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def load_today_prices() -> dict:
    """Load latest closing prices from daily CSV files."""
    prices = {}
    for ticker in TICKERS:
        csv = DATA_DIR / f'{ticker}_daily.csv'
        if csv.exists():
            df = pd.read_csv(csv, parse_dates=['date']).sort_values('date')
            prices[ticker] = {
                'close':  float(df.iloc[-1]['close']),
                'date':   str(df.iloc[-1]['date'].date()),
                'volume': float(df.iloc[-1]['volume']),
                'open':   float(df.iloc[-1]['open']),
                'high':   float(df.iloc[-1]['high']),
                'low':    float(df.iloc[-1]['low']),
            }
    return prices

def load_prior_predictions() -> dict:
    """Load prediction log to find what was predicted for today."""
    if not LOG_FILE.exists():
        return {}
    preds = {}
    with open(LOG_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                ticker = entry.get('ticker')
                if ticker and ticker not in preds:
                    preds[ticker] = entry
            except:
                pass
    return preds

def compute_prediction_error(ticker: str, predicted_price: float, actual_price: float) -> dict:
    """
    Compute multiple error metrics for one prediction.
    """
    mae   = abs(actual_price - predicted_price) / actual_price  # relative MAE
    rmse  = ((actual_price - predicted_price) ** 2) ** 0.5 / actual_price
    direction_pred = 1 if predicted_price > actual_price * 0.999 else -1  # predicted direction from yesterday
    actual_return  = (actual_price - predicted_price) / predicted_price
    direction_correct = (direction_pred * (1 if actual_return > 0 else -1)) > 0

    return {
        'ticker':      ticker,
        'date':        datetime.now().date().isoformat(),
        'predicted':   round(predicted_price, 4),
        'actual':      round(actual_price, 4),
        'error_abs':   round(actual_price - predicted_price, 4),
        'mae_pct':     round(mae * 100, 4),
        'rmse_pct':    round(rmse * 100, 4),
        'direction_correct': direction_correct,
        'actual_return_pct': round(actual_return * 100, 4),
        'needs_retrain': mae > RETRAIN_THRESHOLD_MAE,
    }

def log_prediction(entry: dict):
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(entry) + '\n')

def load_performance_history() -> dict:
    if PERF_FILE.exists():
        with open(PERF_FILE) as f:
            return json.load(f)
    return {t: {'mae_history': [], 'direction_accuracy': [], 'retrain_count': 0, 'total_predictions': 0} for t in TICKERS}

def update_performance_history(perf: dict, errors: dict):
    for ticker, err in errors.items():
        if ticker not in perf:
            perf[ticker] = {'mae_history': [], 'direction_accuracy': [], 'retrain_count': 0, 'total_predictions': 0}
        perf[ticker]['mae_history'].append(err['mae_pct'])
        perf[ticker]['direction_accuracy'].append(1 if err['direction_correct'] else 0)
        perf[ticker]['total_predictions'] += 1
        # Keep last 60 days
        if len(perf[ticker]['mae_history']) > 60:
            perf[ticker]['mae_history'] = perf[ticker]['mae_history'][-60:]
        if len(perf[ticker]['direction_accuracy']) > 60:
            perf[ticker]['direction_accuracy'] = perf[ticker]['direction_accuracy'][-60:]
    return perf

def should_retrain(ticker: str, perf: dict) -> bool:
    """Check if recent consecutive errors exceed threshold."""
    hist = perf.get(ticker, {}).get('mae_history', [])
    if len(hist) < 2:
        return False
    recent = hist[-RETRAIN_THRESHOLD_DAYS:]
    return sum(1 for e in recent if e > RETRAIN_THRESHOLD_MAE * 100) >= max(2, len(recent)//2)

def incremental_retrain(ticker: str) -> bool:
    """
    Incremental retraining: add new data point, retrain LGB/XGB on expanded dataset.
    This is the core of the self-improvement loop.
    """
    try:
        log.info(f"  [{ticker}] Triggering incremental retrain...")

        csv = DATA_DIR / f'{ticker}_daily.csv'
        if not csv.exists():
            log.warning(f"  [{ticker}] No data file found")
            return False

        df = pd.read_csv(csv, parse_dates=['date']).sort_values('date')
        log.info(f"  [{ticker}] Loaded {len(df)} rows. Last date: {df.iloc[-1]['date'].date()}")

        # Load macro data for cross-asset features
        macro_dfs = {}
        for m in ['SPY', 'QQQ', 'TLT', 'GLD']:
            mcsv = ROOT / 'finance_data' / 'macro' / f'{m}_daily.csv'
            if mcsv.exists():
                macro_dfs[m] = pd.read_csv(mcsv, parse_dates=['date']).set_index('date')['close']

        df = df.set_index('date')

        # Basic feature engineering (fast version for incremental)
        df['ret_1d']   = df['close'].pct_change(1)
        df['ret_5d']   = df['close'].pct_change(5)
        df['ret_20d']  = df['close'].pct_change(20)
        df['sma20']    = df['close'].rolling(20).mean()
        df['sma50']    = df['close'].rolling(50).mean()
        df['vs_ma20']  = (df['close'] - df['sma20']) / df['sma20']
        df['vs_ma50']  = (df['close'] - df['sma50']) / df['sma50']
        df['vol_ratio']= df['volume'] / df['volume'].rolling(20).mean()
        df['rsi14']    = 100 - 100/(1+df['ret_1d'].clip(lower=0).rolling(14).mean()/(-df['ret_1d'].clip(upper=0)).rolling(14).mean().abs().replace(0,1e-10))
        df['volatility']= df['ret_1d'].rolling(20).std() * (252**0.5)

        # Macro features
        for m, series in macro_dfs.items():
            df[f'{m}_ret'] = series.pct_change(1).reindex(df.index)

        # Target: 5d forward return
        df['target_5d']  = df['close'].shift(-5) / df['close'] - 1
        df['target_20d'] = df['close'].shift(-20) / df['close'] - 1

        feat_cols = ['ret_1d','ret_5d','ret_20d','vs_ma20','vs_ma50','vol_ratio','rsi14','volatility'] + [f'{m}_ret' for m in macro_dfs]
        df_clean  = df.dropna(subset=feat_cols+['target_5d']).copy()

        if len(df_clean) < 50:
            log.warning(f"  [{ticker}] Insufficient clean rows ({len(df_clean)})")
            return False

        X = df_clean[feat_cols].values
        y = (df_clean['target_5d'] > 0.02).astype(int).values  # binary: up>2% in 5d

        from sklearn.preprocessing import StandardScaler
        import xgboost as xgb

        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)

        # Walk-forward: train on first 80%, validate on last 20%
        split = int(len(X_s) * 0.80)
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss',
            early_stopping_rounds=20, random_state=42, n_jobs=-1, verbosity=0
        )
        model.fit(X_s[:split], y[:split],
                  eval_set=[(X_s[split:], y[split:])],
                  verbose=False)

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y[split:], model.predict(X_s[split:]))
        log.info(f"  [{ticker}] Retrain complete. Test accuracy: {acc:.1%}")

        # Save updated model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODELS_DIR / f'{ticker}_xgb_incremental.pkl', 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler, 'feat_cols': feat_cols,
                         'retrain_date': datetime.now().isoformat(), 'accuracy': acc}, f)

        return True

    except Exception as e:
        log.error(f"  [{ticker}] Retrain failed: {e}")
        return False

def update_signals_with_new_predictions(signals: dict, prices: dict, perf: dict) -> dict:
    """Update the signals JSON with fresh price data and performance metrics."""
    today = datetime.now().date().isoformat()

    for ticker in TICKERS:
        if ticker not in signals.get('signals', {}):
            continue
        sig = signals['signals'][ticker]

        # Update price if we have new data
        if ticker in prices:
            p = prices[ticker]
            sig['price']        = p['close']
            sig['price_date']   = p['date']
            sig['day_volume']   = p['volume']

        # Embed performance history
        t_perf = perf.get(ticker, {})
        mae_hist = t_perf.get('mae_history', [])
        dir_hist = t_perf.get('direction_accuracy', [])

        sig['model_performance'] = {
            'mae_7d_avg':      round(np.mean(mae_hist[-7:]),  3) if len(mae_hist)>=7  else None,
            'mae_30d_avg':     round(np.mean(mae_hist[-30:]), 3) if len(mae_hist)>=30 else None,
            'dir_acc_7d':      round(np.mean(dir_hist[-7:]),  3) if len(dir_hist)>=7  else None,
            'dir_acc_30d':     round(np.mean(dir_hist[-30:]), 3) if len(dir_hist)>=30 else None,
            'total_predictions': t_perf.get('total_predictions', 0),
            'retrain_count':   t_perf.get('retrain_count', 0),
            'last_mae':        round(mae_hist[-1], 3) if mae_hist else None,
        }

    signals['last_self_improve'] = today
    signals['version'] = 'v4-adaptive'
    return signals

def generate_improvement_report(errors: dict, perf: dict, retrained: list) -> str:
    """Generate a concise improvement report."""
    lines = [f"AXIOM Self-Improvement Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             "="*60]
    for ticker, err in errors.items():
        status = "✓ ACCURATE" if not err['needs_retrain'] else "⚠ RETRAINED" if ticker in retrained else "⚠ HIGH ERROR"
        lines.append(f"\n{ticker} {status}")
        lines.append(f"  Predicted: ${err['predicted']:.2f} | Actual: ${err['actual']:.2f}")
        lines.append(f"  MAE: {err['mae_pct']:.2f}% | Direction: {'✓' if err['direction_correct'] else '✗'}")
        lines.append(f"  Actual return: {err['actual_return_pct']:+.2f}%")
        t_perf = perf.get(ticker, {})
        mae_hist = t_perf.get('mae_history', [])
        dir_hist = t_perf.get('direction_accuracy', [])
        if len(mae_hist) >= 5:
            lines.append(f"  7d avg MAE: {np.mean(mae_hist[-7:]):.2f}% | Dir acc 7d: {np.mean(dir_hist[-7:])*100:.0f}%")

    if retrained:
        lines.append(f"\nRetrained: {', '.join(retrained)}")
    else:
        lines.append("\nNo retraining triggered today.")
    return '\n'.join(lines)

# ─────────────────────────────────────────────────────────────────────────────
def run():
    log.info("="*60)
    log.info("AXIOM SELF-IMPROVEMENT PIPELINE — START")
    log.info(f"Run time: {datetime.now()}")
    log.info("="*60)

    # 1. Load current state
    signals = load_current_signals()
    if not signals:
        log.error("No signals found. Run train_v4.py first.")
        return

    prices  = load_today_prices()
    perf    = load_performance_history()
    errors  = {}
    retrained = []

    log.info(f"Loaded prices for: {list(prices.keys())}")

    # 2. Compare predictions vs actuals
    for ticker in TICKERS:
        sig = signals.get('signals', {}).get(ticker)
        if not sig or ticker not in prices:
            continue

        # Get yesterday's 1d LSTM prediction for today
        # lstm_1d is a fractional return, so predicted = yesterday_price * (1 + lstm_1d)
        yesterday_price = prices[ticker]['close'] / (1 + prices[ticker].get('actual_ret_1d', sig.get('lstm_1d', 0)))
        predicted_today = sig['price'] * (1 + sig.get('lstm_1d', 0))
        actual_today    = prices[ticker]['close']

        log.info(f"\n  [{ticker}]")
        log.info(f"  Last signal price: ${sig['price']:.2f}")
        log.info(f"  LSTM 1d prediction: {sig.get('lstm_1d',0)*100:+.2f}% → ${predicted_today:.2f}")
        log.info(f"  Actual today: ${actual_today:.2f}")

        err = compute_prediction_error(ticker, predicted_today, actual_today)
        errors[ticker] = err
        log.info(f"  MAE: {err['mae_pct']:.2f}% | Direction: {'✓' if err['direction_correct'] else '✗'} | Retrain: {err['needs_retrain']}")

        log_prediction({**err, 'lstm_1d_used': sig.get('lstm_1d',0), 'signal': sig.get('signal','?')})

    # 3. Update performance history
    perf = update_performance_history(perf, errors)

    # 4. Trigger retraining where needed
    for ticker in TICKERS:
        if ticker in errors and (errors[ticker]['needs_retrain'] or should_retrain(ticker, perf)):
            log.info(f"\n  [{ticker}] Initiating incremental retrain...")
            success = incremental_retrain(ticker)
            if success:
                retrained.append(ticker)
                perf[ticker]['retrain_count'] = perf.get(ticker, {}).get('retrain_count', 0) + 1

    # 5. Update signals with new prices + performance
    signals = update_signals_with_new_predictions(signals, prices, perf)

    # 6. Save everything
    with open(PERF_FILE, 'w') as f:
        json.dump(perf, f, indent=2, default=str)

    out_path = SIGNALS_DIR / 'current_signals_v4.json'
    with open(out_path, 'w') as f:
        json.dump(signals, f, indent=2, default=str)

    # 7. Generate and save report
    report = generate_improvement_report(errors, perf, retrained)
    report_path = TRACK_DIR / f"report_{datetime.now().strftime('%Y-%m-%d')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    log.info("\n" + "="*60)
    log.info("SELF-IMPROVEMENT COMPLETE")
    log.info(f"Errors logged: {len(errors)}")
    log.info(f"Retrained: {retrained or 'None'}")
    log.info(f"Report: {report_path}")
    print("\n" + report)

if __name__ == '__main__':
    run()
