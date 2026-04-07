# AXIOM Stock Alert Skill

## Overview
This skill connects OpenClaw to the AXIOM trading backend API.
It fetches real ML signals, LSTM forecasts, news, and intraday predictions
and sends you formatted alerts on your preferred messaging platform.

## Setup
1. Start the backend: `cd backend && uvicorn app.main:app --reload`
   Or deploy to Railway (see backend/railway.toml)
2. Set your API URL in config below
3. Choose your notification channel

## Configuration
```json
{
  "axiom_api": "https://YOUR-RAILWAY-APP.railway.app",
  "tickers": ["PLTR", "AAPL", "NVDA", "TSLA"],
  "confidence_threshold": 0.45,
  "channel": "telegram",
  "quiet_hours": {"start": "22:00", "end": "06:30"}
}
```

## Heartbeat Tasks

### Pre-Market Brief (6:00 AM on weekdays)
```
Fetch pre-market data from the AXIOM API and send a morning brief.

Steps:
1. Call GET https://YOUR-API/signals/ to get all 4 ticker signals
2. Call GET https://YOUR-API/predict/PLTR/intraday (repeat for AAPL, NVDA, TSLA)
3. Call GET https://YOUR-API/news/ to get overnight news for all tickers
4. Format the brief as follows:

🌅 AXIOM PRE-MARKET BRIEF — {date}
━━━━━━━━━━━━━━━━━━━━━━━━
{for each ticker}
{emoji} {TICKER} | {SIGNAL} ({conf}%) | ${price}
  → Today: {intraday direction} | News: {sentiment}
  → LSTM 4w: ${week4_price} ({pct}%)
  → Top story: {headline}

{end for}
━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ Not financial advice. DYOR.

Send via the configured channel.
```

### Signal Monitor (every 4 hours)
```
Check if any signals have changed or are high-confidence.

Steps:
1. Call GET https://YOUR-API/signals/ 
2. For each ticker where confidence > 0.55 AND signal is BUY, STRONG BUY, or SELL:
   - Call GET https://YOUR-API/predict/{ticker}/scenarios
   - Format an alert:

🚨 SIGNAL ALERT — {TICKER}
━━━━━━━━━━━━━━━━━━━━
{signal_emoji} {SIGNAL} | {conf}% confidence
💰 ${price} → Target ${target} ({upside}%)
📊 LSTM 4w: ${lstm_w4} ({pct}%)
🎯 Scenarios:
   Bull: ${bull_price} ({bull_pct}%) — {bull_catalyst}
   Base: ${base_price} ({base_pct}%)
   Bear: ${bear_price} ({bear_pct}%)
⚠️ Not financial advice.

Only send if the signal is different from the last check.
```

### EOD Summary (4:00 PM on weekdays)
```
End of day summary with weekly predictions.

Steps:
1. GET https://YOUR-API/signals/ for all signals
2. GET https://YOUR-API/predict/{ticker}/weekly for each ticker
3. Search the web for today's closing prices for PLTR, AAPL, NVDA, TSLA
4. Format:

📊 AXIOM EOD SUMMARY — {date}
━━━━━━━━━━━━━━━━━━━━━━━━
{for each ticker}
{TICKER}: ${close} ({daily_pct}%) | Signal: {signal}
  Week ahead: W+1 ${w1} · W+2 ${w2} · W+3 ${w3} · W+4 ${w4}
  LGB 4w est: {lgb_pct}%
{end for}
━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ Not financial advice.
```

## Manual Commands
You can ask OpenClaw directly:
- "What's the NVDA signal?" → GET /signals/NVDA/summary
- "What will PLTR do today?" → GET /predict/PLTR/intraday
- "Show me TSLA scenarios" → GET /predict/TSLA/scenarios
- "Latest AAPL news" → GET /news/AAPL
- "NVDA backtest results" → GET /backtest/NVDA

## API Reference
| Endpoint | Description |
|----------|-------------|
| GET /signals/ | All 4 signals |
| GET /signals/{ticker} | Full signal object |
| GET /signals/{ticker}/summary | One-line alert text |
| GET /predict/{ticker}/intraday | Today's direction |
| GET /predict/{ticker}/weekly | 4-week LSTM forecast |
| GET /predict/{ticker}/scenarios | Bull/Base/Bear |
| GET /news/{ticker} | Live news + sentiment |
| GET /news/{ticker}/premarket | Top stories brief |
| GET /backtest/{ticker} | Backtest stats |
| GET /health | API health |
