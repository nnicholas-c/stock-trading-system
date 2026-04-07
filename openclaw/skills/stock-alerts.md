# Stock Trading Alert Skill for OpenClaw

## Skill: stock-alerts

This skill connects your local OpenClaw agent to the ML/RL trading system.
It runs on a schedule, fetches the latest buy/sell signals, and sends alerts
to your configured chat app (WhatsApp, Telegram, Discord, Signal, iMessage).

---

## Setup

1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/YOUR_USERNAME/stock-trading-system
   cd stock-trading-system
   pip install -r requirements.txt
   ```

2. Copy this skill file into your OpenClaw skills directory:
   ```bash
   cp openclaw/skills/stock-alerts.md ~/.openclaw/skills/
   ```

3. Configure your notification channel in `config/settings.json`:
   ```json
   {
     "notification_channel": "telegram",  // telegram | whatsapp | discord | slack | signal
     "notify_phone": "+1XXXXXXXXXX",
     "alert_tickers": ["PLTR", "AAPL", "NVDA", "TSLA"],
     "confidence_threshold": 0.55,
     "run_every": "6h"
   }
   ```

---

## What This Skill Does

Every 6 hours (configurable), OpenClaw will:

1. **Fetch latest price data** from Perplexity Finance for PLTR, AAPL, NVDA, TSLA
2. **Run the ML signal engine** (Random Forest + Gradient Boosting ensemble)
3. **Run the RL portfolio agent** (PPO agent to size positions)
4. **Scan live news** for material events affecting any ticker
5. **Send you a formatted alert** via your chosen chat app

---

## Alert Format

When a BUY or SELL signal fires above your confidence threshold, you receive:

```
🚨 TRADING SIGNAL ALERT
━━━━━━━━━━━━━━━━━━━━━━
📈 NVDA — STRONG BUY
💰 Price: $178.10
🎯 Target: $281.04 (+57.8% upside)
🧠 ML Confidence: 64.3%
📊 RSI(14): 48.2 | MACD: Bullish crossover
⚡ PPO Agent: BUY 50% position
⚠️ Risk Score: 4.2/10 (Moderate)

Top Signal Drivers:
  • MACD signal line bullish
  • Realized vol contracting (26w)
  • OBV above SMA — accumulation
  • 100% analyst consensus: Strong Buy
  • Revenue +65.5% YoY | FCF $102.7B

📰 Latest News:
  • "NVIDIA announces GB300 Ultra ramp ahead of schedule"
  • "Blackwell NVL72 demand exceeds supply through Q3"

⏱ Generated: 2026-04-07 21:34 UTC
━━━━━━━━━━━━━━━━━━━━━━
⚠️ Not financial advice. Do your own research.
```

---

## Cron Schedule

Add this to your OpenClaw heartbeat config:

```json
{
  "heartbeats": [
    {
      "name": "stock-signal-check",
      "schedule": "0 */6 * * *",
      "skill": "stock-alerts",
      "task": "Run the trading signal engine and send alerts for any BUY or SELL signals above 55% confidence threshold. Include latest news context for each triggered ticker. Format the message professionally and send via configured channel."
    },
    {
      "name": "market-open-brief",
      "schedule": "30 13 * * 1-5",
      "skill": "stock-alerts",
      "task": "Send a pre-market briefing for PLTR, AAPL, NVDA, TSLA. Include overnight news, futures context, current ML signal for each, and the RL agent's recommended position sizing. Keep it concise — bullet points only."
    },
    {
      "name": "market-close-summary",
      "schedule": "0 21 * * 1-5",
      "skill": "stock-alerts",
      "task": "Send end-of-day summary: today's price moves for PLTR/AAPL/NVDA/TSLA, any signal changes from morning, notable news, and tomorrow's watchlist items."
    }
  ]
}
```

---

## Manual Commands

You can also ask OpenClaw directly:

- `"What's the current signal for NVDA?"`
- `"Run a full scan on my 4 stocks"`
- `"Show me the backtest results"`
- `"What did the RL agent recommend for PLTR?"`
- `"Any news on TSLA today that changes the thesis?"`
- `"Compare hedge fund positioning across my 4 stocks"`

OpenClaw will run `scripts/run_signal.py` and format the results for you.

---

## Files Used

| File | Purpose |
|------|---------|
| `ml_trading_system.py` | Core ML/RL engine |
| `scripts/run_signal.py` | Quick signal runner (called by OpenClaw) |
| `scripts/news_scanner.py` | Live news + sentiment scanner |
| `trading_system/signals/current_signals.json` | Latest signals output |
| `trading_system/models/` | Saved RF, GB, PPO model files |
| `trading_system/charts/` | Generated analysis charts |

---

## Integration Architecture

```
OpenClaw (your machine)
    │
    ├── Heartbeat scheduler (cron)
    │       └── Every 6h: run stock-alerts skill
    │
    ├── Skill: stock-alerts
    │       ├── calls scripts/run_signal.py
    │       ├── reads trading_system/signals/current_signals.json
    │       ├── calls scripts/news_scanner.py
    │       └── formats & sends alert
    │
    ├── Notification channel
    │       ├── Telegram / WhatsApp / Discord / Signal
    │       └── Alert sent to your phone
    │
    └── Manual queries
            └── "What's the PLTR signal?" → instant response
```
