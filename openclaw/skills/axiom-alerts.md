# AXIOM Enterprise v4 — OpenClaw Skill

## What this skill does
Delivers professional trading intelligence directly to your phone via your chosen messaging app (Telegram, WhatsApp, Discord, Signal, or iMessage) — automatically, on schedule, every trading day.

No action required from you. OpenClaw runs this in the background on your Mac/PC.

---

## Setup (one-time, ~10 minutes)

### 1. Install OpenClaw
```bash
# From https://openclaw.ai — one-liner install
curl -fsSL https://openclaw.ai/install.sh | bash
```

### 2. Clone the repo and install dependencies
```bash
git clone https://github.com/nnicholas-c/stock-trading-system
cd stock-trading-system
pip install -r requirements.txt
```

### 3. Choose your notification channel
Edit `openclaw/openclaw-config.json` and set `"channel"` to one of:
`"telegram"` | `"whatsapp"` | `"discord"` | `"signal"` | `"imessage"`

Then follow OpenClaw's channel setup guide at https://openclaw.ai/docs/channels.

### 4. Copy skill into OpenClaw
```bash
cp openclaw/skills/axiom-alerts.md ~/.openclaw/skills/
cp openclaw/openclaw-config.json ~/.openclaw/config.json
```

### 5. Run the backend (optional — for live API)
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# Runs at http://localhost:8000  |  Docs: http://localhost:8000/docs
```

Or deploy to Railway (free tier) for always-on access:
```bash
npm i -g @railway/cli && railway login && railway up
# Your API will be at https://YOUR-APP.railway.app
```

---

## What you receive — Alert Formats

### 🌅 Pre-Market Brief (6:00 AM PDT, Mon–Fri)
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌅 AXIOM ENTERPRISE | Pre-Market Brief
[Date] | Market: BEAR | 6:00 AM PDT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 PLTR — TODAY: FLAT (36% conf)
News: "Why Palantir's Best Days May Still Be Ahead" [+0.86]
ML: SELL | LSTM 5d: −2.2% | Earnings in 27 days (May 4)
Risk: High P/E (242x) + insider selling pressure

✅ NVDA — TODAY: UP (54% conf)
News: "Goldman Sachs spots Nvidia shift not seen in 13 years" [+0.00]
ML: BUY | LSTM 5d: +5.1% | Earnings in 43 days (May 20)
Watch: Export control headlines — instant −7% impact historically

▼ TSLA — TODAY: DOWN (29% conf)
News: "JPMorgan warns Tesla could sink 60%" [−0.66]
ML: HOLD | LSTM 5d: +4.5% | Earnings in 15 days (Apr 22) ⚠️
Risk: −8.5% below MA20, JPMorgan 60% downside call

▼ AAPL — TODAY: DOWN (28% conf)
News: "Apple foldable phone delayed to 2027" [−0.40]
ML: BUY | LSTM 5d: +5.0% | Earnings in 23 days (Apr 30)
Watch: +50% volume on bad news today = distribution signal

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ AXIOM v4 | Not financial advice.
```

### 🚨 Signal Alert (instant, when confidence > 55%)
```
🚨 AXIOM SIGNAL ALERT
━━━━━━━━━━━━━━━━━━━━
✅ NVDA — STRONG BUY (54%)
💰 $178.10 → Target $281.04 (+57.8%)
📊 LSTM 5d: +5.1% | LGB 20d: +6.6%
📰 News: +0.005 (neutral-positive)
🎯 Scenarios:
   Bull: $210 (+17.9%) — data center beat
   Base: $187 (+5.0%) — LSTM trajectory
   Bear: $165 (−7.4%) — export controls
⚠️ Not financial advice.
```

### 📊 EOD Summary (4:00 PM PDT, Mon–Fri)
```
📊 AXIOM EOD | Apr 7, 2026
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▲ PLTR  $150.07  +1.45% | Vol: 27.7M (−31% avg) | Near MA20
▼ AAPL  $253.50  −2.07% | Vol: 61M  (+50% avg) | Foldable delay ⚠️
▲ NVDA  $178.10  +0.26% | Vol: 125M (−27% avg) | Consolidating
▼ TSLA  $346.65  −1.75% | Vol: 70M  (+9% avg)  | −8.5% below MA20

⚠️ TSLA earnings in 15 days — JPMorgan 60% downside warning active
Earnings calendar: TSLA Apr22 | AAPL Apr30 | PLTR May4 | NVDA May20
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ Not financial advice.
```

### 📅 Earnings Pre-Brief (2 days before each print)
```
📅 AXIOM EARNINGS ALERT
━━━━━━━━━━━━━━━━━━━━━━
TSLA reports Q1 2026 on Apr 22 (2 days)

Current setup:
💰 $346.65 | ML Signal: HOLD (29%)
📊 −8.5% below MA20 | JPMorgan: SELL $145
📰 News sentiment: −0.303 (bearish)
📉 Last print: EPS miss −$0.14 → stock −3.5%
⚠️ SELL-THE-NEWS RISK: EPS miss pattern 4/5 quarters

Expected move: ±4.8% (options market)
Historical avg reaction: −6.2% (last 5 prints)
⚠️ Not financial advice.
```

---

## Heartbeat Schedule (copy into ~/.openclaw/config.json)

```json
{
  "heartbeats": [
    {
      "name": "axiom-premarket",
      "schedule": "0 13 * * 1-5",
      "skill": "axiom-alerts",
      "task": "Run the AXIOM pre-market brief. Fetch live news from Google News RSS for PLTR, AAPL, NVDA, TSLA. Score each headline with VADER sentiment. Combine with ML signals from trading_system/signals/current_signals_v4.json. Generate intraday direction forecast per ticker. Format as the pre-market brief template and send via configured channel. Include earnings countdown for any ticker within 20 days."
    },
    {
      "name": "axiom-signal-check",
      "schedule": "0 */4 * * 1-5",
      "skill": "axiom-alerts",
      "task": "Check trading_system/signals/current_signals_v4.json. For any ticker with signal confidence > 0.55 AND signal is BUY, STRONG BUY, or SELL: fetch 2 latest news headlines, run scripts/run_signal.py --ticker {TICKER} --format telegram, and send the signal alert. Skip HOLD signals unless the signal changed since last check. Store last signal state in ~/.axiom_last_signals.json."
    },
    {
      "name": "axiom-eod",
      "schedule": "0 23 * * 1-5",
      "skill": "axiom-alerts",
      "task": "Run EOD summary. Fetch today's closing prices and % moves via web search for PLTR, AAPL, NVDA, TSLA. Read technical snapshot from cron_tracking if available. Identify the top catalyst for any move > 1.5%. Update earnings countdown. Send EOD summary via configured channel."
    },
    {
      "name": "axiom-earnings-watch",
      "schedule": "0 20 * * 1-5",
      "skill": "axiom-alerts",
      "task": "Check earnings calendar: TSLA Apr22, AAPL Apr30, PLTR May4, NVDA May20. If any ticker is within 3 days of earnings: send earnings pre-brief with historical post-earnings reaction, EPS surprise history from trading_system/signals/current_signals_v4.json earnings_context, current ML signal, options-implied expected move (search web for 'TICKER expected move earnings options'). Flag sell-the-news risk if P/E > 200x."
    }
  ]
}
```

---

## Manual Commands (ask OpenClaw directly)

| You say | OpenClaw does |
|---------|--------------|
| "What's the NVDA signal?" | Reads v4 JSON, formats signal summary |
| "Any news on TSLA today?" | Fetches Google News RSS, VADER-scores it, reports |
| "Pre-market brief" | Runs full brief on demand |
| "How far until PLTR earnings?" | Checks calendar, returns countdown + last reaction |
| "Run the full scan" | Runs all 4 tickers, sends complete report |
| "NVDA bull vs bear case" | Pulls from signal scenarios endpoint |
| "Is TSLA overbought?" | Checks RSI, MA position, reports |

---

## Data Flow (how it works end to end)

```
Google News RSS (live headlines)
        │
        ▼ VADER NLP
  Sentiment Score
        │
        ├──────────────────────────┐
        │                         │
        ▼                         ▼
  current_signals_v4.json    scripts/run_signal.py
  (LSTM/XGB/LGB/PPO/Meta)   (quick formatter)
        │                         │
        └────────────┬────────────┘
                     │
                     ▼
             OpenClaw Skill
                     │
                     ▼
        Your Phone / Laptop
     (Telegram / WhatsApp / Discord
       Signal / iMessage / Slack)
```

---

## File Paths OpenClaw Needs Access To

| File | Purpose |
|------|---------|
| `trading_system/signals/current_signals_v4.json` | Latest ML signals |
| `trading_system/signals/live_news_v4.json` | Latest news scan |
| `scripts/run_signal.py` | Signal formatter |
| `scripts/news_scanner.py` | News fetcher |
| `cron_tracking/aa62c16e/eod_report_*.md` | EOD data from Perplexity crons |

---

⚠️ This system is for research and educational purposes only. Nothing here is financial advice. Always do your own research before making any investment decisions.
