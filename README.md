# 📈 AI Stock Trading System — PLTR | AAPL | NVDA | TSLA

A professional-grade ML/RL stock analysis and alerting system built with Perplexity Computer.  
Combines **Random Forest + Gradient Boosting** signal generation, **PPO reinforcement learning** portfolio management, and **live news monitoring** — all delivered to your phone via **OpenClaw**.

---

## 🚀 Live Signals (as of April 7, 2026)

| Ticker | Signal | Confidence | Price | Analyst Target | Upside | Risk |
|--------|--------|-----------|-------|---------------|--------|------|
| PLTR | 🚀 STRONG BUY | 89.1% | $150.07 | $197.57 | +31.7% | 10.0/10 |
| AAPL | ✅ STRONG BUY | 49.5% | $253.50 | $306.25 | +20.8% | 3.7/10 |
| NVDA | ✅ STRONG BUY | 64.3% | $178.10 | $281.04 | +57.8% | 4.2/10 |
| TSLA | 🔴 SELL | 62.3% | $346.65 | $416.49 | +20.1% | 8.7/10 |

> ⚠️ Not financial advice. Always do your own research.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
│  Perplexity Finance API → Weekly OHLCV (2022–2026)         │
│  Earnings Transcripts • Institutional Holders • Insiders   │
│  PitchBook • CB Insights • Statista Market Data            │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   ML/RL ENGINE                              │
│                                                             │
│  Feature Engineering (51 indicators)                       │
│    └── RSI, MACD, Bollinger, ATR, OBV, Momentum,          │
│        Stochastic, Volume signals, Fundamentals            │
│                                                             │
│  Layer 1: Random Forest (200 trees, depth 6)               │
│  Layer 2: Gradient Boosting (100 trees, lr 0.05)           │
│  Ensemble: 60% RF + 40% GB weighted probabilities          │
│    └── Output: STRONG BUY / BUY / HOLD / SELL             │
│                                                             │
│  Layer 3: PPO Reinforcement Learning Agent                  │
│    └── 6 actions: Hold / Buy 25/50/100% / Sell 50/All     │
│    └── Reward: Risk-adjusted returns + Sharpe bonus        │
│    └── MLP policy: 128→64→32 hidden layers                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   ALERT LAYER                               │
│                                                             │
│  OpenClaw (runs on your machine)                           │
│    ├── Heartbeat every 6h → runs signal engine             │
│    ├── Pre-market brief (9:30am ET weekdays)               │
│    ├── Post-market summary (4:00pm ET weekdays)            │
│    └── Earnings monitor (checks weekly)                    │
│                                                             │
│  Delivers to: Telegram / WhatsApp / Discord /              │
│               Signal / iMessage (your choice)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Backtest Results (Walk-Forward, No Look-Ahead Bias)

| Ticker | ML Strategy | Buy & Hold | Alpha | Sharpe | Max DD |
|--------|------------|-----------|-------|--------|--------|
| PLTR | +187.9% | +125.0% | **+62.9%** | 1.65 | — |
| AAPL | +69.9% | +5.4% | **+64.5%** | 2.01 | — |
| NVDA | +111.1% | +30.4% | **+80.7%** | 2.17 | — |
| TSLA | +90.8% | +13.3% | **+77.5%** | 1.87 | — |

## 🤖 RL Agent Performance (PPO, Full Dataset)

| Ticker | Total Return | Sharpe | Max Drawdown |
|--------|-------------|--------|-------------|
| PLTR | +788.0% | 1.97 | -32.0% |
| AAPL | +72.7% | 1.08 | -13.9% |
| NVDA | +457.1% | 2.07 | -32.1% |
| TSLA | +87.8% | 0.72 | -42.0% |

---

## 🔧 Installation

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/stock-trading-system
cd stock-trading-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the ML engine (generates signals + trains models)
python ml_trading_system.py

# 4. View current signals
python scripts/run_signal.py --format plain

# 5. Scan live news
python scripts/news_scanner.py --format text
```

---

## 📱 OpenClaw Integration (Get Alerts on Your Phone)

OpenClaw is an open-source personal AI agent that runs on your machine and delivers alerts via chat apps.

```bash
# Install OpenClaw
# See https://openclaw.ai for installation

# Copy skill config
cp openclaw/openclaw-config.json ~/.openclaw/config.json
cp openclaw/skills/stock-alerts.md ~/.openclaw/skills/

# Configure your notification channel in config/settings.json
# Set "notification_channel" to: telegram / whatsapp / discord / signal

# OpenClaw will now automatically:
# • Check signals every 6 hours
# • Send pre-market briefs at 9:30am ET
# • Send end-of-day summaries at 4pm ET
# • Alert on earnings dates
```

**Example alert you'll receive:**
```
🚨 TRADING SIGNAL ALERT
━━━━━━━━━━━━━━━━━━━━━━
✅ NVDA — STRONG BUY (64% confidence)
💰 $178.10 → Target $281.04 (+57.8%)
📊 RSI: 48.2 | MACD: Bullish | Risk: Moderate
🧠 Key drivers: macd_signal, realized_vol_26w, bb_width_13
👥 100% analyst consensus: Strong Buy
📰 "Blackwell NVL72 demand exceeds supply through Q3"
━━━━━━━━━━━━━━━━━━━━━━
⚠️ Not financial advice. DYOR.
```

---

## 📁 Repository Structure

```
stock-trading-system/
├── ml_trading_system.py         # Core ML/RL engine
├── requirements.txt
├── README.md
├── config/
│   └── settings.json            # Configuration
├── scripts/
│   ├── run_signal.py            # Quick signal runner (called by OpenClaw)
│   └── news_scanner.py          # Live news scanner with sentiment scoring
├── openclaw/
│   ├── openclaw-config.json     # OpenClaw heartbeat + skill config
│   └── skills/
│       └── stock-alerts.md      # OpenClaw skill definition
├── trading_system/              # Generated at runtime
│   ├── signals/
│   │   ├── current_signals.json # Latest ML signals
│   │   └── news_scan.json       # Latest news scan
│   ├── models/                  # Saved RF, GB, PPO models (.pkl / .zip)
│   └── charts/                  # Analysis charts (PNG)
└── data/                        # Price history CSVs
```

---

## 📐 Feature Engineering (51 indicators)

**Price Returns:** 1w, 2w, 4w, 8w, 13w, 26w, 52w  
**Moving Averages:** SMA/EMA vs price (4, 8, 13, 26, 52 week)  
**RSI:** 7, 14, 21 period  
**MACD:** Standard + histogram + crossover signal  
**Bollinger Bands:** Width + %B (13w, 26w)  
**ATR / Volatility:** 7w, 14w ATR; realized vol 13w, 26w  
**Volume:** Ratio, trend, OBV vs SMA, money flow  
**Stochastic:** 7w, 14w  
**Momentum:** 4w, 13w, 26w + acceleration (2nd derivative)  
**Candlestick:** Body size, upper/lower wick, direction  
**Fundamentals:** Analyst upside %, bull consensus %, revenue growth, gross margin, sentiment score, insider activity, Rule of 40  

---

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is **not financial advice**. Past backtest performance does not guarantee future results. All investment decisions carry risk. Never invest more than you can afford to lose. Consult a qualified financial advisor before making investment decisions.

---

*Built with [Perplexity Computer](https://perplexity.ai) • Data: Perplexity Finance, PitchBook, CB Insights, Statista*
