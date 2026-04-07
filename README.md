# AXIOM — Hedge Fund Grade AI Trading System

> Full-stack ML/RL trading intelligence for PLTR, AAPL, NVDA, TSLA.
> Bloomberg Terminal dashboard · FastAPI backend · OpenClaw alerts · iOS app scaffold.

---

## Live Links
| | |
|--|--|
| **Dashboard** | https://nnicholas-c.github.io/stock-trading-system/ |
| **API Docs** | https://YOUR-APP.railway.app/docs |
| **GitHub** | https://github.com/nnicholas-c/stock-trading-system |

---

## Architecture

```
axiom/
├── backend/                  ← FastAPI Python backend
│   ├── app/
│   │   ├── main.py           ← App entry, CORS, router registration
│   │   ├── core/config.py    ← Settings (paths, tickers, TTLs)
│   │   ├── models/schemas.py ← Pydantic request/response contracts
│   │   ├── routers/
│   │   │   ├── signals.py    ← GET /signals/{ticker}
│   │   │   ├── predict.py    ← GET /predict/{ticker}/intraday|weekly|scenarios
│   │   │   ├── news.py       ← GET /news/{ticker}
│   │   │   ├── backtest.py   ← GET /backtest/{ticker}
│   │   │   └── health.py     ← GET /health
│   │   └── services/
│   │       ├── model_service.py  ← Loads all models at startup, serves inference
│   │       └── news_service.py   ← Async news fetch + cache + sentiment scoring
│   ├── Dockerfile            ← Railway/Render/Fly.io deployment
│   └── railway.toml          ← One-click Railway deployment config
│
├── frontend/                 ← Static dashboard (GitHub Pages)
│   └── index.html            ← Bloomberg Terminal UI — Chart.js + vanilla JS
│
├── openclaw/                 ← OpenClaw AI integration
│   └── skills/axiom-alerts.md ← Skill: calls the API, sends Telegram/WhatsApp alerts
│
├── ios/                      ← Native iOS SwiftUI app scaffold
│   └── AXIOM/
│       ├── Views/DashboardView.swift    ← All screens (SwiftUI + Swift Charts)
│       ├── Models/DataModels.swift      ← Codable structs
│       ├── Services/APIService.swift    ← Async/await API client
│       └── Services/NotificationService.swift ← APNs + local alerts
│
├── ml_trading_system.py      ← Original ML system (RF + PPO)
├── prediction_engine.py      ← v2 engine (LSTM + XGB + LGB + Meta)
├── data/                     ← Weekly OHLCV CSVs (2022–2026)
└── trading_system/
    ├── models/               ← All trained model files (.pkl, .pt, .zip)
    ├── signals/              ← Latest signal JSON outputs
    └── charts/               ← Analysis charts (PNG)
```

---

## Model Suite

| Model | Library | Purpose | Performance |
|-------|---------|---------|-------------|
| LSTM Forecaster | PyTorch | 4-week price trajectory | val_loss 0.004 |
| XGBoost Classifier | xgboost | Buy/Sell/Hold signal | 26–59% test acc |
| LightGBM Regressor | lightgbm | Forward return magnitude | MAE 0.04–0.25 |
| Random Forest | sklearn | Baseline signal | Ensemble component |
| Meta-Ensemble | sklearn | Stack RF+XGB probas | 41–62% acc |
| PPO RL Agent | stable-baselines3 | Portfolio sizing (6 actions) | Sharpe 0.7–2.1 |
| Vol Regime | custom | LOW/MED/HIGH classification | Context signal |

**65 features:** Price returns (7 horizons), SMA/EMA vs price, RSI (3 periods), MACD + histogram, Bollinger Bands, ATR, Realized volatility (2 windows), OBV, Volume ratio, Stochastic (2 periods), Momentum (4 horizons + acceleration + jerk), Donchian channels, Candlestick proxies, Fundamentals (7 signals).

---

## Backtest Results (walk-forward, zero look-ahead)

| Ticker | ML Strategy | Buy & Hold | Alpha | Sharpe | RL Return |
|--------|------------|-----------|-------|--------|-----------|
| PLTR | +187.9% | +125.0% | **+62.9%** | 1.65 | +788.0% |
| AAPL | +69.9% | +5.4% | **+64.5%** | 2.01 | +72.7% |
| NVDA | +111.1% | +30.4% | **+80.7%** | 2.17 | +457.1% |
| TSLA | +90.8% | +13.3% | **+77.5%** | 1.87 | +87.8% |

---

## Running the Backend Locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

## Deploying to Railway (free tier)
```bash
npm install -g @railway/cli
railway login
railway init
railway up
# Your API is live at https://YOUR-APP.railway.app
```

## Running the Training Pipeline
```bash
# Install deps
pip install -r requirements.txt

# v2: LSTM + XGBoost + LightGBM + Meta-ensemble
python prediction_engine.py

# v1: RF + PPO RL
python ml_trading_system.py
```

## OpenClaw Integration
```bash
# Install OpenClaw: https://openclaw.ai
# Copy skill:
cp openclaw/skills/axiom-alerts.md ~/.openclaw/skills/

# Edit the skill file and set your API URL:
# axiom_api: "https://YOUR-APP.railway.app"

# OpenClaw will now automatically:
# • 6am PDT: Pre-market brief via Telegram/WhatsApp/Discord
# • Every 4h: Signal scan, alert if high-confidence BUY/SELL
# • 4pm PDT: EOD summary + weekly predictions
```

## iOS App
See `ios/README-iOS.md` for complete Xcode setup instructions.

---

## Scheduled Intelligence (active)
| Schedule | Task |
|----------|------|
| 6:00am PDT weekdays | Pre-market brief + intraday direction prediction |
| 4:00pm PDT weekdays | EOD summary + 4-week outlook update |

---

## Disclaimer
**This software is for educational and research purposes only. It is NOT financial advice.
Past backtest performance does not guarantee future results. Never invest more than you can afford to lose.**

*Built with [Perplexity Computer](https://perplexity.ai) · Data: Perplexity Finance, PitchBook, CB Insights, Statista*
