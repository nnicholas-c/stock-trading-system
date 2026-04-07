"""
AXIOM Trading System — FastAPI Backend
Production-grade API serving ML signals, LSTM forecasts, news, and RL decisions.
Deploy to: Railway / Render / Fly.io (free tier)
"""

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio

from app.core.config import settings
from app.routers import signals, predict, news, backtest, health
from app.services.model_service import ModelService

# ── Startup: load all models into memory once ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 AXIOM backend starting — loading models...")
    await ModelService.initialize()
    print("✅ All models loaded")
    yield
    print("👋 AXIOM backend shutting down")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AXIOM Trading Intelligence API",
    description="Hedge-fund grade ML/RL trading signals for PLTR, AAPL, NVDA, TSLA",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS (allow GitHub Pages + local dev) ─────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nnicholas-c.github.io",
        "http://localhost:3000",
        "http://localhost:5173",
        "*",  # tighten in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router,   prefix="/health",   tags=["Health"])
app.include_router(signals.router,  prefix="/signals",  tags=["Signals"])
app.include_router(predict.router,  prefix="/predict",  tags=["Predictions"])
app.include_router(news.router,     prefix="/news",     tags=["News"])
app.include_router(backtest.router, prefix="/backtest", tags=["Backtest"])

@app.get("/")
async def root():
    return {
        "name": "AXIOM Trading Intelligence API",
        "version": "2.0.0",
        "status": "operational",
        "tickers": ["PLTR", "AAPL", "NVDA", "TSLA"],
        "endpoints": {
            "signals":  "/signals/{ticker}",
            "predict":  "/predict/{ticker}",
            "news":     "/news/{ticker}",
            "backtest": "/backtest/{ticker}",
            "health":   "/health",
            "docs":     "/docs",
        }
    }
