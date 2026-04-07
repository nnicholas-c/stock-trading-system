"""GET /signals/{ticker} — return latest ML signal from cache or re-run model."""

from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import Optional

from app.models.schemas import SignalResponse
from app.services.model_service import ModelService
from app.core.config import settings

router = APIRouter()

@router.get("/", summary="All 4 ticker signals")
async def get_all_signals():
    signals = {}
    for ticker in settings.tickers:
        sig = ModelService.get_cached_signal(ticker)
        if sig:
            signals[ticker] = sig
    return {"generated_at": datetime.now().isoformat(), "signals": signals}


@router.get("/{ticker}", summary="Signal for a specific ticker")
async def get_signal(
    ticker: str,
    refresh: bool = Query(False, description="Force model re-run")
):
    ticker = ticker.upper()
    if ticker not in settings.tickers:
        raise HTTPException(404, f"Ticker {ticker} not supported. Use: {settings.tickers}")

    sig = ModelService.get_cached_signal(ticker)
    if not sig:
        raise HTTPException(503, "Signal not yet generated. Try again in a moment.")

    # Enrich with current timestamp
    sig["fetched_at"] = datetime.now().isoformat()
    return sig


@router.get("/{ticker}/summary", summary="One-line signal summary for OpenClaw/alerts")
async def get_signal_summary(ticker: str):
    ticker = ticker.upper()
    sig = ModelService.get_cached_signal(ticker)
    if not sig:
        raise HTTPException(503, "Signal unavailable")

    fc = sig.get("lstm_forecast_4w", [])
    lstm_4w = ((fc[-1] / sig["price"] - 1) * 100) if fc else 0
    lgb = sig.get("lgb_fwd_ret", 0) * 100

    return {
        "ticker":     ticker,
        "signal":     sig["signal"],
        "confidence": f"{sig['confidence']:.0%}",
        "price":      f"${sig['price']:.2f}",
        "target":     f"${sig['analyst_target']:.2f}",
        "upside":     f"{sig['analyst_upside']:+.1f}%",
        "lstm_4w":    f"{lstm_4w:+.1f}%",
        "lgb_est":    f"{lgb:+.1f}%",
        "regime":     sig.get("vol_regime", "—"),
        "risk":       f"{sig.get('risk_score', 5):.1f}/10",
        "alert_text": (
            f"{sig['signal']} {ticker} @ ${sig['price']:.2f} | "
            f"Conf {sig['confidence']:.0%} | "
            f"Target ${sig['analyst_target']:.2f} ({sig['analyst_upside']:+.1f}%) | "
            f"LSTM 4w: {lstm_4w:+.1f}% | LGB: {lgb:+.1f}%"
        )
    }
