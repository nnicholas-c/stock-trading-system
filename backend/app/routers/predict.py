"""
GET /predict/{ticker}/intraday  — today's direction prediction
GET /predict/{ticker}/weekly    — 4-week LSTM trajectory
GET /predict/{ticker}/scenarios — bull/base/bear price scenarios
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime

from app.services.model_service import ModelService
from app.services.news_service import NewsService
from app.core.config import settings

router = APIRouter()


@router.get("/{ticker}/intraday", summary="Intraday direction prediction")
async def predict_intraday(ticker: str):
    ticker = ticker.upper()
    if ticker not in settings.tickers:
        raise HTTPException(404, f"Ticker not supported")

    sig  = ModelService.get_cached_signal(ticker)
    news = await NewsService.fetch(ticker)

    if not sig:
        raise HTTPException(503, "Signal unavailable")

    # Combine ML signal + news to produce intraday direction
    ml_int       = sig.get("signal_int", 0)
    news_impact  = news.get("intraday_impact", "FLAT")
    rsi          = sig.get("rsi14", 50)
    macd_h       = sig.get("macd_hist", 0)
    regime       = sig.get("vol_regime", "MED_VOL")

    # Scoring model
    score = 0
    score += ml_int * 1.5                   # ML signal weight
    score += 1 if news_impact == "UP"   else (-1 if news_impact == "DOWN" else 0)
    score += 0.5 if macd_h > 0          else -0.5
    score += 0.3 if rsi < 40            else (-0.3 if rsi > 70 else 0)  # mean reversion

    direction  = "UP" if score > 0.5 else "DOWN" if score < -0.5 else "FLAT"
    confidence = min(0.95, abs(score) / 4)

    price = sig["price"]
    vol_mult = 1.5 if regime == "HIGH_VOL" else 0.8 if regime == "LOW_VOL" else 1.0
    daily_range = price * 0.02 * vol_mult

    return {
        "ticker":             ticker,
        "generated_at":       datetime.now().isoformat(),
        "direction":          direction,
        "confidence":         round(confidence, 3),
        "expected_range_lo":  round(price - daily_range, 2),
        "expected_range_hi":  round(price + daily_range, 2),
        "catalyst":           news["articles"][0]["headline"] if news["articles"] else "No catalyst identified",
        "news_sentiment":     news["overall_sentiment"],
        "ml_signal":          sig["signal"],
        "macd_direction":     "BULLISH" if macd_h > 0 else "BEARISH",
        "rsi":                round(rsi, 1),
        "vol_regime":         regime,
        "score":              round(score, 2),
    }


@router.get("/{ticker}/weekly", summary="4-week LSTM price trajectory")
async def predict_weekly(ticker: str):
    ticker = ticker.upper()
    if ticker not in settings.tickers:
        raise HTTPException(404)

    sig = ModelService.get_cached_signal(ticker)
    if not sig:
        raise HTTPException(503, "Signal unavailable")

    fc    = sig.get("lstm_forecast_4w", [])
    price = sig["price"]
    lgb   = sig.get("lgb_fwd_ret", 0)

    weeks = []
    for i, p in enumerate(fc):
        pct = (p / price - 1) * 100
        weeks.append({
            "week":      i + 1,
            "price":     round(p, 2),
            "pct_chg":   round(pct, 2),
            "direction": "UP" if pct > 0.5 else "DOWN" if pct < -0.5 else "FLAT",
        })

    # Conviction level
    if fc:
        total_chg = abs((fc[-1] / price - 1) * 100)
        conviction = "HIGH" if total_chg > 8 else "MEDIUM" if total_chg > 3 else "LOW"
    else:
        conviction = "LOW"

    return {
        "ticker":        ticker,
        "generated_at":  datetime.now().isoformat(),
        "current_price": price,
        "week_targets":  weeks,
        "lgb_4w_est":    round(lgb * 100, 2),
        "model_signal":  sig["signal"],
        "conviction":    conviction,
        "analyst_target":sig["analyst_target"],
        "analyst_upside":sig["analyst_upside"],
    }


@router.get("/{ticker}/scenarios", summary="Bull / Base / Bear price scenarios")
async def predict_scenarios(ticker: str):
    ticker = ticker.upper()
    if ticker not in settings.tickers:
        raise HTTPException(404)

    sig = ModelService.get_cached_signal(ticker)
    if not sig:
        raise HTTPException(503)

    price  = sig["price"]
    target = sig["analyst_target"]
    fc     = sig.get("lstm_forecast_4w", [price]*4)
    lgb    = sig.get("lgb_fwd_ret", 0)

    lstm_4w = fc[-1] if fc else price

    bull_mult = 1 + max(lgb * 2, (lstm_4w / price - 1) * 1.3, (target / price - 1) * 0.8)
    base_mult = 1 + (lstm_4w / price - 1)
    bear_mult = 1 - max(0.08, abs(sig.get("risk_score", 5)) * 0.015)

    return {
        "ticker":       ticker,
        "current_price": price,
        "horizon":      "4 weeks",
        "scenarios": {
            "bull": {
                "price":       round(price * bull_mult, 2),
                "pct":         round((bull_mult - 1) * 100, 1),
                "catalyst":    "Earnings beat + macro tailwind",
                "probability": round(sig["bull_pct"] / 100, 2),
            },
            "base": {
                "price":       round(price * base_mult, 2),
                "pct":         round((base_mult - 1) * 100, 1),
                "catalyst":    "LSTM trajectory, status quo",
                "probability": 0.45,
            },
            "bear": {
                "price":       round(price * bear_mult, 2),
                "pct":         round((bear_mult - 1) * 100, 1),
                "catalyst":    "Macro deterioration + sector rotation",
                "probability": round(1 - sig["bull_pct"] / 100 - 0.45, 2),
            },
        },
    }
