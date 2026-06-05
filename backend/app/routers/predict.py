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

    forecast = ModelService.get_forecast(ticker)
    if forecast:
        signal = forecast["signal"]
        news = forecast.get("news") or await NewsService.fetch(ticker)
        trend_snapshot = forecast.get("trend_snapshot", {})
        probability_up = float(signal.get("probability_up", 50.0)) / 100.0
        trust = float(signal.get("trust_score", 50.0)) / 100.0
        news_score = float(news.get("net_score", 0.0))
        trend_score = float(trend_snapshot.get("score", 0.0))
        score = (probability_up - 0.5) * 2.2 + news_score * 0.18 + (trust - 0.5) * 0.8 + trend_score * 0.9
        direction = "UP" if score > 0.15 else "DOWN" if score < -0.15 else "FLAT"
        confidence = min(0.95, max(0.15, abs(score) * 0.55 + trust * 0.45))
        price = float(signal["current_price"])
        expected_return = float(forecast["horizons"]["1d"]["expected_return_pct"]) / 100.0
        daily_range = max(price * 0.012, abs(expected_return) * price * 1.3)
        articles = news.get("articles", [])
        return {
            "ticker": ticker,
            "generated_at": datetime.now().isoformat(),
            "direction": direction,
            "confidence": round(confidence, 3),
            "expected_range_lo": round(price - daily_range, 2),
            "expected_range_hi": round(price + daily_range, 2),
            "catalyst": news.get("summary") or (articles[0]["headline"] if articles else "No catalyst identified"),
            "news_sentiment": news.get("overall_sentiment", "NEUTRAL"),
            "ml_signal": signal["signal"],
            "macd_direction": "BULLISH" if probability_up >= 0.5 else "BEARISH",
            "rsi": round(float(forecast["technical_snapshot"].get("rsi14", 50.0)), 1),
            "vol_regime": "HIGH_VOL" if float(forecast["technical_snapshot"].get("ret_20d_pct", 0.0)) < -8 else "MED_VOL",
            "trend_state": trend_snapshot.get("state", "MIXED"),
            "trend_score": round(trend_score, 4),
            "trend_supported": bool(trend_snapshot.get("trend_supported", True)),
            "score": round(score, 2),
        }

    sig  = ModelService.get_cached_signal(ticker)
    news = await NewsService.fetch(ticker)

    if not sig:
        raise HTTPException(503, "Signal unavailable")

    # Combine ML signal + news to produce intraday direction
    ml_int       = sig.get("signal_int", 0)
    news_impact  = news.get("intraday_impact", "FLAT")
    rsi          = sig.get("rsi14", 50)
    macd_h       = sig.get("macd_hist", sig.get("macd_h", 0))
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

    forecast = ModelService.get_forecast(ticker)
    if forecast:
        signal = forecast["signal"]
        trend_snapshot = forecast.get("trend_snapshot", {})
        week_targets = []
        for horizon_key, label in (("1d", 1), ("5d", 5), ("10d", 10)):
            horizon = forecast["horizons"][horizon_key]
            week_targets.append({
                "week": label,
                "price": horizon["target_price"],
                "pct_chg": horizon["expected_return_pct"],
                "direction": "UP" if horizon["expected_return_pct"] > 0.25 else "DOWN" if horizon["expected_return_pct"] < -0.25 else "FLAT",
                "probability_up": horizon["probability_up"],
                "trust_score": horizon["trust_score"],
                "trend_supported": horizon.get("trend_supported", True),
            })

        return {
            "ticker": ticker,
            "generated_at": datetime.now().isoformat(),
            "current_price": signal["current_price"],
            "week_targets": week_targets,
            "lgb_4w_est": round(float(forecast["horizons"]["10d"]["expected_return_pct"]), 2),
            "model_signal": signal["signal"],
            "conviction": "HIGH" if float(signal["trust_score"]) >= 65 else "MEDIUM" if float(signal["trust_score"]) >= 50 else "LOW",
            "analyst_target": signal["target_price"],
            "analyst_upside": round(((float(signal["target_price"]) / float(signal["current_price"])) - 1.0) * 100.0, 2),
            "trend_state": trend_snapshot.get("state", "MIXED"),
            "trend_score": trend_snapshot.get("score", 0.0),
        }

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

    forecast = ModelService.get_forecast(ticker)
    if forecast:
        signal = forecast["signal"]
        current_price = float(signal["current_price"])
        one_day = forecast["horizons"]["1d"]
        five_day = forecast["horizons"]["5d"]
        ten_day = forecast["horizons"]["10d"]
        trend_snapshot = forecast.get("trend_snapshot", {})
        return {
            "ticker": ticker,
            "current_price": current_price,
            "horizon": "1 to 10 trading days",
            "trend_state": trend_snapshot.get("state", "MIXED"),
            "trend_score": trend_snapshot.get("score", 0.0),
            "scenarios": {
                "bull": {
                    "price": round(current_price * (1.0 + max(ten_day["expected_return_pct"], five_day["expected_return_pct"]) / 100.0), 2),
                    "pct": round(max(ten_day["expected_return_pct"], five_day["expected_return_pct"]), 1),
                    "catalyst": "Supportive calibration bucket plus constructive tape and aligned daily trend",
                    "probability": round(max(one_day["probability_up"], five_day["probability_up"], ten_day["probability_up"]) / 100.0, 2),
                },
                "base": {
                    "price": five_day["target_price"],
                    "pct": round(five_day["expected_return_pct"], 1),
                    "catalyst": "Champion base path from calibrated walk-forward probabilities and current trend state",
                    "probability": 0.45,
                },
                "bear": {
                    "price": round(current_price * (1.0 - max(2.0, abs(min(one_day["expected_return_pct"], five_day["expected_return_pct"]))) / 100.0), 2),
                    "pct": round(-max(2.0, abs(min(one_day["expected_return_pct"], five_day["expected_return_pct"]))), 1),
                    "catalyst": "Confidence bucket fails, news reverses, or the daily trend remains damaged",
                    "probability": round(max(0.05, 1.0 - five_day["probability_up"] / 100.0 - 0.20), 2),
                },
            },
        }

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
