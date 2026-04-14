#!/usr/bin/env python3
"""
Build conservative next-session forecasts for the GitHub UI.

This combines:
- refreshed local daily market data
- the PLTR deep model output for PLTR
- refreshed DRL live signals for AAPL / NVDA / TSLA
- live post-close / pre-open news fetched from Google News RSS

The forecast is intentionally trust-calibrated: when model, technicals, and
news disagree, the final directional edge is shrunk back toward neutral.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pltr_premarket_context import fetch_live_premarket_context, write_json


DATA_DIR = ROOT / "data" / "daily"
DRL_RESULTS = ROOT / "trading_system" / "drl" / "drl_v2_results.json"
PLTR_SIGNAL = ROOT / "trading_system" / "pltr_deep" / "pltr_signal.json"
PLTR_RESULTS = ROOT / "trading_system" / "pltr_deep" / "pltr_deep_results.json"

SIGNALS_DIR = ROOT / "trading_system" / "signals"
OUT_FILE = SIGNALS_DIR / "tomorrow_premarket_forecast.json"
DOCS_FILE = ROOT / "docs" / "live_tomorrow_forecasts.json"
BLOOMBERG_FILE = ROOT / "bloomberg" / "live_tomorrow_forecasts.json"

COMPANIES = {
    "PLTR": "Palantir",
    "AAPL": "Apple",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
}

IMPACT_SHORT = {"HIGH": "H", "MEDIUM": "M", "LOW": "L"}
INTEL_TYPE = {
    "model": "mdl",
    "news": "nws",
    "technical": "tec",
    "risk": "mac",
}


def clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def format_age(iso_value: str | None, now: datetime) -> str:
    if not iso_value:
        return "now"
    try:
        then = datetime.fromisoformat(iso_value)
    except Exception:
        return "now"
    diff_hours = max(0.0, (now - then).total_seconds() / 3600.0)
    if diff_hours < 1:
        return f"{max(1, round(diff_hours * 60))}m"
    if diff_hours < 24:
        return f"{round(diff_hours)}h"
    return f"{round(diff_hours / 24)}d"


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(period).mean()
    losses = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gains / (losses + 1e-10)
    return 100 - (100 / (1 + rs))


def build_technical_snapshot(ticker: str) -> dict[str, float | str]:
    frame = pd.read_csv(DATA_DIR / f"{ticker}_daily.csv")
    frame["date"] = pd.to_datetime(frame["date"])
    close = frame["close"].astype(float)
    returns = close.pct_change().fillna(0.0)

    price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) > 1 else price
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ret_1d = float(close.pct_change(1).iloc[-1]) if len(close) > 1 else 0.0
    ret_5d = float(close.pct_change(5).iloc[-1]) if len(close) > 5 else ret_1d
    ret_20d = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else ret_5d
    daily_vol = float(returns.tail(10).std()) if len(returns) >= 10 else 0.01
    rsi14 = float(compute_rsi(close, 14).iloc[-1]) if len(close) >= 15 else 50.0
    volume = frame["volume"].astype(float)
    vol_ratio = float(volume.iloc[-1] / (volume.tail(20).mean() + 1e-9)) if len(volume) >= 20 else 1.0

    short_trend = clip(((price / (ma20 + 1e-9)) - 1.0) / 0.05, -1.0, 1.0)
    medium_trend = clip(((price / (ma50 + 1e-9)) - 1.0) / 0.08, -1.0, 1.0)
    momentum = clip((ret_1d / 0.04) * 0.45 + (ret_5d / 0.12) * 0.55, -1.0, 1.0)

    mean_reversion = 0.0
    if rsi14 < 42 and ret_5d < 0:
        mean_reversion = clip(((42 - rsi14) / 18.0) + ((-ret_5d) / 0.15), 0.0, 1.0)
    elif rsi14 > 65 and ret_5d > 0:
        mean_reversion = -clip(((rsi14 - 65) / 15.0) + (ret_5d / 0.12), 0.0, 1.0)

    technical_score = clip(
        0.35 * short_trend +
        0.20 * medium_trend +
        0.25 * momentum +
        0.20 * mean_reversion,
        -1.0,
        1.0,
    )
    move_scale = clip(max(daily_vol * 1.35, abs(ret_1d) * 0.70, 0.006), 0.006, 0.030)

    return {
        "market_date": frame["date"].iloc[-1].strftime("%Y-%m-%d"),
        "current_price": round(price, 4),
        "prev_close": round(prev_close, 4),
        "ma20": round(ma20, 4),
        "ma50": round(ma50, 4),
        "ret_1d": ret_1d,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "daily_vol": daily_vol,
        "rsi14": round(rsi14, 2),
        "volume_ratio": round(vol_ratio, 3),
        "short_trend": short_trend,
        "medium_trend": medium_trend,
        "momentum": momentum,
        "mean_reversion": mean_reversion,
        "technical_score": technical_score,
        "move_scale": move_scale,
        "pct_from_ma20": ((price / (ma20 + 1e-9)) - 1.0),
        "pct_from_ma50": ((price / (ma50 + 1e-9)) - 1.0),
    }


def score_from_probability(probability_up: float) -> float:
    return clip((probability_up - 0.5) / 0.35, -1.0, 1.0)


def load_model_component(ticker: str, snapshot: dict[str, float | str]) -> dict[str, Any]:
    if ticker == "PLTR":
        signal = load_json(PLTR_SIGNAL)
        probability_up = float(signal.get("probability_up", 50.0)) / 100.0
        return {
            "source": "pltr_deep_v1",
            "probability_up": probability_up,
            "score": score_from_probability(probability_up),
            "pred_return": float(signal.get("pred_return_pct", 0.0)) / 100.0,
            "trust": float(signal.get("trust_score", 50.0)) / 100.0,
            "confidence": float(signal.get("confidence", 0.0)) / 100.0,
            "raw": signal,
        }

    drl = load_json(DRL_RESULTS)
    signal = drl.get("signals", {}).get(ticker, {})
    xgb_prob_up = float(signal.get("xgb_prob_up", 50.0)) / 100.0
    drl_buy = float(signal.get("drl_proba_buy", 33.0)) / 100.0
    drl_sell = float(signal.get("drl_proba_sell", 33.0)) / 100.0
    combined_conf = float(signal.get("combined_conf", 50.0)) / 100.0

    xgb_score = clip((xgb_prob_up - 0.5) / 0.28, -1.0, 1.0)
    drl_score = clip((drl_buy - drl_sell) / 0.35, -1.0, 1.0)
    score = clip(0.60 * xgb_score + 0.40 * drl_score, -1.0, 1.0)
    agreement = 1.0 - min(1.0, abs(xgb_score - drl_score) / 2.0)
    trust = clip(
        0.32 +
        0.18 * agreement +
        0.12 * combined_conf +
        0.15 * abs(score),
        0.30,
        0.62,
    )

    return {
        "source": "drl_v2_live",
        "probability_up": clip(0.5 + 0.28 * score, 0.18, 0.82),
        "score": score,
        "pred_return": score * float(snapshot["move_scale"]),
        "trust": trust,
        "confidence": combined_conf,
        "raw": signal,
        "subscores": {
            "xgb_score": round(xgb_score, 4),
            "drl_score": round(drl_score, 4),
            "agreement": round(agreement, 4),
        },
    }


def build_news_score(live_context: dict[str, Any]) -> float:
    article_count = int(live_context.get("article_count", 0))
    material_count = int(live_context.get("material_count", 0))
    score = clip(float(live_context.get("net_score", 0.0)) / 2.5, -1.0, 1.0)
    if article_count == 0:
        return 0.0
    if live_context.get("used_recent_fallback"):
        score *= 0.55
    if article_count == 1:
        score *= 0.60
    elif article_count == 2:
        score *= 0.80
    score *= 1.0 + min(0.35, material_count * 0.10)
    return clip(score, -1.0, 1.0)


def technical_driver(snapshot: dict[str, float | str]) -> dict[str, str]:
    ret_5d = float(snapshot["ret_5d"]) * 100.0
    pct_from_ma20 = float(snapshot["pct_from_ma20"]) * 100.0
    rsi14 = float(snapshot["rsi14"])
    if pct_from_ma20 < -2.0 and ret_5d < -4.0:
        direction = "bull" if rsi14 < 42 else "bear"
        detail = (
            f"The stock is {pct_from_ma20:.1f}% below its 20-day average after a {ret_5d:.1f}% five-day move, "
            f"so the near-term setup is {'more mean-reversion than trend-following' if rsi14 < 42 else 'still trend-damaged'}."
        )
    elif pct_from_ma20 > 1.0 and ret_5d > 2.5:
        direction = "bull"
        detail = (
            f"Price is holding above the short-term trend anchor with a {ret_5d:.1f}% five-day move, "
            "which keeps the immediate tape constructive."
        )
    else:
        direction = "neutral"
        detail = (
            f"Technicals are mixed: RSI 14 is {rsi14:.1f} and price is {pct_from_ma20:.1f}% from the 20-day average, "
            "so the model avoids taking a large directional swing from price action alone."
        )
    return {"title": "Technical setup", "direction": direction, "detail": detail}


def model_driver(ticker: str, component: dict[str, Any]) -> dict[str, str]:
    prob_up = float(component["probability_up"]) * 100.0
    pred_return = float(component["pred_return"]) * 100.0
    if ticker == "PLTR":
        detail = (
            f"The retrained PLTR deep model is the main anchor here: it currently assigns {prob_up:.1f}% odds of an up day "
            f"with a {pred_return:+.2f}% 1-day move estimate."
        )
    else:
        raw = component.get("raw", {})
        detail = (
            f"The refreshed DRL + XGBoost live stack leans {prob_up:.1f}% up. "
            f"XGB is {raw.get('xgb_prob_up', 50.0):.1f}% up while DRL buy / sell reads "
            f"{raw.get('drl_proba_buy', 33.0):.1f}% / {raw.get('drl_proba_sell', 33.0):.1f}%."
        )
    direction = "bull" if prob_up > 55 else "bear" if prob_up < 45 else "neutral"
    return {"title": "Model anchor", "direction": direction, "detail": detail}


def trust_driver(trust: float, agreement: float) -> dict[str, str]:
    if trust >= 0.60:
        detail = "The model, price action, and current headline tape are aligned enough that the forecast does not need a heavy haircut."
        direction = "bull"
    elif trust >= 0.45:
        detail = "There is some alignment, but not enough to trust a large directional move, so the forecast stays deliberately conservative."
        direction = "neutral"
    else:
        detail = "Inputs are conflicting or weak, so the forecast is shrunk hard toward HOLD instead of forcing a false precision call."
        direction = "bear"
    return {
        "title": f"Trust calibration ({trust * 100:.1f}% trust, {agreement * 100:.0f}% agreement)",
        "direction": direction,
        "detail": detail,
    }


def build_card(
    signal: dict[str, Any],
    snapshot: dict[str, float | str],
    news_score: float,
) -> dict[str, Any]:
    current_price = float(signal["current_price"])
    pred_return_pct = float(signal["pred_return_pct"])
    ret_5d = float(snapshot["ret_5d"]) * 100.0
    ret_20d = float(snapshot["ret_20d"]) * 100.0
    display_upside = ((float(signal["target_price"]) / current_price) - 1.0) * 100.0 if current_price else 0.0
    pre_open_bias = clip(news_score * 3.2, -3.5, 3.5)
    return {
        "sig": signal["signal"],
        "sc": str(signal["signal"]).lower(),
        "conf": round(float(signal["confidence"]) / 100.0, 3),
        "px": round(current_price, 2),
        "chg": round(float(snapshot["ret_1d"]) * 100.0, 2),
        "l1h": round(pre_open_bias * 0.45, 2),
        "l4h": round(pre_open_bias, 2),
        "l1d": round(pred_return_pct, 2),
        "l5d": round(clip(pred_return_pct * 2.1 + ret_5d * 0.30, -8.0, 8.0), 2),
        "l10d": round(clip(pred_return_pct * 3.3 + ret_20d * 0.20, -12.0, 12.0), 2),
        "l20d": round(clip(pred_return_pct * 4.6 + ret_20d * 0.35, -18.0, 18.0), 2),
        "tgt": round(float(signal["target_price"]), 2),
        "up": round(display_upside, 1),
        "headline_summary": signal["summary"],
    }


def build_ticker_payload(ticker: str, now: datetime) -> dict[str, Any]:
    snapshot = build_technical_snapshot(ticker)
    model = load_model_component(ticker, snapshot)
    live_context = fetch_live_premarket_context(ticker=ticker, company_name=COMPANIES[ticker])
    news_score = build_news_score(live_context)

    model_weight = 0.65 if ticker == "PLTR" else 0.55
    technical_weight = 0.20 if ticker == "PLTR" else 0.25
    news_weight = 1.0 - model_weight - technical_weight

    model_score = float(model["score"])
    technical_score = float(snapshot["technical_score"])
    raw_score = (
        model_weight * model_score +
        technical_weight * technical_score +
        news_weight * news_score
    )

    components = [score for score in (model_score, technical_score, news_score) if abs(score) >= 0.12]
    if not components:
        agreement = 0.50
    else:
        dominant_sign = math.copysign(1.0, raw_score) if abs(raw_score) >= 0.05 else 0.0
        if dominant_sign == 0.0:
            agreement = 0.50
        else:
            agreement = sum(1 for score in components if math.copysign(1.0, score) == dominant_sign) / len(components)

    conflict_penalty = 0.10 if model_score * news_score < 0 and abs(news_score) >= 0.35 and abs(model_score) >= 0.25 else 0.0
    trust = clip(
        0.18 +
        0.42 * float(model["trust"]) +
        0.12 * agreement +
        0.10 * min(1.0, int(live_context.get("article_count", 0)) / 3.0) +
        0.08 * min(1.0, int(live_context.get("material_count", 0)) / 2.0) +
        0.10 * (1.0 - min(1.0, abs(technical_score - model_score))) -
        conflict_penalty,
        0.25,
        0.82,
    )
    if model["source"] != "pltr_deep_v1":
        # The cross-ticker DRL refresh is a live-only read, so cap trust lower
        # than the PLTR deep model, which has an explicit walk-forward audit.
        trust = min(trust, 0.58)

    reliability_scale = 0.35 + 0.65 * trust
    final_score = raw_score * reliability_scale
    probability_up = clip(0.5 + 0.28 * final_score, 0.18, 0.82)

    base_return = float(model["pred_return"])
    blended_return = clip(
        0.60 * base_return + 0.40 * (final_score * float(snapshot["move_scale"])),
        -0.05,
        0.05,
    )
    confidence = clip(0.55 * trust + 0.45 * abs(final_score), 0.10, 0.90)

    buy_threshold = 0.58 if trust >= 0.55 else 0.61
    sell_threshold = 0.42 if trust >= 0.55 else 0.39
    if probability_up >= buy_threshold:
        signal_label = "BUY"
    elif probability_up <= sell_threshold:
        signal_label = "SELL"
    else:
        signal_label = "HOLD"

    current_price = float(snapshot["current_price"])
    target_price = round(current_price * (1.0 + blended_return), 2)
    forecast_for_date = live_context.get("forecast_for_date") or snapshot["market_date"]

    drivers = [
        model_driver(ticker, model),
        {
            "title": "Pre-open news window",
            "direction": "bull" if news_score > 0.15 else "bear" if news_score < -0.15 else "neutral",
            "detail": live_context.get("summary", "No live news snapshot was available."),
        },
        technical_driver(snapshot),
        trust_driver(trust, agreement),
    ]

    summary = (
        f"{ticker} is set up as a {signal_label} into {forecast_for_date}, "
        f"with {probability_up * 100:.1f}% odds of finishing higher and a conservative {blended_return * 100:+.2f}% next-day move estimate."
    )
    signal = {
        "signal": signal_label,
        "probability_up": round(probability_up * 100.0, 1),
        "confidence": round(confidence * 100.0, 1),
        "pred_return_pct": round(blended_return * 100.0, 2),
        "current_price": round(current_price, 2),
        "target_price": target_price,
        "trust_score": round(trust * 100.0, 1),
        "forecast_for_date": forecast_for_date,
        "date": snapshot["market_date"],
        "summary": summary,
    }

    return {
        "ticker": ticker,
        "company_name": COMPANIES[ticker],
        "generated_at": now.isoformat(),
        "market_date": snapshot["market_date"],
        "forecast_for_date": forecast_for_date,
        "model_source": model["source"],
        "signal": signal,
        "summary": summary,
        "drivers": drivers,
        "news": live_context,
        "technical_snapshot": {
            "rsi14": round(float(snapshot["rsi14"]), 2),
            "ret_1d_pct": round(float(snapshot["ret_1d"]) * 100.0, 2),
            "ret_5d_pct": round(float(snapshot["ret_5d"]) * 100.0, 2),
            "ret_20d_pct": round(float(snapshot["ret_20d"]) * 100.0, 2),
            "pct_from_ma20": round(float(snapshot["pct_from_ma20"]) * 100.0, 2),
            "pct_from_ma50": round(float(snapshot["pct_from_ma50"]) * 100.0, 2),
            "volume_ratio": round(float(snapshot["volume_ratio"]), 2),
        },
        "component_scores": {
            "model": round(model_score, 4),
            "technical": round(technical_score, 4),
            "news": round(news_score, 4),
            "raw": round(raw_score, 4),
            "final": round(final_score, 4),
            "agreement": round(agreement, 4),
        },
        "card": build_card(signal, snapshot, news_score),
    }


def build_news_feed_items(ticker_payload: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    ticker = ticker_payload["ticker"]
    news = ticker_payload.get("news", {})
    articles = news.get("articles", [])[:4]
    if not articles:
        return []

    signal = ticker_payload["signal"]
    one_day = float(signal["pred_return_pct"])
    five_day = float(ticker_payload["card"]["l5d"])
    ten_day = float(ticker_payload["card"]["l10d"])
    items: list[dict[str, Any]] = []
    for idx, article in enumerate(articles):
        net_score = float(article.get("net_score", 0.0))
        direction = "bull" if net_score >= 0 else "bear"
        category = ((article.get("categories") or ["premarket"])[0]).replace("_", " ")
        published = article.get("published", "")
        items.append({
            "id": 9500 + (hash((ticker, idx, article.get("headline", ""))) % 1000),
            "ticker": ticker,
            "dir": direction,
            "impact": article.get("impact", "LOW"),
            "vader": clip(net_score / 3.0, -1.0, 1.0),
            "age": format_age(article.get("published_at_et"), now),
            "url": article.get("url", ""),
            "headline": article.get("headline", ""),
            "source": f"{article.get('source', 'Google News')}{f' · {published}' if published else ''}",
            "news_cat": category,
            "cal_impact": f"{one_day + net_score * 0.18:+.1f}%",
            "v5": {
                "mb": round(one_day / 1000.0, 6),
                "nm": round(net_score / 1000.0, 6),
                "jpn": 0.0,
                "ep": round(float(news.get("feature_values", {}).get("premkt_live_earnings_signal", 0.0)) / 1000.0, 6),
                "rl": 0.0,
                "tot": round((one_day + net_score * 0.18) / 100.0, 6),
            },
            "summary": article.get("rationale") or news.get("summary") or "Premarket headline added to the live forecast window.",
            "body": (
                f"{article.get('description') or article.get('headline', '')}\n\n"
                f"{article.get('rationale') or 'This headline was included in the live pre-open scoring window.'}\n\n"
                f"AXIOM live context: {news.get('summary', 'Mixed pre-open setup.')}"
            ),
            "vader_detail": article.get("rationale") or "Premarket sentiment signal.",
            "px_impact": f"{one_day + net_score * 0.18:+.1f}% next-day bias",
            "horizons": [
                {"l": "1D", "v": f"{one_day + net_score * 0.18:+.1f}%"},
                {"l": "1W", "v": f"{five_day + net_score * 0.25:+.1f}%"},
                {"l": "1M", "v": f"{ten_day + net_score * 0.35:+.1f}%"},
            ],
        })
    return items


def main() -> None:
    now = datetime.now().astimezone()
    payload = {
        "generated_at": now.isoformat(),
        "methodology": {
            "summary": (
                "Trust-calibrated tomorrow forecast built from refreshed local market data, "
                "live pre-open news, and the latest local model outputs."
            ),
            "note": (
                "PLTR uses the retrained deep model. AAPL, NVDA, and TSLA use the refreshed DRL live stack, "
                "then get shrunk toward neutral when model, technicals, and news disagree."
            ),
        },
        "tickers": {},
        "news_feed": [],
    }

    for ticker in COMPANIES:
        ticker_payload = build_ticker_payload(ticker, now)
        payload["tickers"][ticker] = ticker_payload
        payload["news_feed"].extend(build_news_feed_items(ticker_payload, now))

    payload["forecast_for_date"] = next(iter(payload["tickers"].values()))["forecast_for_date"]
    payload["market_date"] = max(
        ticker_payload["market_date"] for ticker_payload in payload["tickers"].values()
    )

    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_FILE, payload)
    write_json(DOCS_FILE, payload)
    write_json(BLOOMBERG_FILE, payload)

    print(json.dumps({
        "forecast_for_date": payload["forecast_for_date"],
        "market_date": payload["market_date"],
        "signals": {ticker: info["signal"] for ticker, info in payload["tickers"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
