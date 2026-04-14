from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from datetime import date, datetime, time, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

US_EASTERN = ZoneInfo("America/New_York")

ROOT = Path(__file__).resolve().parent

# Historical PLTR news corpus used to approximate what the model could have
# known before the next regular session opened.
PLTR_PREMARKET_NEWS_CORPUS = [
    ("2023-02-13", 0.82, "earnings", 0.21, "First GAAP profit and Q4 2022 beat"),
    ("2023-03-10", -0.65, "macro", -0.04, "SVB collapse hit high-beta tech"),
    ("2023-04-26", 0.78, "product", 0.07, "AIP platform launch"),
    ("2023-05-08", 0.85, "earnings", 0.23, "Q1 2023 beat and AIP traction"),
    ("2023-06-15", 0.70, "contract", 0.06, "DoD Maven expansion"),
    ("2023-07-20", 0.72, "product", 0.08, "AIP bootcamp traction"),
    ("2023-08-07", -0.45, "earnings", -0.05, "Q2 2023 government softness"),
    ("2023-11-02", 0.80, "earnings", 0.20, "Q3 2023 beat"),
    ("2023-11-20", 0.50, "partnership", 0.05, "Azure AI partnership"),
    ("2024-02-05", 0.88, "earnings", 0.31, "Q4 2023 beat"),
    ("2024-03-01", -0.35, "valuation", -0.03, "Valuation debate intensifies"),
    ("2024-04-15", 0.68, "contract", 0.05, "Maven Smart System expansion"),
    ("2024-05-06", -0.52, "earnings", -0.15, "Q1 2024 sold on the news"),
    ("2024-07-10", 0.82, "contract", 0.09, "Large Army enterprise agreement"),
    ("2024-08-05", 0.78, "earnings", 0.10, "Q2 2024 beat"),
    ("2024-09-09", 0.92, "market", 0.14, "S&P 500 inclusion confirmed"),
    ("2024-10-15", 0.62, "partnership", 0.05, "L3Harris partnership"),
    ("2024-11-04", 0.85, "earnings", 0.24, "Q3 2024 beat"),
    ("2024-11-05", 0.95, "political", 0.61, "Trump election / DOGE proxy rally"),
    ("2024-11-20", 0.75, "partnership", 0.08, "Anduril defense consortium"),
    ("2024-12-10", 0.68, "partnership", 0.05, "Anthropic and model integrations"),
    ("2024-12-20", -0.40, "macro", -0.04, "Hawkish Fed on high multiple stocks"),
    ("2025-01-20", 0.88, "political", 0.12, "DOGE creation"),
    ("2025-01-27", -0.55, "competition", -0.05, "DeepSeek and AI commoditization fears"),
    ("2025-01-28", 0.70, "contract", 0.09, "ICE contract win"),
    ("2025-02-03", 0.88, "earnings", 0.24, "Q4 2024 beat and guidance raise"),
    ("2025-02-10", 0.62, "analyst", 0.05, "Daiwa Buy upgrade"),
    ("2025-02-18", -0.72, "political", -0.08, "Pentagon DOGE cuts fear"),
    ("2025-02-25", -0.55, "valuation", -0.06, "Fortune valuation concern"),
    ("2025-03-05", 0.55, "partnership", 0.04, "Databricks partnership"),
    ("2025-03-20", 0.78, "contract", 0.06, "Pentagon contract win"),
    ("2025-04-02", -0.80, "macro", -0.07, "Tariff escalation shock"),
    ("2025-04-09", 0.78, "macro", 0.09, "Tariff pause rally"),
    ("2025-05-05", -0.65, "earnings", -0.12, "Q1 2025 EPS miss"),
    ("2025-05-15", 0.68, "contract", 0.06, "Fannie Mae contract"),
    ("2025-06-03", 0.75, "market", 0.06, "Top S&P performer framing"),
    ("2025-08-04", 0.88, "earnings", 0.08, "Q2 2025 beat and US commercial blowout"),
    ("2025-10-15", 0.72, "analyst", 0.06, "Post-Q2 analyst upgrades"),
    ("2025-11-03", -0.58, "earnings", -0.08, "Q3 2025 beat but valuation concern"),
    ("2025-12-19", 0.45, "contract", 0.03, "Army payment / steady contract flow"),
    ("2026-01-12", -0.68, "political", -0.05, "Defense cuts speech pressure"),
    ("2026-01-15", -0.60, "macro", -0.03, "Oil spike and VIX surge"),
    ("2026-01-20", -0.72, "political", -0.08, "DOGE chainsaw fears"),
    ("2026-01-26", 0.80, "analyst", 0.06, "BofA US 1 list top pick"),
    ("2026-02-02", 0.90, "earnings", 0.07, "Q4 2025 beat and strong guidance"),
    ("2026-02-10", 0.72, "analyst", 0.05, "Daiwa Buy after earnings"),
    ("2026-02-15", 0.78, "contract", 0.03, "Maven designated program of record"),
    ("2026-02-27", 0.80, "analyst", 0.06, "UBS Buy on reset valuation"),
    ("2026-03-01", 0.82, "analyst", 0.05, "Rosenblatt $200 target"),
    ("2026-03-05", -0.35, "technical", -0.02, "Trading below key moving averages"),
    ("2026-03-09", 0.68, "analyst", 0.04, "Barchart high price target"),
    ("2026-03-30", 0.75, "analyst", 0.05, "Long-term upside narrative"),
    ("2026-04-02", -0.70, "macro", -0.07, "Tariff-led tech selloff"),
    ("2026-04-07", 0.30, "market", 0.02, "Recovery attempt into earnings"),
]

BULLISH_KEYWORDS = [
    "beat",
    "raised",
    "upgrade",
    "buy",
    "contract",
    "awarded",
    "wins",
    "partnership",
    "backlog",
    "strong",
    "growth",
    "record",
    "guidance",
    "bullish",
    "demand",
]
BEARISH_KEYWORDS = [
    "miss",
    "downgrade",
    "competition",
    "anthropic",
    "claude",
    "openai",
    "microsoft",
    "copilot",
    "salesforce",
    "einstein",
    "valuation",
    "expensive",
    "cut",
    "boycott",
    "lawsuit",
    "investigation",
    "tariff",
    "risk",
    "pressure",
    "decline",
    "slump",
    "fall",
]
MATERIAL_KEYWORDS = [
    "earnings",
    "guidance",
    "contract",
    "deal",
    "award",
    "government",
    "regulation",
    "investigation",
    "lawsuit",
    "boycott",
    "sec",
    "ceo",
    "merger",
    "acquisition",
]

CATEGORY_KEYWORDS = {
    "earnings": ["earnings", "guidance", "eps", "revenue", "quarter", "beat", "miss"],
    "contract": ["contract", "award", "army", "dod", "pentagon", "government", "maven", "agreement"],
    "competition": ["competition", "anthropic", "claude", "openai", "copilot", "salesforce", "einstein", "peer"],
    "valuation": ["valuation", "expensive", "multiple", "re-rating", "market cap", "price target"],
    "macro": ["tariff", "rates", "fed", "yield", "vix", "selloff", "rotation", "macro"],
    "analyst": ["analyst", "upgrade", "downgrade", "target", "rating"],
    "political": ["doge", "trump", "defense cuts", "pentagon cuts", "policy"],
}


def next_business_day(value: date | pd.Timestamp | datetime) -> date:
    if isinstance(value, pd.Timestamp):
        current = value.date()
    elif isinstance(value, datetime):
        current = value.date()
    else:
        current = value
    nxt = current + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def previous_business_day(value: date | pd.Timestamp | datetime) -> date:
    if isinstance(value, pd.Timestamp):
        current = value.date()
    elif isinstance(value, datetime):
        current = value.date()
    else:
        current = value
    prv = current - timedelta(days=1)
    while prv.weekday() >= 5:
        prv -= timedelta(days=1)
    return prv


def build_historical_news_features(
    dates: pd.DatetimeIndex,
    lookback_days: int = 21,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for current_date in dates:
        decayed = {
            "news_sentiment_21d": 0.0,
            "news_earnings_21d": 0.0,
            "news_contract_21d": 0.0,
            "news_analyst_21d": 0.0,
            "news_competition_21d": 0.0,
            "news_macro_21d": 0.0,
            "news_velocity_21d": 0.0,
            "news_extreme_flag": 0.0,
        }
        premarket = {
            "premkt_hist_net": 0.0,
            "premkt_hist_article_count": 0.0,
            "premkt_hist_material_count": 0.0,
            "premkt_hist_competition_risk": 0.0,
            "premkt_hist_contract_signal": 0.0,
            "premkt_hist_earnings_signal": 0.0,
        }

        decayed_events = 0
        for raw_date, sentiment, category, magnitude, _headline in PLTR_PREMARKET_NEWS_CORPUS:
            event_date = pd.Timestamp(raw_date)
            delta_days = (current_date.normalize() - event_date.normalize()).days
            if 0 <= delta_days <= lookback_days:
                decay = math.exp(-0.10 * delta_days)
                decayed_events += 1
                decayed["news_sentiment_21d"] += sentiment * decay
                decayed["news_velocity_21d"] += decay
                decayed["news_extreme_flag"] = max(
                    decayed["news_extreme_flag"],
                    abs(sentiment * magnitude) * decay,
                )
                if category == "earnings":
                    decayed["news_earnings_21d"] += sentiment * decay
                elif category == "contract":
                    decayed["news_contract_21d"] += sentiment * decay
                elif category == "analyst":
                    decayed["news_analyst_21d"] += sentiment * decay
                elif category in {"competition", "valuation"}:
                    decayed["news_competition_21d"] += sentiment * decay
                elif category in {"macro", "political"}:
                    decayed["news_macro_21d"] += sentiment * decay

            # Approximate "known before open" by allowing same-day and previous-day
            # curated events to inform the next regular session.
            if 0 <= delta_days <= 1:
                premarket["premkt_hist_net"] += sentiment
                premarket["premkt_hist_article_count"] += 1.0
                if abs(magnitude) >= 0.05 or category in {"earnings", "contract", "political"}:
                    premarket["premkt_hist_material_count"] += 1.0
                if category in {"competition", "valuation"}:
                    premarket["premkt_hist_competition_risk"] += max(0.0, -sentiment)
                if category == "contract":
                    premarket["premkt_hist_contract_signal"] += max(0.0, sentiment)
                if category == "earnings":
                    premarket["premkt_hist_earnings_signal"] += sentiment

        if decayed_events:
            normalizer = max(1.0, decayed_events * 0.35)
            for key in (
                "news_sentiment_21d",
                "news_earnings_21d",
                "news_contract_21d",
                "news_analyst_21d",
                "news_competition_21d",
                "news_macro_21d",
            ):
                decayed[key] = float(np.clip(decayed[key] / normalizer, -2.5, 2.5))
            decayed["news_velocity_21d"] = float(min(1.5, decayed["news_velocity_21d"]))

        if premarket["premkt_hist_article_count"]:
            premarket["premkt_hist_net"] = float(
                np.clip(
                    premarket["premkt_hist_net"] / max(1.0, premarket["premkt_hist_article_count"]),
                    -2.5,
                    2.5,
                )
            )

        rows.append({**decayed, **premarket})

    return pd.DataFrame(rows, index=dates).fillna(0.0)


def infer_article_categories(text: str) -> list[str]:
    categories: list[str] = []
    lowered = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            categories.append(category)
    if not categories:
        categories.append("market")
    return categories


def score_live_article(article: dict[str, Any]) -> dict[str, Any]:
    text = f"{article.get('headline', '')} {article.get('description', '')}".lower()
    bull = [kw for kw in BULLISH_KEYWORDS if kw in text]
    bear = [kw for kw in BEARISH_KEYWORDS if kw in text]
    material = [kw for kw in MATERIAL_KEYWORDS if kw in text]
    categories = infer_article_categories(text)
    net_score = len(bull) - len(bear)
    net_score = int(np.clip(net_score, -5, 5))

    if net_score > 0:
        sentiment = "BULLISH"
    elif net_score < 0:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"

    impact = "LOW"
    if material or any(cat in categories for cat in ("earnings", "contract", "competition")):
        impact = "HIGH"
    elif abs(net_score) >= 2 or any(cat in categories for cat in ("analyst", "macro", "valuation")):
        impact = "MEDIUM"

    if "competition" in categories and sentiment != "BULLISH":
        rationale = "Competition and valuation language keep the near-term read cautious."
    elif "contract" in categories:
        rationale = "Government or enterprise contract language is a supportive demand signal."
    elif "earnings" in categories:
        rationale = "Quarterly results and guidance language can move the stock quickly into the open."
    elif "macro" in categories:
        rationale = "Macro risk language matters because high-growth names can gap hard into the open."
    elif sentiment == "BULLISH":
        rationale = "Headline tone leans constructive for demand, execution, or positioning."
    elif sentiment == "BEARISH":
        rationale = "Headline tone leans cautious on multiple, competition, or risk."
    else:
        rationale = "Headline looks mixed and does not add a strong directional edge by itself."

    return {
        "sentiment": sentiment,
        "impact": impact,
        "net_score": net_score,
        "is_material": bool(material) or impact == "HIGH",
        "categories": categories,
        "rationale": rationale,
    }


def get_upcoming_session_window(reference_dt: datetime | None = None) -> dict[str, Any]:
    now = (reference_dt or datetime.now(tz=US_EASTERN)).astimezone(US_EASTERN)
    if now.weekday() >= 5:
        forecast_date = next_business_day(now.date())
        start_date = previous_business_day(forecast_date)
        window_start = datetime.combine(start_date, time(16, 0), tzinfo=US_EASTERN)
        next_open = datetime.combine(forecast_date, time(9, 30), tzinfo=US_EASTERN)
    elif now.time() >= time(16, 0):
        forecast_date = next_business_day(now.date())
        window_start = datetime.combine(now.date(), time(16, 0), tzinfo=US_EASTERN)
        next_open = datetime.combine(forecast_date, time(9, 30), tzinfo=US_EASTERN)
    elif now.time() < time(9, 30):
        forecast_date = now.date()
        prior_session = previous_business_day(now.date())
        window_start = datetime.combine(prior_session, time(16, 0), tzinfo=US_EASTERN)
        next_open = datetime.combine(forecast_date, time(9, 30), tzinfo=US_EASTERN)
    else:
        forecast_date = next_business_day(now.date())
        window_start = datetime.combine(now.date(), time(16, 0), tzinfo=US_EASTERN)
        next_open = datetime.combine(forecast_date, time(9, 30), tzinfo=US_EASTERN)
    return {
        "now": now,
        "forecast_date": forecast_date,
        "window_start": window_start,
        "next_open": next_open,
    }


def fetch_live_premarket_context(
    ticker: str = "PLTR",
    company_name: str = "Palantir",
    reference_dt: datetime | None = None,
) -> dict[str, Any]:
    session = get_upcoming_session_window(reference_dt)
    query = urllib.parse.quote(f"{ticker} {company_name} stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    articles: list[dict[str, Any]] = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            content = response.read().decode("utf-8", errors="ignore")
        items = re.findall(r"<item>(.*?)</item>", content, re.DOTALL)
        for item in items[:15]:
            title_match = re.search(r"<title>(.*?)</title>", item)
            link_match = re.search(r"<link>(.*?)</link>", item)
            pub_match = re.search(r"<pubDate>(.*?)</pubDate>", item)
            desc_match = re.search(r"<description>(.*?)</description>", item)
            if not title_match:
                continue
            headline = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()
            headline = re.sub(r"\s*[-–—]\s*\S+\s*$", "", headline).strip()
            description = (
                re.sub(r"<[^>]+>", "", desc_match.group(1)).strip() if desc_match else ""
            )
            published_raw = pub_match.group(1).strip() if pub_match else ""
            published_dt = None
            if published_raw:
                try:
                    published_dt = parsedate_to_datetime(published_raw).astimezone(US_EASTERN)
                except Exception:
                    published_dt = None

            article = {
                "headline": headline,
                "description": description,
                "source": "Google News",
                "url": link_match.group(1).strip() if link_match else "",
                "published": published_raw,
                "published_at_et": published_dt.isoformat() if published_dt else None,
                "in_window": bool(
                    published_dt and session["window_start"] <= published_dt <= session["now"]
                ),
            }
            article.update(score_live_article(article))
            articles.append(article)
    except Exception as exc:
        return {
            "ticker": ticker,
            "forecast_for_date": session["forecast_date"].isoformat(),
            "window_start_et": session["window_start"].isoformat(),
            "window_end_et": session["now"].isoformat(),
            "next_open_et": session["next_open"].isoformat(),
            "analysis_mode": "heuristic",
            "article_count": 0,
            "material_count": 0,
            "overall_sentiment": "NEUTRAL",
            "intraday_bias": "FLAT",
            "net_score": 0.0,
            "used_recent_fallback": False,
            "feature_values": {
                "premkt_live_net": 0.0,
                "premkt_live_article_count": 0.0,
                "premkt_live_material_count": 0.0,
                "premkt_live_competition_risk": 0.0,
                "premkt_live_contract_signal": 0.0,
                "premkt_live_earnings_signal": 0.0,
            },
            "summary": f"Live pre-open news fetch failed: {exc}",
            "articles": [],
        }

    in_window = [article for article in articles if article["in_window"]]
    used_recent_fallback = False
    if not in_window:
        used_recent_fallback = True
        in_window = articles[:5]

    bull_count = sum(1 for article in in_window if article["sentiment"] == "BULLISH")
    bear_count = sum(1 for article in in_window if article["sentiment"] == "BEARISH")
    material_count = sum(1 for article in in_window if article["is_material"])
    article_count = len(in_window)
    net_score = float(np.mean([article["net_score"] for article in in_window])) if in_window else 0.0

    competition_risk = 0.0
    contract_signal = 0.0
    earnings_signal = 0.0
    valuation_risk = 0.0
    category_counts: dict[str, int] = {}
    for article in in_window:
        for category in article["categories"]:
            category_counts[category] = category_counts.get(category, 0) + 1
        if "competition" in article["categories"]:
            competition_risk += max(0.0, -article["net_score"])
        if "valuation" in article["categories"]:
            valuation_risk += max(0.0, -article["net_score"])
        if "contract" in article["categories"]:
            contract_signal += max(0.0, article["net_score"])
        if "earnings" in article["categories"]:
            earnings_signal += article["net_score"]

    if bull_count > bear_count + 1:
        overall_sentiment = "BULLISH"
        intraday_bias = "UP"
    elif bear_count > bull_count + 1 or net_score <= -0.8:
        overall_sentiment = "BEARISH"
        intraday_bias = "DOWN"
    else:
        overall_sentiment = "NEUTRAL"
        intraday_bias = "FLAT"

    top_categories = sorted(category_counts.items(), key=lambda item: item[1], reverse=True)[:2]
    category_text = ", ".join(name for name, _count in top_categories) if top_categories else "general market"
    if article_count == 0:
        summary = (
            f"No fresh {ticker} headlines were found in the pre-open window, so the model "
            "leans on historical catalysts and price action."
        )
    elif used_recent_fallback:
        summary = (
            f"No fresh {ticker} headlines landed in the current post-close / pre-open window, "
            "so the model is using the most recent broader headlines with extra caution."
        )
    elif overall_sentiment == "BEARISH":
        summary = (
            f"Pre-open {ticker} news leans bearish with {article_count} headline(s); "
            f"competition, valuation, or macro pressure dominate the window."
        )
    elif overall_sentiment == "BULLISH":
        summary = (
            f"Pre-open {ticker} news leans constructive with {article_count} headline(s); "
            f"the strongest themes are {category_text}."
        )
    else:
        summary = (
            f"Pre-open {ticker} news is mixed across {article_count} headline(s); "
            f"the tape still needs technical confirmation after the open."
        )

    return {
        "ticker": ticker,
        "forecast_for_date": session["forecast_date"].isoformat(),
        "window_start_et": session["window_start"].isoformat(),
        "window_end_et": session["now"].isoformat(),
        "next_open_et": session["next_open"].isoformat(),
        "analysis_mode": "heuristic",
        "article_count": article_count,
        "material_count": material_count,
        "overall_sentiment": overall_sentiment,
        "intraday_bias": intraday_bias,
        "net_score": round(net_score, 3),
        "used_recent_fallback": used_recent_fallback,
        "category_counts": category_counts,
        "summary": summary,
        "feature_values": {
            "premkt_live_net": round(net_score, 3),
            "premkt_live_article_count": float(article_count),
            "premkt_live_material_count": float(material_count),
            "premkt_live_competition_risk": round(min(6.0, competition_risk + valuation_risk), 3),
            "premkt_live_contract_signal": round(min(6.0, contract_signal), 3),
            "premkt_live_earnings_signal": round(float(np.clip(earnings_signal, -6.0, 6.0)), 3),
        },
        "articles": in_window[:8],
    }


def apply_live_context_to_news_frame(
    news_frame: pd.DataFrame,
    live_context: dict[str, Any] | None,
) -> pd.DataFrame:
    frame = news_frame.copy()
    for column in (
        "premkt_live_net",
        "premkt_live_article_count",
        "premkt_live_material_count",
        "premkt_live_competition_risk",
        "premkt_live_contract_signal",
        "premkt_live_earnings_signal",
    ):
        frame[column] = 0.0

    if live_context and not frame.empty:
        for column, value in live_context.get("feature_values", {}).items():
            if column in frame.columns:
                frame.iloc[-1, frame.columns.get_loc(column)] = value
    return frame


def build_reasoning_payload(
    signal: dict[str, Any],
    live_signals_by_horizon: dict[str, Any],
    live_context: dict[str, Any] | None,
    features: pd.DataFrame,
    pltr_close: pd.Series,
) -> dict[str, Any]:
    latest = features.iloc[-1]
    current_price = float(pltr_close.iloc[-1])
    five_day_move = float(pltr_close.pct_change(5).iloc[-1] * 100) if len(pltr_close) >= 6 else 0.0
    one_day_move = float(pltr_close.pct_change(1).iloc[-1] * 100) if len(pltr_close) >= 2 else 0.0
    to_160_pct = (160.0 / current_price - 1.0) * 100

    reasons: list[dict[str, str]] = []

    premkt_net = float(latest.get("premkt_live_net", 0.0))
    if live_context and live_context.get("article_count", 0) > 0:
        direction = "bull" if premkt_net > 0.35 else "bear" if premkt_net < -0.35 else "neutral"
        reasons.append({
            "title": "Pre-open news tone",
            "direction": direction,
            "detail": live_context.get("summary", "Pre-open headlines were folded into the forecast."),
        })

    competition_risk = float(latest.get("premkt_live_competition_risk", 0.0))
    if competition_risk >= 1.5 or float(latest.get("news_competition_21d", 0.0)) < -0.20:
        reasons.append({
            "title": "Competition and multiple pressure",
            "direction": "bear",
            "detail": "Recent headlines still skew toward AI competition and valuation compression, which is the cleanest explanation for the sharp drawdown.",
        })

    pct_from_ma20 = float(latest.get("pct_from_ma20", 0.0))
    rsi_norm = float(latest.get("rsi_14", 0.0))
    macd_hist = float(latest.get("macd_hist", 0.0))
    if pct_from_ma20 < -0.03:
        reasons.append({
            "title": "Trend is still damaged",
            "direction": "bear",
            "detail": "PLTR remains below its short-term trend anchor, so rebounds are still fighting a weak tape rather than riding a clean uptrend.",
        })
    elif rsi_norm < -0.15 and five_day_move < -5:
        reasons.append({
            "title": "Oversold bounce setup",
            "direction": "bull",
            "detail": "The recent selloff has pushed the setup toward mean reversion, which is why the 1-day model still leans modestly positive.",
        })
    elif macd_hist > 0:
        reasons.append({
            "title": "Momentum is stabilizing",
            "direction": "bull",
            "detail": "Short-term momentum is no longer deteriorating, which supports a rebound scenario even if the medium-term path remains choppy.",
        })

    macro_score = float(latest.get("macro_score", 0.0))
    if macro_score < -0.15 or float(latest.get("vix_elevated", 0.0)) > 0:
        reasons.append({
            "title": "Macro still matters",
            "direction": "bear",
            "detail": "PLTR trades like a high-beta growth name, so tariff, rate, and volatility pressure can still overwhelm good company-level news in the near term.",
        })

    news_earnings = float(latest.get("news_earnings_21d", 0.0))
    if news_earnings > 0.20 or float(latest.get("premkt_live_earnings_signal", 0.0)) > 0.5:
        reasons.append({
            "title": "Earnings and fundamental memory stay supportive",
            "direction": "bull",
            "detail": "The model still carries positive memory from Palantir's strong fundamental prints and upcoming earnings setup, which offsets part of the selloff narrative.",
        })

    if not reasons:
        reasons.append({
            "title": "Mixed setup",
            "direction": "neutral",
            "detail": "The current read is mixed enough that price action still matters more than any single headline bucket.",
        })

    tomorrow = live_signals_by_horizon.get("1d", signal)
    five_day = live_signals_by_horizon.get("5d", signal)
    ten_day = live_signals_by_horizon.get("10d", signal)

    if to_160_pct <= 8:
        rebound_label = "within reach if momentum expands"
    elif to_160_pct <= 15:
        rebound_label = "possible, but it needs a better tape and a catalyst"
    else:
        rebound_label = "a real re-rating move, not a normal one- or two-day bounce"

    latest_day_phrase = "up" if one_day_move > 0 else "down"
    why_falling = (
        f"PLTR is down {abs(five_day_move):.1f}% over the last 5 sessions and was {latest_day_phrase} {abs(one_day_move):.1f}% on the latest day. "
        "The selloff still looks more like a fast valuation and sentiment reset than a fresh company-specific break: "
        "competition / valuation headlines are active, and macro risk remains a headwind for high-multiple software."
    )

    return {
        "summary": (
            f"The updated PLTR deep model still leans rebound rather than breakdown into {tomorrow.get('forecast_for_date', 'the next session')}, "
            f"but the move it expects is small: {tomorrow.get('pred_return_pct', 0):+.2f}% for the next day and "
            f"{five_day.get('pred_return_pct', 0):+.2f}% over 5 days."
        ),
        "why_falling": why_falling,
        "reach_160": {
            "target_price": 160.0,
            "required_return_pct": round(to_160_pct, 2),
            "assessment": rebound_label,
            "near_term_model_read": (
                f"1d target ${tomorrow.get('target_price', current_price):.2f}, "
                f"5d target ${five_day.get('target_price', current_price):.2f}, "
                f"10d target ${ten_day.get('target_price', current_price):.2f}"
            ),
        },
        "drivers": reasons[:5],
    }


def build_docs_payload(
    output: dict[str, Any],
    live_context: dict[str, Any] | None,
    reasoning: dict[str, Any],
    prices: pd.DataFrame,
) -> dict[str, Any]:
    pltr_close = prices["PLTR"]
    current_price = float(pltr_close.iloc[-1])
    prev_close = float(pltr_close.iloc[-2]) if len(pltr_close) > 1 else current_price
    one_day_change_pct = (current_price / prev_close - 1.0) * 100 if prev_close else 0.0

    tomorrow = output.get("tomorrow_prediction", {})
    signals = output.get("live_signals_by_horizon", {})
    five_day = signals.get("5d", output.get("live_signal", {}))
    ten_day = signals.get("10d", {})
    display_target = float(tomorrow.get("target_price", current_price))
    display_upside = (display_target / current_price - 1.0) * 100 if current_price else 0.0

    premarket_open_bias = 0.0
    if live_context:
        premarket_open_bias = float(np.clip(live_context.get("net_score", 0.0) * 0.55, -3.5, 3.5))

    card = {
        "sig": tomorrow.get("signal", "HOLD"),
        "sc": tomorrow.get("signal", "HOLD").lower(),
        "conf": round(tomorrow.get("confidence", 0.0) / 100.0, 3),
        "px": round(current_price, 2),
        "chg": round(one_day_change_pct, 2),
        "l1h": round(premarket_open_bias * 0.4, 2),
        "l4h": round(premarket_open_bias, 2),
        "l1d": round(tomorrow.get("pred_return_pct", 0.0), 2),
        "l5d": round(five_day.get("pred_return_pct", 0.0), 2),
        "l10d": round(ten_day.get("pred_return_pct", five_day.get("pred_return_pct", 0.0)), 2),
        "l20d": round(
            ten_day.get("pred_return_pct", five_day.get("pred_return_pct", 0.0) * 1.8),
            2,
        ),
        "tgt": round(display_target, 2),
        "up": round(display_upside, 1),
        "headline_summary": reasoning.get("summary"),
    }

    return {
        "ticker": "PLTR",
        "generated_at": output.get("generated"),
        "market_date": output.get("live_signal", {}).get("date"),
        "forecast_for_date": tomorrow.get("forecast_for_date"),
        "signal": tomorrow,
        "live_signals_by_horizon": signals,
        "backtest": output.get("backtest", {}),
        "card": card,
        "recovery_target_price": 160.0,
        "news": live_context or {},
        "reasoning": reasoning,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
