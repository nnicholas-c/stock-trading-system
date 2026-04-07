#!/usr/bin/env python3
"""
news_scanner.py — Live news scanner for PLTR, AAPL, NVDA, TSLA.

Fetches latest headlines, scores sentiment, flags material events,
and outputs structured JSON for OpenClaw to include in alerts.

Usage:
    python scripts/news_scanner.py
    python scripts/news_scanner.py --ticker NVDA
    python scripts/news_scanner.py --alert-only  # only print if material news found
"""

import sys
import json
import argparse
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Positive/negative keywords for basic sentiment scoring
BULLISH_KEYWORDS = [
    "beat", "exceeds", "record", "upgrade", "strong", "raised", "accelerat",
    "partnership", "contract", "win", "awarded", "breakthrough", "ramp",
    "outperform", "buy", "bullish", "growth", "revenue up", "margin expand",
    "government contract", "AI", "autonomous", "demand", "backlog"
]
BEARISH_KEYWORDS = [
    "miss", "below", "cut", "downgrade", "weak", "decline", "lower",
    "sell", "bearish", "recall", "investigation", "lawsuit", "regulation",
    "tariff", "china", "export control", "miss", "guidance cut",
    "layoff", "loss", "margin compress", "competition"
]
MATERIAL_KEYWORDS = [
    "earnings", "acquisition", "merger", "SEC", "DOJ", "fine", "bankruptcy",
    "CEO", "CFO", "resign", "fired", "fraud", "restatement", "guidance",
    "dividend", "buyback", "split", "offering", "lockup"
]


def score_sentiment(headline: str, summary: str = "") -> dict:
    """Score a headline for bullish/bearish/material signals."""
    text = (headline + " " + summary).lower()

    bull_hits = [kw for kw in BULLISH_KEYWORDS if kw.lower() in text]
    bear_hits = [kw for kw in BEARISH_KEYWORDS if kw.lower() in text]
    material_hits = [kw for kw in MATERIAL_KEYWORDS if kw.lower() in text]

    bull_score = len(bull_hits)
    bear_score = len(bear_hits)

    if bull_score > bear_score:
        sentiment = "bullish"
        net = bull_score - bear_score
    elif bear_score > bull_score:
        sentiment = "bearish"
        net = -(bear_score - bull_score)
    else:
        sentiment = "neutral"
        net = 0

    is_material = len(material_hits) > 0

    return {
        "sentiment": sentiment,
        "net_score": net,
        "bull_signals": bull_hits[:3],
        "bear_signals": bear_hits[:3],
        "is_material": is_material,
        "material_triggers": material_hits
    }


def fetch_news_rss(ticker: str, company_name: str) -> list:
    """Fetch news via Google News RSS (no API key required)."""
    query = urllib.parse.quote(f"{ticker} {company_name} stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            content = resp.read().decode("utf-8", errors="ignore")

        # Parse RSS items manually (no lxml dependency)
        import re
        items = re.findall(r"<item>(.*?)</item>", content, re.DOTALL)
        articles = []
        for item in items[:8]:
            title_m = re.search(r"<title>(.*?)</title>", item)
            link_m = re.search(r"<link>(.*?)</link>", item)
            pub_m = re.search(r"<pubDate>(.*?)</pubDate>", item)
            desc_m = re.search(r"<description>(.*?)</description>", item)

            if title_m:
                title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip()
                # Remove source suffix like " - Reuters"
                title = re.sub(r"\s*-\s*[A-Z][a-zA-Z\s]+$", "", title).strip()

                pub_date = pub_m.group(1).strip() if pub_m else ""
                description = re.sub(r"<[^>]+>", "", desc_m.group(1)).strip() if desc_m else ""
                link = link_m.group(1).strip() if link_m else ""

                sentiment = score_sentiment(title, description)

                articles.append({
                    "ticker": ticker,
                    "headline": title,
                    "published": pub_date,
                    "url": link,
                    "sentiment": sentiment["sentiment"],
                    "net_score": sentiment["net_score"],
                    "is_material": sentiment["is_material"],
                    "bull_signals": sentiment["bull_signals"],
                    "bear_signals": sentiment["bear_signals"]
                })

        return articles

    except Exception as e:
        return [{"ticker": ticker, "error": str(e), "headline": "Failed to fetch news", "sentiment": "neutral"}]


TICKER_NAMES = {
    "PLTR": "Palantir",
    "AAPL": "Apple",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla"
}


def scan_all(tickers: list) -> dict:
    """Scan news for all tickers and return structured results."""
    results = {
        "generated_at": datetime.now().isoformat(),
        "tickers": {}
    }

    for ticker in tickers:
        name = TICKER_NAMES.get(ticker, ticker)
        articles = fetch_news_rss(ticker, name)

        # Compute aggregate sentiment
        sentiments = [a.get("sentiment", "neutral") for a in articles if "error" not in a]
        bull = sentiments.count("bullish")
        bear = sentiments.count("bearish")
        neutral = sentiments.count("neutral")

        if bull > bear + neutral:
            overall = "bullish"
        elif bear > bull + neutral:
            overall = "bearish"
        elif bear > bull:
            overall = "cautious"
        else:
            overall = "neutral"

        material = [a for a in articles if a.get("is_material")]
        top_headlines = sorted(articles, key=lambda x: abs(x.get("net_score", 0)), reverse=True)[:5]

        results["tickers"][ticker] = {
            "overall_sentiment": overall,
            "bull_articles": bull,
            "bear_articles": bear,
            "neutral_articles": neutral,
            "material_events": len(material),
            "top_headlines": top_headlines,
            "material_alerts": material[:3]
        }

    # Save to file
    out_path = ROOT / "trading_system" / "signals" / "news_scan.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def format_for_openclaw(results: dict) -> str:
    """Format news scan results as a clean alert message for OpenClaw."""
    lines = ["📰 MARKET NEWS SCAN", f"_{datetime.now().strftime('%b %d %H:%M')} PDT_", ""]

    sentiment_emoji = {"bullish": "🟢", "bearish": "🔴", "cautious": "🟡", "neutral": "⚪"}

    for ticker, data in results.get("tickers", {}).items():
        emoji = sentiment_emoji.get(data["overall_sentiment"], "⚪")
        lines.append(f"{emoji} **{ticker}** — News sentiment: {data['overall_sentiment'].upper()}")

        if data["material_events"] > 0:
            lines.append(f"   ⚠️ {data['material_events']} material event(s) detected!")

        for art in data["top_headlines"][:3]:
            s_emoji = "📈" if art["sentiment"] == "bullish" else "📉" if art["sentiment"] == "bearish" else "➡️"
            lines.append(f"   {s_emoji} {art['headline']}")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", nargs="+", default=["PLTR", "AAPL", "NVDA", "TSLA"])
    parser.add_argument("--format", choices=["json", "text"], default="json")
    parser.add_argument("--alert-only", action="store_true")
    args = parser.parse_args()

    results = scan_all(args.ticker)

    if args.alert_only:
        # Only output if there's material news or strong sentiment
        has_alert = any(
            d["material_events"] > 0 or d["overall_sentiment"] in ("bullish", "bearish")
            for d in results["tickers"].values()
        )
        if not has_alert:
            sys.exit(0)

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        print(format_for_openclaw(results))


if __name__ == "__main__":
    main()
