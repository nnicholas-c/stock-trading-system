"""
NewsService — fetches and caches live news for all tickers.
Uses Google News RSS (no API key needed).
Scores sentiment, detects material events, estimates intraday impact.
"""

import asyncio
import aiohttp
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from app.core.config import settings

BULLISH_KW = ["beat","record","upgraded","strong","awarded","contract","partnership",
              "breakthrough","acceleration","AI","demand","backlog","growth","raised"]
BEARISH_KW = ["miss","cut","downgrade","weak","decline","recall","investigation",
              "lawsuit","tariff","export control","loss","layoff","competition","resign"]
MATERIAL_KW= ["earnings","acquisition","merger","SEC","FDA","guidance","CEO",
               "dividend","buyback","split","offering","fraud","restatement"]

COMPANY_NAMES = {
    "PLTR": "Palantir",
    "AAPL": "Apple",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
}

class NewsService:
    _cache: dict = {}  # ticker → {data, expires_at}

    @classmethod
    def _score(cls, text: str) -> dict:
        t = text.lower()
        bull = [k for k in BULLISH_KW if k.lower() in t]
        bear = [k for k in BEARISH_KW if k.lower() in t]
        mat  = [k for k in MATERIAL_KW if k.lower() in t]
        net  = len(bull) - len(bear)
        return {
            "sentiment":  "BULLISH" if net>0 else "BEARISH" if net<0 else "NEUTRAL",
            "net_score":  net,
            "is_material": bool(mat),
            "bull_signals": bull[:3],
            "bear_signals": bear[:3],
            "impact": "HIGH" if mat or abs(net)>=3 else "MEDIUM" if abs(net)>=2 else "LOW",
        }

    @classmethod
    async def fetch(cls, ticker: str) -> dict:
        # Return cache if still fresh
        cached = cls._cache.get(ticker)
        if cached and datetime.now() < cached["expires_at"]:
            return {**cached["data"], "cached": True}

        company = COMPANY_NAMES.get(ticker, ticker)
        query   = f"{ticker} {company} stock"
        url     = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

        articles = []
        try:
            timeout = aiohttp.ClientTimeout(total=8)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                    content = await resp.text()

            items = re.findall(r"<item>(.*?)</item>", content, re.DOTALL)
            for item in items[:10]:
                title_m = re.search(r"<title>(.*?)</title>", item)
                link_m  = re.search(r"<link>(.*?)</link>", item)
                pub_m   = re.search(r"<pubDate>(.*?)</pubDate>", item)
                desc_m  = re.search(r"<description>(.*?)</description>", item)

                if not title_m:
                    continue

                title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip()
                title = re.sub(r"\s*[-–—]\s*\S+\s*$", "", title).strip()
                desc  = re.sub(r"<[^>]+>", "", desc_m.group(1)).strip() if desc_m else ""
                link  = link_m.group(1).strip() if link_m else ""
                pub   = pub_m.group(1).strip() if pub_m else ""

                scored = cls._score(title + " " + desc)
                articles.append({
                    "ticker":    ticker,
                    "headline":  title,
                    "sentiment": scored["sentiment"],
                    "impact":    scored["impact"],
                    "source":    "Google News",
                    "url":       link,
                    "published": pub,
                    "net_score": scored["net_score"],
                    "is_material": scored["is_material"],
                })

        except Exception as e:
            articles = [{"ticker": ticker, "headline": f"News fetch error: {e}",
                         "sentiment": "NEUTRAL", "impact": "LOW",
                         "source": "—", "url": "", "published": "", "net_score": 0,
                         "is_material": False}]

        # Aggregate sentiment
        sentiments = [a["sentiment"] for a in articles]
        bull_n = sentiments.count("BULLISH")
        bear_n = sentiments.count("BEARISH")
        overall = "BULLISH" if bull_n > bear_n+1 else "BEARISH" if bear_n > bull_n+1 else "NEUTRAL"
        material_count = sum(1 for a in articles if a["is_material"])

        # Intraday impact estimate
        if overall == "BULLISH" and bull_n >= 3:
            intraday = "UP"
        elif overall == "BEARISH" and bear_n >= 3:
            intraday = "DOWN"
        else:
            intraday = "FLAT"

        data = {
            "ticker":            ticker,
            "generated_at":      datetime.now().isoformat(),
            "overall_sentiment": overall,
            "articles":          sorted(articles, key=lambda x: -abs(x["net_score"]))[:8],
            "material_events":   material_count,
            "intraday_impact":   intraday,
            "cached":            False,
        }

        cls._cache[ticker] = {
            "data":       data,
            "expires_at": datetime.now() + timedelta(seconds=settings.news_cache_ttl)
        }
        return data

    @classmethod
    async def fetch_all(cls) -> dict:
        tasks = [cls.fetch(t) for t in settings.tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {t: r for t, r in zip(settings.tickers, results) if not isinstance(r, Exception)}
