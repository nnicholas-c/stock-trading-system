"""
OpenAI-backed helpers for enriching live market news with structured analysis.
Falls back gracefully when the API key is not configured.
"""

from __future__ import annotations

import json
from typing import Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from app.core.config import settings


class NewsArticleAnalysis(BaseModel):
    headline: str
    sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    impact: Literal["LOW", "MEDIUM", "HIGH"]
    is_material: bool
    net_score: int = Field(ge=-5, le=5)
    rationale: str = Field(
        description="One short sentence explaining the sentiment and impact."
    )


class NewsBatchAnalysis(BaseModel):
    overall_sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    intraday_impact: Literal["UP", "DOWN", "FLAT"]
    material_events: int = Field(ge=0)
    summary: str = Field(
        description="Two sentences max summarizing the market takeaway for the ticker."
    )
    articles: list[NewsArticleAnalysis]


class OpenAIService:
    _client: Optional[OpenAI] = None

    @classmethod
    def is_configured(cls) -> bool:
        return settings.openai_news_enabled and bool(settings.openai_api_key)

    @classmethod
    def get_client(cls) -> OpenAI:
        if cls._client is None:
            cls._client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_s,
            )
        return cls._client

    @classmethod
    def analyze_news(
        cls,
        ticker: str,
        company: str,
        articles: list[dict],
    ) -> Optional[NewsBatchAnalysis]:
        if not cls.is_configured() or not articles:
            return None

        payload = {
            "ticker": ticker,
            "company": company,
            "articles": [
                {
                    "headline": article.get("headline", ""),
                    "description": article.get("description", ""),
                    "source": article.get("source", ""),
                    "published": article.get("published", ""),
                }
                for article in articles[:8]
            ],
        }

        response = cls.get_client().responses.parse(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious equity-news analyst. "
                        "Use only the provided headlines and descriptions. "
                        "Classify each article for short-term trading impact on the named ticker. "
                        "Choose overall_sentiment from BULLISH, BEARISH, or NEUTRAL. "
                        "Choose intraday_impact from UP, DOWN, or FLAT. "
                        "Use HIGH impact only for clearly material catalysts like earnings, guidance, "
                        "major contracts, legal/regulatory actions, product launches, leadership changes, "
                        "M&A, or financing events. "
                        "Use net_score on a -5 to 5 scale where positive is bullish and negative is bearish. "
                        "Be conservative when headlines are mixed or ambiguous."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=True),
                },
            ],
            text_format=NewsBatchAnalysis,
        )

        return response.output_parsed
