from fastapi import APIRouter, HTTPException
from app.services.news_service import NewsService
from app.core.config import settings

router = APIRouter()

@router.get("/", summary="News for all tickers")
async def get_all_news():
    return await NewsService.fetch_all()

@router.get("/{ticker}", summary="News for a specific ticker")
async def get_news(ticker: str):
    ticker = ticker.upper()
    if ticker not in settings.tickers:
        raise HTTPException(404, f"Ticker {ticker} not supported")
    return await NewsService.fetch(ticker)

@router.get("/{ticker}/premarket", summary="Pre-market brief for a ticker")
async def premarket_brief(ticker: str):
    ticker = ticker.upper()
    news = await NewsService.fetch(ticker)
    top  = news["articles"][:5]
    return {
        "ticker":       ticker,
        "sentiment":    news["overall_sentiment"],
        "intraday_est": news["intraday_impact"],
        "material":     news["material_events"],
        "top_stories":  [{"headline": a["headline"], "sentiment": a["sentiment"],
                          "impact": a["impact"]} for a in top],
    }
