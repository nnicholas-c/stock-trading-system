from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.core.config import settings

router = APIRouter()

INVALIDATED_BACKTEST_NOTE = (
    "Legacy ticker-level backtest metrics were invalidated by the leakage audit. "
    "Use honest_backtest/results/summary_metrics.csv for the current reproducible headline."
)


def invalidated_payload(ticker: str | None = None) -> dict:
    payload = {
        "generated_at": datetime.now().isoformat(),
        "status": "not_validated",
        "message": INVALIDATED_BACKTEST_NOTE,
        "honest_backtest_results": "honest_backtest/results/summary_metrics.csv",
    }
    if ticker:
        payload["ticker"] = ticker
    return payload

@router.get("/{ticker}", summary="Backtest results for a ticker")
async def get_backtest(ticker: str):
    ticker = ticker.upper()
    if ticker not in settings.tickers:
        raise HTTPException(404)
    return invalidated_payload(ticker)

@router.get("/", summary="Backtest comparison across all tickers")
async def get_all_backtests():
    payload = invalidated_payload()
    payload["results"] = {}
    return payload
