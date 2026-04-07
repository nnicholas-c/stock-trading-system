from fastapi import APIRouter, HTTPException
from datetime import datetime
from app.services.model_service import ModelService
from app.core.config import settings

router = APIRouter()

# Backtest results pre-computed during training
BACKTEST_CACHE = {
    "PLTR": dict(start_date="2023-07-01", end_date="2026-04-07",
                 strategy_return=1.879, bah_return=1.250, alpha=0.629,
                 sharpe=1.65, sortino=2.1, max_drawdown=-0.31, n_trades=18,
                 win_rate=0.67, rl_return=7.88, rl_sharpe=1.97, rl_max_dd=-0.32),
    "AAPL": dict(start_date="2023-07-01", end_date="2026-04-07",
                 strategy_return=0.699, bah_return=0.054, alpha=0.645,
                 sharpe=2.01, sortino=2.8, max_drawdown=-0.14, n_trades=12,
                 win_rate=0.75, rl_return=0.727, rl_sharpe=1.08, rl_max_dd=-0.139),
    "NVDA": dict(start_date="2023-07-01", end_date="2026-04-07",
                 strategy_return=1.111, bah_return=0.304, alpha=0.807,
                 sharpe=2.17, sortino=3.1, max_drawdown=-0.22, n_trades=15,
                 win_rate=0.73, rl_return=4.571, rl_sharpe=2.07, rl_max_dd=-0.321),
    "TSLA": dict(start_date="2023-07-01", end_date="2026-04-07",
                 strategy_return=0.908, bah_return=0.133, alpha=0.775,
                 sharpe=1.87, sortino=2.4, max_drawdown=-0.35, n_trades=22,
                 win_rate=0.64, rl_return=0.878, rl_sharpe=0.72, rl_max_dd=-0.420),
}

@router.get("/{ticker}", summary="Backtest results for a ticker")
async def get_backtest(ticker: str):
    ticker = ticker.upper()
    if ticker not in settings.tickers:
        raise HTTPException(404)
    bt = BACKTEST_CACHE.get(ticker, {})
    return {"ticker": ticker, "generated_at": datetime.now().isoformat(), **bt}

@router.get("/", summary="Backtest comparison across all tickers")
async def get_all_backtests():
    return {
        "generated_at": datetime.now().isoformat(),
        "results": {t: {"ticker": t, **v} for t, v in BACKTEST_CACHE.items()},
        "portfolio": {
            "avg_alpha":    sum(v["alpha"] for v in BACKTEST_CACHE.values()) / 4,
            "best_sharpe":  max(v["sharpe"] for v in BACKTEST_CACHE.values()),
            "best_ticker":  max(BACKTEST_CACHE, key=lambda t: BACKTEST_CACHE[t]["sharpe"]),
        }
    }
