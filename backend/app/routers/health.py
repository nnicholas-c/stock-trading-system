from fastapi import APIRouter
from datetime import datetime
from app.services.model_service import ModelService

router = APIRouter()

@router.get("/", summary="Health check")
async def health():
    return {
        "status":        "ok",
        "models_loaded": ModelService.is_loaded(),
        "last_updated":  datetime.now().isoformat(),
        "uptime_s":      round(ModelService.get_uptime(), 1),
        "tickers":       ["PLTR", "AAPL", "NVDA", "TSLA"],
    }
