from fastapi import APIRouter
from datetime import datetime
from app.services.model_service import ModelService

router = APIRouter()

@router.get("/", summary="Health check")
async def health():
    artifact = ModelService.get_all_forecasts()
    return {
        "status":        "ok",
        "models_loaded": ModelService.is_loaded(),
        "last_updated":  datetime.now().isoformat(),
        "uptime_s":      round(ModelService.get_uptime(), 1),
        "tickers":       ["PLTR", "AAPL", "NVDA", "TSLA"],
        "artifact_version": artifact.get("artifact_version"),
        "run_type": artifact.get("run_type"),
        "forecast_for_date": artifact.get("forecast_for_date"),
    }
