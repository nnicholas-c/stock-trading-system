from pydantic_settings import BaseSettings
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent  # repo root

class Settings(BaseSettings):
    # Paths
    models_dir:   Path = ROOT / "trading_system" / "models"
    signals_dir:  Path = ROOT / "trading_system" / "signals"
    data_dir:     Path = ROOT / "data"

    # API
    api_host:     str  = "0.0.0.0"
    api_port:     int  = 8000
    debug:        bool = False

    # Tickers
    tickers: list[str] = ["PLTR", "AAPL", "NVDA", "TSLA"]

    # News refresh interval (seconds)
    news_cache_ttl: int = 300   # 5 minutes

    class Config:
        env_file = ".env"

settings = Settings()
