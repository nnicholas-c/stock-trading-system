from pathlib import Path
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"

class Settings(BaseSettings):
    # Paths
    models_dir:   Path = REPO_ROOT / "trading_system" / "models"
    signals_dir:  Path = REPO_ROOT / "trading_system" / "signals"
    data_dir:     Path = REPO_ROOT / "data"
    research_dir: Path = REPO_ROOT / "trading_system" / "research"
    research_artifact_path: Path = REPO_ROOT / "trading_system" / "signals" / "research_forecasts.json"

    # API
    api_host:     str  = "0.0.0.0"
    api_port:     int  = 8000
    debug:        bool = False

    # Tickers
    tickers: list[str] = ["PLTR", "AAPL", "NVDA", "TSLA"]

    # News refresh interval (seconds)
    news_cache_ttl: int = 300   # 5 minutes

    # OpenAI
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AXIOM_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_model: str = "gpt-5.4-mini"
    openai_timeout_s: float = 20.0
    openai_news_enabled: bool = True

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug"}:
                return True
            if normalized in {"", "0", "false", "no", "off", "release", "prod", "production"}:
                return False
        return bool(value)

    model_config = SettingsConfigDict(
        env_file=(BACKEND_ROOT / ".env", REPO_ROOT / ".env"),
        env_prefix="AXIOM_",
        extra="ignore",
    )

settings = Settings()
