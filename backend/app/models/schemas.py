"""Pydantic schemas — request/response contracts for every endpoint."""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from datetime import datetime


class SchemaModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


# ── Signal ────────────────────────────────────────────────────────────────────
class SignalResponse(SchemaModel):
    ticker:           str
    generated_at:     datetime
    price:            float
    signal:           str          # STRONG BUY / BUY / HOLD / SELL
    signal_int:       int          # 2=SB, 1=B, 0=H, -1=S
    confidence:       float        # 0–1
    color:            str          # hex
    analyst_target:   float
    analyst_upside:   float        # %
    bull_pct:         float        # % analysts bullish
    lgb_fwd_ret:      float        # LightGBM 4-week return estimate
    lstm_forecast:    list[float]  # [W+1, W+2, W+3, W+4] prices
    vol_regime:       str          # LOW_VOL / MED_VOL / HIGH_VOL
    current_vol:      float
    risk_score:       float        # 0–10
    rsi14:            float
    macd_hist:        float
    trend_score:      float
    trend_state:      Optional[str] = None
    trend_supported:  Optional[bool] = None
    edge_label:       Optional[str] = None
    edge_status:      Optional[str] = None
    top_features:     list[list]   # [[name, importance], ...]
    sector:           str
    market_cap:       float
    pe:               float
    rev_growth:       float
    gross_margin:     float
    net_margin:       float
    fcf:              float
    r40:              float


class HorizonForecast(SchemaModel):
    horizon_days:        int
    model_family:        str
    model_version:       str
    probability_up:      float
    expected_return_pct: float
    target_price:        float
    trust_score:         float
    supported_probability: bool
    trend_supported:     bool
    edge_assessment:     Optional[dict] = None
    evaluation_mode:     str
    calibration:         dict
    recent_performance:  dict
    backtest:            dict
    regime_split:        dict
    trend_regime_split:  dict
    ablation:            dict
    top_drivers:         list[dict]
    signal:              str


class CanonicalForecastResponse(SchemaModel):
    ticker:             str
    company_name:       str
    market_date:        str
    forecast_for_date:  str
    primary_horizon:    str
    signal:             dict
    summary:            str
    horizons:           dict[str, HorizonForecast]
    champion_model:     dict
    calibration:        dict
    recent_performance: dict
    data_freshness:     dict
    news:               dict
    technical_snapshot: dict
    trend_snapshot:     dict
    one_day_edge:       Optional[dict] = None
    drivers:            list[dict]
    reasoning:          dict
    component_scores:   dict
    card:               dict


# ── Prediction ────────────────────────────────────────────────────────────────
class IntraDayPrediction(SchemaModel):
    ticker:            str
    generated_at:      datetime
    direction:         str        # UP / DOWN / FLAT
    confidence:        float
    expected_range_lo: float      # intraday price range estimate
    expected_range_hi: float
    catalyst:          str        # primary driver
    news_sentiment:    str        # BULLISH / BEARISH / NEUTRAL
    technical_bias:    str


class WeeklyPrediction(SchemaModel):
    ticker:       str
    generated_at: datetime
    week_targets: list[dict]  # [{week: 1, price: x, pct: y}, ...]
    model_signal: str
    lgb_estimate: float
    lstm_prices:  list[float]
    conviction:   str            # HIGH / MEDIUM / LOW


# ── News ──────────────────────────────────────────────────────────────────────
class NewsArticle(SchemaModel):
    ticker:    str
    headline:  str
    sentiment: str        # BULLISH / BEARISH / NEUTRAL
    impact:    str        # HIGH / MEDIUM / LOW
    source:    str
    url:       str
    published: str
    net_score: int
    is_material: bool = False
    rationale: Optional[str] = None


class NewsResponse(SchemaModel):
    ticker:            str
    generated_at:      datetime
    overall_sentiment: str
    articles:          list[NewsArticle]
    material_events:   int
    intraday_impact:   str   # UP / DOWN / FLAT
    summary:           Optional[str] = None
    analysis_provider: Optional[str] = None
    analysis_model:    Optional[str] = None
    cached:            bool


# ── Backtest ──────────────────────────────────────────────────────────────────
class BacktestResponse(SchemaModel):
    ticker:           str
    start_date:       str
    end_date:         str
    strategy_return:  float
    bah_return:       float
    alpha:            float
    sharpe:           float
    sortino:          float
    max_drawdown:     float
    n_trades:         int
    win_rate:         float
    portfolio_values: list[float]
    bah_values:       list[float]
    rl_return:        float
    rl_sharpe:        float
    rl_max_dd:        float


# ── Health ────────────────────────────────────────────────────────────────────
class HealthResponse(SchemaModel):
    status:       str
    models_loaded: bool
    last_updated: Optional[datetime]
    uptime_s:     float
