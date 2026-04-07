"""Pydantic schemas — request/response contracts for every endpoint."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Signal ────────────────────────────────────────────────────────────────────
class SignalResponse(BaseModel):
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
    top_features:     list[list]   # [[name, importance], ...]
    sector:           str
    market_cap:       float
    pe:               float
    rev_growth:       float
    gross_margin:     float
    net_margin:       float
    fcf:              float
    r40:              float


# ── Prediction ────────────────────────────────────────────────────────────────
class IntraDayPrediction(BaseModel):
    ticker:            str
    generated_at:      datetime
    direction:         str        # UP / DOWN / FLAT
    confidence:        float
    expected_range_lo: float      # intraday price range estimate
    expected_range_hi: float
    catalyst:          str        # primary driver
    news_sentiment:    str        # BULLISH / BEARISH / NEUTRAL
    technical_bias:    str


class WeeklyPrediction(BaseModel):
    ticker:       str
    generated_at: datetime
    week_targets: list[dict]  # [{week: 1, price: x, pct: y}, ...]
    model_signal: str
    lgb_estimate: float
    lstm_prices:  list[float]
    conviction:   str            # HIGH / MEDIUM / LOW


# ── News ──────────────────────────────────────────────────────────────────────
class NewsArticle(BaseModel):
    ticker:    str
    headline:  str
    sentiment: str        # BULLISH / BEARISH / NEUTRAL
    impact:    str        # HIGH / MEDIUM / LOW
    source:    str
    url:       str
    published: str
    net_score: int


class NewsResponse(BaseModel):
    ticker:            str
    generated_at:      datetime
    overall_sentiment: str
    articles:          list[NewsArticle]
    material_events:   int
    intraday_impact:   str   # UP / DOWN / FLAT
    cached:            bool


# ── Backtest ──────────────────────────────────────────────────────────────────
class BacktestResponse(BaseModel):
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
class HealthResponse(BaseModel):
    status:       str
    models_loaded: bool
    last_updated: Optional[datetime]
    uptime_s:     float
