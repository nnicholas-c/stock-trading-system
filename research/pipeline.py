from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pltr_premarket_context import fetch_live_premarket_context, get_upcoming_session_window, write_json
from research.catalog import (
    COMPANIES,
    COMPANY_SNAPSHOTS,
    EVENT_CATEGORY_MAP,
    EVENT_LEDGER,
    HORIZONS,
    MONTHLY_MACRO,
    REGIME_TO_CODE,
    TICKERS,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "daily"
MACRO_DIR = ROOT / "data" / "macro"
DOCS_DIR = ROOT / "docs"
SIGNALS_DIR = ROOT / "trading_system" / "signals"
RESEARCH_DIR = ROOT / "trading_system" / "research"
RUNS_DIR = RESEARCH_DIR / "runs"
CHAMPION_DIR = RESEARCH_DIR / "champion"
PLTR_DIR = ROOT / "trading_system" / "pltr_deep"
DRL_DIR = ROOT / "trading_system" / "drl"
BLOOMBERG_DIR = ROOT / "bloomberg"
US_EASTERN = ZoneInfo("America/New_York")
PAGES_TERMINAL_BUNDLE = DOCS_DIR / "terminal_live_bundle.json"
PAGES_TERMINAL_MANIFEST = DOCS_DIR / "terminal_manifest.json"
TIMEFRAME_WINDOWS = {
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
    "all": None,
}


@dataclass
class ResearchConfig:
    artifact_version: str = "research_v2"
    tickers: list[str] = field(default_factory=lambda: list(TICKERS))
    horizons: list[int] = field(default_factory=lambda: list(HORIZONS))
    min_train_size: int = 252
    validation_size: int = 20
    step_size: int = 20
    purge_size: int = 10
    min_bucket_samples: int = 8
    probability_cap_without_support: float = 0.65
    random_state: int = 42
    top_driver_count: int = 5
    light_mode: bool = False
    publish_live_outputs: bool = True
    output_namespace: str = "live"


class ConstantProbabilityModel:
    def __init__(self, probability: float):
        self.probability = float(np.clip(probability, 0.001, 0.999))

    def fit(self, _X: pd.DataFrame, _y: pd.Series) -> "ConstantProbabilityModel":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


class IdentityCalibrator:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(values, dtype=float), 0.001, 0.999)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def parse_reference_dt(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=US_EASTERN)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        parsed = datetime.fromisoformat(f"{value}T08:15:00-04:00")
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=US_EASTERN)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def namespace_root(config: ResearchConfig) -> Path:
    if not config.output_namespace or config.output_namespace == "live":
        return RESEARCH_DIR
    return RESEARCH_DIR / "sandboxes" / config.output_namespace


def champion_dir_for(config: ResearchConfig) -> Path:
    return namespace_root(config) / "champion"


def runs_dir_for(config: ResearchConfig) -> Path:
    return namespace_root(config) / "runs"


def champion_manifest_path_for(config: ResearchConfig) -> Path:
    return namespace_root(config) / "champion_manifest.json"


def champion_latest_path_for(config: ResearchConfig) -> Path:
    return namespace_root(config) / "champion_latest.json"


def premarket_latest_path_for(config: ResearchConfig) -> Path:
    return namespace_root(config) / "premarket_latest.json"


def latest_experimental_path_for(config: ResearchConfig) -> Path:
    return namespace_root(config) / "latest_experimental.json"


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0).rolling(period).mean()
    losses = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gains / (losses + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def ema_spread(series: pd.Series, short: int, long: int) -> pd.Series:
    return (series.ewm(span=short, adjust=False).mean() - series.ewm(span=long, adjust=False).mean()) / (series + 1e-10)


def safe_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).std().fillna(0.0)


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    if window < 2:
        return pd.Series(0.0, index=series.index)
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered**2).sum()) + 1e-10

    def _slope(values: np.ndarray) -> float:
        sample = np.asarray(values, dtype=float)
        centered = sample - sample.mean()
        numer = float((x_centered * centered).sum())
        return numer / denom / (abs(float(sample.mean())) + 1e-10)

    return series.rolling(window).apply(_slope, raw=True).fillna(0.0)


def directional_streak(mask: pd.Series) -> pd.Series:
    streak = np.zeros(len(mask), dtype=float)
    running = 0
    for idx, active in enumerate(mask.fillna(False).astype(bool).to_numpy()):
        running = running + 1 if active else 0
        streak[idx] = running
    return pd.Series(streak, index=mask.index)


def tanh_normalize(series: pd.Series, scale: float) -> pd.Series:
    values = np.asarray(series, dtype=float) / max(scale, 1e-10)
    return pd.Series(np.tanh(values), index=series.index)


def centered_flag(series: pd.Series) -> pd.Series:
    return pd.Series(np.asarray(series, dtype=float) * 2.0 - 1.0, index=series.index)


def trend_state_from_score(score: float) -> str:
    if score >= 0.25:
        return "BULLISH"
    if score <= -0.25:
        return "BEARISH"
    return "MIXED"


def trend_regime_from_score(score: float) -> str:
    if score >= 0.25:
        return "UP_TREND"
    if score <= -0.25:
        return "DOWN_TREND"
    return "RANGE"


def trend_direction_from_score(score: float) -> int:
    if score >= 0.25:
        return 1
    if score <= -0.25:
        return -1
    return 0


def safe_log_loss(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(labels) == 0:
        return 1.0
    return float(log_loss(labels, clip_probs(probs), labels=[0, 1]))


def load_ohlcv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame


def asof_join(left: pd.DataFrame, right: pd.DataFrame, on: str = "date") -> pd.DataFrame:
    left_sorted = left.sort_values(on).reset_index(drop=True)
    right_sorted = right.sort_values(on).reset_index(drop=True)
    return pd.merge_asof(left_sorted, right_sorted, on=on, direction="backward")


def load_monthly_macro_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, values in MONTHLY_MACRO.items():
        rows.append(
            {
                "date": pd.Timestamp(f"{key}-01"),
                "macro_fed": float(values["fed"]),
                "macro_cpi": float(values["cpi"]),
                "macro_vix_avg": float(values["vix_avg"]),
                "macro_y10": float(values["y10"]),
                "macro_regime": str(values["regime"]),
                "macro_regime_code": REGIME_TO_CODE[str(values["regime"])],
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_market_frame(ticker: str) -> pd.DataFrame:
    frame = load_ohlcv(DATA_DIR / f"{ticker}_daily.csv")
    frame["ticker"] = ticker
    for macro_ticker in ("SPY", "QQQ", "TLT", "GLD"):
        macro_path = (MACRO_DIR if macro_ticker in {"SPY", "QQQ", "TLT", "GLD"} else DATA_DIR) / f"{macro_ticker}_daily.csv"
        macro = load_ohlcv(macro_path)[["date", "close", "volume"]].rename(
            columns={"close": f"{macro_ticker.lower()}_close", "volume": f"{macro_ticker.lower()}_volume"}
        )
        frame = asof_join(frame, macro)
    frame = asof_join(frame, load_monthly_macro_frame())
    return frame


def load_price_history(symbol: str, is_macro: bool = False) -> pd.DataFrame:
    directory = MACRO_DIR if is_macro else DATA_DIR
    path = directory / f"{symbol}_daily.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame = load_ohlcv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def latest_monthly_macro_value(field: str) -> float:
    last_key = sorted(MONTHLY_MACRO.keys())[-1]
    return float(MONTHLY_MACRO[last_key][field])


def build_event_feature_frame(dates: pd.Series, ticker: str) -> pd.DataFrame:
    events = EVENT_LEDGER.get(ticker, [])
    rows: list[dict[str, Any]] = []
    future_earnings = [pd.Timestamp(evt["date"]) for evt in events if evt["category"] == "earnings"]
    for current_date in pd.to_datetime(dates):
        row = {
            "days_to_next_earnings": 90.0,
            "days_since_major_event": 90.0,
            "major_event_flag": 0.0,
            "premkt_hist_net": 0.0,
            "premkt_hist_article_count": 0.0,
            "premkt_hist_material_count": 0.0,
            "premkt_hist_competition_risk": 0.0,
            "premkt_hist_contract_signal": 0.0,
            "premkt_hist_earnings_signal": 0.0,
        }
        for mapped in EVENT_CATEGORY_MAP.values():
            row[mapped] = 0.0
        next_earnings = [evt_date for evt_date in future_earnings if evt_date >= current_date]
        if next_earnings:
            row["days_to_next_earnings"] = float((next_earnings[0] - current_date).days)
        last_major_days: list[int] = []
        for event in events:
            event_date = pd.Timestamp(event["date"])
            delta_days = int((current_date.normalize() - event_date.normalize()).days)
            if delta_days >= 0:
                last_major_days.append(delta_days)
            if 0 <= delta_days <= 21:
                decay = math.exp(-0.12 * delta_days)
                mapped = EVENT_CATEGORY_MAP.get(str(event["category"]))
                if mapped:
                    row[mapped] += float(event["sentiment"]) * decay
                if delta_days <= 1:
                    row["premkt_hist_net"] += float(event["sentiment"])
                    row["premkt_hist_article_count"] += 1.0
                    if abs(float(event["magnitude"])) >= 0.05 or str(event["category"]) in {"earnings", "contract", "political"}:
                        row["premkt_hist_material_count"] += 1.0
                    if str(event["category"]) in {"competition", "valuation"}:
                        row["premkt_hist_competition_risk"] += max(0.0, -float(event["sentiment"]))
                    if str(event["category"]) == "contract":
                        row["premkt_hist_contract_signal"] += max(0.0, float(event["sentiment"]))
                    if str(event["category"]) == "earnings":
                        row["premkt_hist_earnings_signal"] += float(event["sentiment"])
            if abs(delta_days) <= 1 and abs(float(event["magnitude"])) >= 0.05:
                row["major_event_flag"] = 1.0
        if last_major_days:
            row["days_since_major_event"] = float(min(last_major_days))
        if row["premkt_hist_article_count"] > 0:
            row["premkt_hist_net"] /= row["premkt_hist_article_count"]
        rows.append(row)
    return pd.DataFrame(rows, index=pd.to_datetime(dates))


def news_weight_from_context(live_context: dict[str, Any] | None) -> float:
    if not live_context:
        return 0.0
    count = int(live_context.get("article_count", 0))
    if count <= 0:
        return 0.0
    weight = min(1.0, 0.35 + count * 0.2)
    if live_context.get("used_recent_fallback"):
        weight *= 0.45
    if int(live_context.get("material_count", 0)) == 0:
        weight *= 0.8
    return float(np.clip(weight, 0.0, 1.0))


def is_in_premarket_window(published_dt: datetime, reference_dt: datetime | None = None) -> bool:
    session = get_upcoming_session_window(reference_dt)
    return bool(session["window_start"] <= published_dt.astimezone(session["now"].tzinfo) <= session["now"])


def add_trend_features(frame: pd.DataFrame, close: pd.Series, high: pd.Series, low: pd.Series, returns: pd.Series) -> pd.DataFrame:
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    for window in (3, 5, 10, 20, 60):
        frame[f"close_slope_{window}d"] = rolling_slope(close, window)

    frame["up_streak_days"] = directional_streak(returns > 0)
    frame["down_streak_days"] = directional_streak(returns < 0)
    frame["higher_high_ratio_5"] = (high.diff() > 0).astype(float).rolling(5).mean().fillna(0.0)
    frame["higher_high_ratio_10"] = (high.diff() > 0).astype(float).rolling(10).mean().fillna(0.0)
    frame["lower_low_ratio_5"] = (low.diff() < 0).astype(float).rolling(5).mean().fillna(0.0)
    frame["lower_low_ratio_10"] = (low.diff() < 0).astype(float).rolling(10).mean().fillna(0.0)
    frame["price_above_ma20"] = (close > ma20).astype(float)
    frame["price_above_ma50"] = (close > ma50).astype(float)
    frame["price_above_ma200"] = (close > ma200).astype(float)
    frame["ema_stack_alignment"] = (
        ((ema5 > ema20).astype(float) + (ema20 > ema50).astype(float) + (ema50 > ema200).astype(float)) / 3.0
    ) * 2.0 - 1.0
    frame["ma_stack_alignment"] = (
        (frame["price_above_ma20"] + (ma20 > ma50).astype(float) + (ma50 > ma200).astype(float)) / 3.0
    ) * 2.0 - 1.0
    frame["breakout_distance_20d"] = close / (high.rolling(20).max() + 1e-10) - 1.0
    frame["breakdown_distance_20d"] = close / (low.rolling(20).min() + 1e-10) - 1.0
    rolling_max_60 = close.rolling(60).max()
    rolling_min_60 = close.rolling(60).min()
    frame["drawdown_recovery_rate"] = ((close - rolling_min_60) / ((rolling_max_60 - rolling_min_60) + 1e-10)).clip(0.0, 1.0)
    frame["drawdown_recovery_delta_5d"] = frame["drawdown_recovery_rate"].diff(5)
    frame["trend_vol_ratio"] = frame["close_slope_20d"] / (frame["vol_20"] + 1e-10)

    for macro_name in ("spy", "qqq", "tlt", "gld"):
        macro_close = frame[f"{macro_name}_close"].astype(float)
        ratio = (close / macro_close.replace(0.0, np.nan)).ffill().fillna(1.0)
        frame[f"{macro_name}_slope_20d"] = rolling_slope(macro_close, 20)
        frame[f"{macro_name}_corr_20"] = returns.rolling(20).corr(macro_close.pct_change().fillna(0.0)).fillna(0.0)
        frame[f"rel_{macro_name}_trend_20d"] = rolling_slope(ratio, 20)

    frame["relative_strength_20d"] = frame["ret_20"] - frame["spy_close"].pct_change(20)
    frame["relative_strength_qqq_20d"] = frame["ret_20"] - frame["qqq_close"].pct_change(20)
    frame["market_alignment_flag"] = (
        (frame["close_slope_20d"] > frame["spy_slope_20d"]) & (frame["close_slope_20d"] > frame["qqq_slope_20d"])
    ).astype(float)
    frame["correlation_trend_20d"] = frame["spy_corr_20"].diff(5).fillna(0.0)
    frame["premkt_hist_net_3d"] = frame["premkt_hist_net"].rolling(3).mean().fillna(0.0)
    frame["premkt_hist_net_5d"] = frame["premkt_hist_net"].rolling(5).mean().fillna(0.0)
    frame["premkt_hist_material_3d"] = frame["premkt_hist_material_count"].rolling(3).sum().fillna(0.0)
    frame["premkt_hist_material_5d"] = frame["premkt_hist_material_count"].rolling(5).sum().fillna(0.0)
    frame["premkt_hist_shock"] = (
        (frame["premkt_hist_material_count"] >= 2) | (frame["premkt_hist_net"].abs() >= 0.75)
    ).astype(float)

    medium_component = (
        tanh_normalize(frame["close_slope_5d"], 0.006)
        + tanh_normalize(frame["close_slope_10d"], 0.004)
        + tanh_normalize(frame["close_slope_20d"], 0.003)
        + frame["ema_stack_alignment"]
        + frame["ma_stack_alignment"]
        + tanh_normalize(frame["breakout_distance_20d"], 0.04)
    ) / 6.0
    long_component = (
        tanh_normalize(frame["close_slope_60d"], 0.0018)
        + centered_flag(frame["price_above_ma200"])
        + frame["ma_stack_alignment"]
        + centered_flag(frame["drawdown_recovery_rate"])
        + tanh_normalize(frame["trend_vol_ratio"], 0.05)
    ) / 5.0
    relative_component = (
        tanh_normalize(frame["relative_strength_20d"], 0.08)
        + tanh_normalize(frame["relative_strength_qqq_20d"], 0.08)
        + tanh_normalize(frame["rel_spy_trend_20d"], 0.02)
        + tanh_normalize(frame["rel_qqq_trend_20d"], 0.02)
        + centered_flag(frame["market_alignment_flag"])
        - tanh_normalize(frame["tlt_slope_20d"], 0.015)
    ) / 6.0
    persistence_component = (
        tanh_normalize((frame["up_streak_days"] - frame["down_streak_days"]) / 5.0, 1.0)
        + (frame["higher_high_ratio_5"] - frame["lower_low_ratio_5"]).clip(-1.0, 1.0)
        + (frame["higher_high_ratio_10"] - frame["lower_low_ratio_10"]).clip(-1.0, 1.0)
        + tanh_normalize(frame["drawdown_recovery_delta_5d"], 0.12)
    ) / 4.0
    news_base = frame["premkt_hist_net_5d"].where(frame["premkt_hist_net_5d"] != 0.0, frame["premkt_hist_net_3d"])
    news_direction = np.sign(news_base)
    news_component = (
        tanh_normalize(frame["premkt_hist_net_3d"], 0.20)
        + tanh_normalize(frame["premkt_hist_net_5d"], 0.15)
        + tanh_normalize(frame["premkt_hist_material_5d"] * news_direction, 3.0)
        + (frame["premkt_hist_shock"] * news_direction)
    ) / 4.0

    frame["trend_component_medium"] = medium_component.clip(-1.0, 1.0)
    frame["trend_component_long"] = long_component.clip(-1.0, 1.0)
    frame["trend_component_relative"] = relative_component.clip(-1.0, 1.0)
    frame["trend_component_persistence"] = persistence_component.clip(-1.0, 1.0)
    frame["trend_component_news"] = news_component.clip(-1.0, 1.0)
    frame["trend_score"] = (
        0.30 * frame["trend_component_medium"]
        + 0.25 * frame["trend_component_long"]
        + 0.20 * frame["trend_component_relative"]
        + 0.15 * frame["trend_component_persistence"]
        + 0.10 * frame["trend_component_news"]
    ).clip(-1.0, 1.0)
    frame["trend_state_code"] = np.select(
        [frame["trend_score"] >= 0.25, frame["trend_score"] <= -0.25],
        [1.0, -1.0],
        default=0.0,
    )
    frame["trend_state"] = frame["trend_score"].apply(trend_state_from_score)
    frame["trend_regime"] = frame["trend_score"].apply(trend_regime_from_score)
    return frame


def build_feature_frame(ticker: str, live_context: dict[str, Any] | None = None) -> pd.DataFrame:
    frame = load_market_frame(ticker)
    frame["date"] = pd.to_datetime(frame["date"])
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    open_ = frame["open"].astype(float)
    volume = frame["volume"].astype(float)
    returns = close.pct_change().fillna(0.0)

    frame["ret_1"] = close.pct_change(1)
    frame["ret_2"] = close.pct_change(2)
    frame["ret_3"] = close.pct_change(3)
    frame["ret_5"] = close.pct_change(5)
    frame["ret_10"] = close.pct_change(10)
    frame["ret_20"] = close.pct_change(20)
    frame["gap"] = (open_ - close.shift(1)) / (close.shift(1) + 1e-10)
    frame["intraday"] = (close - open_) / (open_ + 1e-10)
    frame["range_pct"] = (high - low) / (close + 1e-10)
    frame["vol_5"] = safe_std(returns, 5) * math.sqrt(252)
    frame["vol_10"] = safe_std(returns, 10) * math.sqrt(252)
    frame["vol_20"] = safe_std(returns, 20) * math.sqrt(252)
    frame["ma_20_dist"] = close / (close.rolling(20).mean() + 1e-10) - 1.0
    frame["ma_50_dist"] = close / (close.rolling(50).mean() + 1e-10) - 1.0
    frame["ma_200_dist"] = close / (close.rolling(200).mean() + 1e-10) - 1.0
    frame["ema_5_20"] = ema_spread(close, 5, 20)
    frame["ema_10_50"] = ema_spread(close, 10, 50)
    frame["rsi_7"] = (rsi(close, 7) - 50.0) / 50.0
    frame["rsi_14"] = (rsi(close, 14) - 50.0) / 50.0
    frame["drawdown_60"] = close / (close.rolling(60).max() + 1e-10) - 1.0
    frame["volume_ratio_5_20"] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-10)
    frame["volume_ratio_1_20"] = volume / (volume.rolling(20).mean() + 1e-10)
    frame["spy_ret_1"] = frame["spy_close"].pct_change(1)
    frame["qqq_ret_1"] = frame["qqq_close"].pct_change(1)
    frame["tlt_ret_5"] = frame["tlt_close"].pct_change(5)
    frame["gld_ret_5"] = frame["gld_close"].pct_change(5)
    frame["rel_spy_1"] = frame["ret_1"] - frame["spy_ret_1"]
    frame["rel_qqq_5"] = frame["ret_5"] - frame["qqq_close"].pct_change(5)
    frame["spy_corr_20"] = returns.rolling(20).corr(frame["spy_close"].pct_change().fillna(0.0)).fillna(0.0)
    frame["risk_on"] = ((frame["spy_ret_1"] > 0) & (frame["tlt_ret_5"] < 0)).astype(float)
    frame["macro_regime_code"] = frame["macro_regime_code"].fillna(0.0)

    snapshot = COMPANY_SNAPSHOTS[ticker]
    for key, value in snapshot.items():
        if isinstance(value, (float, int)):
            frame[f"company_{key}"] = float(value)

    event_frame = build_event_feature_frame(frame["date"], ticker)
    event_frame = event_frame.reset_index(drop=True)
    frame = pd.concat([frame.reset_index(drop=True), event_frame], axis=1)

    frame["premkt_live_net"] = 0.0
    frame["premkt_live_article_count"] = 0.0
    frame["premkt_live_material_count"] = 0.0
    frame["premkt_live_competition_risk"] = 0.0
    frame["premkt_live_contract_signal"] = 0.0
    frame["premkt_live_earnings_signal"] = 0.0
    frame["premkt_live_used_fallback"] = 0.0
    frame["premkt_live_weight"] = 0.0

    if live_context and not frame.empty:
        last_idx = frame.index[-1]
        for key, value in live_context.get("feature_values", {}).items():
            if key in frame.columns:
                frame.loc[last_idx, key] = float(value)
        frame.loc[last_idx, "premkt_live_used_fallback"] = 1.0 if live_context.get("used_recent_fallback") else 0.0
        frame.loc[last_idx, "premkt_live_weight"] = news_weight_from_context(live_context)

    frame = add_trend_features(frame, close, high, low, returns)

    for lag in range(1, 11):
        frame[f"lag_ret_{lag}"] = frame["ret_1"].shift(lag)
        frame[f"lag_volume_{lag}"] = frame["volume_ratio_1_20"].shift(lag)
    frame["sequence_momentum_10"] = frame[[f"lag_ret_{lag}" for lag in range(1, 6)]].sum(axis=1)
    frame["sequence_reversal_10"] = frame["lag_ret_1"] - frame["lag_ret_5"]
    frame["sequence_volume_trend"] = frame[[f"lag_volume_{lag}" for lag in range(1, 4)]].mean(axis=1)
    frame["market_date"] = frame["date"].dt.strftime("%Y-%m-%d")
    frame["company_name"] = COMPANIES[ticker]

    numeric_cols = frame.select_dtypes(include=[np.number]).columns
    frame[numeric_cols] = frame[numeric_cols].replace([np.inf, -np.inf], np.nan)
    frame[numeric_cols] = frame[numeric_cols].ffill().fillna(0.0)
    return frame


def feature_columns_for_frame(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "ticker",
        "company_name",
        "date",
        "market_date",
        "macro_regime",
        "trend_score",
        "trend_state",
        "trend_regime",
        "trend_state_code",
    }
    return [column for column in frame.columns if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])]


def reduce_features_for_light_mode(features: list[str]) -> list[str]:
    preferred = [
        "ret_1",
        "ret_2",
        "ret_5",
        "ret_10",
        "ret_20",
        "gap",
        "intraday",
        "range_pct",
        "vol_5",
        "vol_10",
        "ma_20_dist",
        "ma_50_dist",
        "ema_5_20",
        "rsi_7",
        "rsi_14",
        "drawdown_60",
        "volume_ratio_1_20",
        "spy_ret_1",
        "qqq_ret_1",
        "rel_spy_1",
        "macro_fed",
        "macro_cpi",
        "macro_vix_avg",
        "macro_regime_code",
        "days_to_next_earnings",
        "days_since_major_event",
        "major_event_flag",
        "evt_earnings",
        "evt_analyst",
        "evt_contract",
        "evt_competition",
        "evt_macro",
        "premkt_hist_net",
        "premkt_hist_net_3d",
        "premkt_hist_net_5d",
        "premkt_hist_material_count",
        "premkt_hist_material_5d",
        "premkt_live_net",
        "premkt_live_material_count",
        "premkt_live_weight",
        "premkt_live_used_fallback",
        "close_slope_5d",
        "close_slope_20d",
        "relative_strength_20d",
        "up_streak_days",
        "down_streak_days",
        "higher_high_ratio_5",
        "lower_low_ratio_5",
        "price_above_ma20",
        "price_above_ma50",
        "price_above_ma200",
        "ma_stack_alignment",
        "breakout_distance_20d",
        "trend_vol_ratio",
        "trend_component_medium",
        "trend_component_relative",
        "trend_component_persistence",
        "trend_score",
        "lag_ret_1",
        "lag_ret_2",
        "lag_ret_3",
        "lag_ret_4",
        "lag_ret_5",
        "sequence_momentum_10",
        "sequence_reversal_10",
        "company_pe",
        "company_revenue_growth",
        "company_gross_margin",
        "company_r40",
        "company_analyst_upside",
    ]
    reduced = [feature for feature in preferred if feature in features]
    return reduced or features[:40]


def build_estimator(family: str, random_state: int, light_mode: bool = False) -> Any:
    if family == "linear":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=600, C=0.75, class_weight="balanced", random_state=random_state)),
            ]
        )
    if family == "tree":
        if light_mode:
            return GradientBoostingClassifier(
                n_estimators=45,
                learning_rate=0.06,
                max_depth=2,
                random_state=random_state,
            )
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.02,
            random_state=random_state,
        )
    if family == "sequence":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(16, 8) if light_mode else (32, 16),
                        alpha=1e-3,
                        max_iter=90 if light_mode else 220,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown family {family}")


def fit_estimator(family: str, X_train: pd.DataFrame, y_train: pd.Series, random_state: int, light_mode: bool = False) -> Any:
    positive_rate = float(np.clip(y_train.mean(), 0.001, 0.999))
    if y_train.nunique() < 2:
        return ConstantProbabilityModel(positive_rate)
    estimator = build_estimator(family, random_state, light_mode=light_mode)
    estimator.fit(X_train, y_train)
    return estimator


def candidate_families(config: ResearchConfig) -> tuple[str, ...]:
    return ("linear", "tree") if config.light_mode else ("linear", "tree", "sequence")


def clip_probs(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.001, 0.999)


def fit_calibrator(raw_probs: np.ndarray, labels: np.ndarray) -> Any:
    raw_probs = clip_probs(raw_probs)
    labels = np.asarray(labels, dtype=float)
    if len(raw_probs) < 30 or len(np.unique(labels)) < 2 or len(np.unique(np.round(raw_probs, 3))) < 6:
        return IdentityCalibrator()
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw_probs, labels)
    return calibrator


def apply_calibrator(calibrator: Any, raw_probs: np.ndarray) -> np.ndarray:
    return clip_probs(calibrator.predict(raw_probs))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    probs = clip_probs(probs)
    labels = np.asarray(labels, dtype=float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for idx in range(bins):
        lower, upper = boundaries[idx], boundaries[idx + 1]
        mask = (probs >= lower) & (probs < upper if idx < bins - 1 else probs <= upper)
        if not mask.any():
            continue
        error += abs(probs[mask].mean() - labels[mask].mean()) * (mask.sum() / len(probs))
    return float(error)


def bucket_table_from_predictions(probs: np.ndarray, labels: np.ndarray, realized_returns: np.ndarray) -> list[dict[str, Any]]:
    data = pd.DataFrame({"prob": clip_probs(probs), "label": labels, "realized_return": realized_returns})
    data["bucket"] = np.minimum((data["prob"] * 20).astype(int), 19)
    buckets: list[dict[str, Any]] = []
    for bucket, bucket_frame in data.groupby("bucket"):
        lower = bucket / 20.0
        upper = lower + 0.05
        buckets.append(
            {
                "bucket": int(bucket),
                "lower": round(lower, 2),
                "upper": round(min(1.0, upper), 2),
                "count": int(len(bucket_frame)),
                "avg_probability": round(float(bucket_frame["prob"].mean()), 4),
                "hit_rate": round(float(bucket_frame["label"].mean()), 4),
                "avg_return_pct": round(float(bucket_frame["realized_return"].mean() * 100.0), 4),
            }
        )
    return buckets


def bucket_for_probability(bucket_table: list[dict[str, Any]], probability: float) -> dict[str, Any] | None:
    for bucket in bucket_table:
        upper = float(bucket["upper"])
        if float(bucket["lower"]) <= probability <= upper or (probability >= 0.95 and upper >= 0.95):
            return bucket
    return bucket_table[-1] if bucket_table else None


def confidence_bucket_supported(bucket_table: list[dict[str, Any]], probability: float, minimum_count: int) -> bool:
    bucket = bucket_for_probability(bucket_table, probability)
    if not bucket:
        return False
    return int(bucket["count"]) >= minimum_count and float(bucket["hit_rate"]) >= min(probability, 0.65)


def strategy_series_from_predictions(probs: np.ndarray, realized_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = clip_probs(probs)
    realized_returns = np.asarray(realized_returns, dtype=float)
    positions = np.where(probs >= 0.55, 1.0, np.where(probs <= 0.45, -1.0, 0.0))
    strategy_returns = positions * realized_returns
    return positions, strategy_returns


def sharpe_ratio(returns: np.ndarray, horizon: int) -> float:
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0 or np.std(returns) < 1e-9:
        return 0.0
    annualizer = math.sqrt(252.0 / max(horizon, 1))
    return float(np.mean(returns) / np.std(returns) * annualizer)


def sortino_ratio(returns: np.ndarray, horizon: int) -> float:
    returns = np.asarray(returns, dtype=float)
    downside = returns[returns < 0]
    if len(returns) == 0 or len(downside) == 0 or np.std(downside) < 1e-9:
        return 0.0
    annualizer = math.sqrt(252.0 / max(horizon, 1))
    return float(np.mean(returns) / np.std(downside) * annualizer)


def max_drawdown(path: np.ndarray) -> float:
    if len(path) == 0:
        return 0.0
    running_peak = np.maximum.accumulate(path)
    drawdown = path / np.where(running_peak == 0.0, 1.0, running_peak) - 1.0
    return float(drawdown.min())


def build_walkforward_predictions(
    frame: pd.DataFrame,
    features: list[str],
    horizon: int,
    family: str,
    config: ResearchConfig,
) -> pd.DataFrame:
    working = frame.copy()
    working["future_return"] = working["close"].shift(-horizon) / working["close"] - 1.0
    working["target_up"] = (working["future_return"] > 0).astype(int)
    working = working.dropna(subset=["future_return"]).reset_index(drop=True)
    records: list[dict[str, Any]] = []
    n_rows = len(working)
    split_start = config.min_train_size
    while split_start + config.validation_size <= n_rows:
        val_slice = working.iloc[split_start : split_start + config.validation_size]
        local_embargo = 3 if float(val_slice["major_event_flag"].sum()) > 0 else 0
        train_end = split_start - max(config.purge_size, horizon) - local_embargo
        if train_end < config.min_train_size:
            split_start += config.step_size
            continue
        train_slice = working.iloc[:train_end]
        estimator = fit_estimator(
            family,
            train_slice[features],
            train_slice["target_up"],
            config.random_state,
            light_mode=config.light_mode,
        )
        raw_probs = estimator.predict_proba(val_slice[features])[:, 1]
        for idx, raw_prob in zip(val_slice.index, raw_probs):
            row = working.loc[idx]
            records.append(
                {
                    "date": row["date"],
                    "raw_probability": float(raw_prob),
                    "label": int(row["target_up"]),
                    "realized_return": float(row["future_return"]),
                    "regime": row["macro_regime"],
                    "trend_regime": row.get("trend_regime", "RANGE"),
                    "major_event": int(row["major_event_flag"]),
                }
            )
        split_start += config.step_size
    prediction_frame = pd.DataFrame(records)
    if prediction_frame.empty:
        return prediction_frame
    calibrator = fit_calibrator(prediction_frame["raw_probability"].to_numpy(), prediction_frame["label"].to_numpy())
    prediction_frame["probability"] = apply_calibrator(calibrator, prediction_frame["raw_probability"].to_numpy())
    return prediction_frame


def summarise_prediction_frame(prediction_frame: pd.DataFrame, horizon: int) -> dict[str, Any]:
    if prediction_frame.empty:
        return {
            "brier": 1.0,
            "log_loss": 1.0,
            "accuracy": 0.5,
            "recent20_accuracy": 0.5,
            "recent60_accuracy": 0.5,
            "recent20_brier": 1.0,
            "recent60_brier": 1.0,
            "recent20_log_loss": 1.0,
            "recent60_log_loss": 1.0,
            "ece": 0.5,
            "stability": 0.0,
            "trend_stability": 0.0,
            "strategy_return": 0.0,
            "bah_return": 0.0,
            "alpha": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "n_trades": 0,
            "portfolio_values": [],
            "bah_values": [],
            "regime_split": {},
            "trend_regime_split": {},
        }

    labels = prediction_frame["label"].to_numpy()
    probs = clip_probs(prediction_frame["probability"].to_numpy())
    realized = prediction_frame["realized_return"].to_numpy()
    _, strategy_returns = strategy_series_from_predictions(probs, realized)
    strategy_path = np.cumprod(1.0 + strategy_returns)
    bah_path = np.cumprod(1.0 + realized)
    recent20 = prediction_frame.tail(20)
    recent60 = prediction_frame.tail(60)
    regime_split = {}
    trend_regime_split = {}
    for regime, group in prediction_frame.groupby("regime"):
        regime_split[str(regime)] = {
            "count": int(len(group)),
            "accuracy": round(float(accuracy_score(group["label"], (group["probability"] >= 0.5).astype(int))), 4),
            "brier": round(float(brier_score_loss(group["label"], clip_probs(group["probability"].to_numpy()))), 4),
        }
    for trend_regime, group in prediction_frame.groupby("trend_regime"):
        trend_regime_split[str(trend_regime)] = {
            "count": int(len(group)),
            "accuracy": round(float(accuracy_score(group["label"], (group["probability"] >= 0.5).astype(int))), 4),
            "brier": round(float(brier_score_loss(group["label"], clip_probs(group["probability"].to_numpy()))), 4),
        }
    overall_accuracy = float(accuracy_score(labels, (probs >= 0.5).astype(int)))
    recent20_accuracy = float(accuracy_score(recent20["label"], (recent20["probability"] >= 0.5).astype(int))) if len(recent20) else overall_accuracy
    recent60_accuracy = float(accuracy_score(recent60["label"], (recent60["probability"] >= 0.5).astype(int))) if len(recent60) else overall_accuracy
    recent20_probs = clip_probs(recent20["probability"].to_numpy()) if len(recent20) else probs
    recent20_labels = recent20["label"].to_numpy() if len(recent20) else labels
    recent60_probs = clip_probs(recent60["probability"].to_numpy()) if len(recent60) else probs
    recent60_labels = recent60["label"].to_numpy() if len(recent60) else labels
    stability = float(
        np.clip(
            1.0 - abs(recent20_accuracy - overall_accuracy) - 0.7 * abs(recent60_accuracy - overall_accuracy),
            0.0,
            1.0,
        )
    )
    trend_accuracies = [split["accuracy"] for split in trend_regime_split.values() if int(split["count"]) >= 5]
    trend_stability = float(np.clip(1.0 - np.std(trend_accuracies), 0.0, 1.0)) if trend_accuracies else 0.5
    return {
        "brier": round(float(brier_score_loss(labels, probs)), 4),
        "log_loss": round(safe_log_loss(labels, probs), 4),
        "accuracy": round(overall_accuracy, 4),
        "recent20_accuracy": round(recent20_accuracy, 4),
        "recent60_accuracy": round(recent60_accuracy, 4),
        "recent20_brier": round(float(brier_score_loss(recent20_labels, recent20_probs)), 4),
        "recent60_brier": round(float(brier_score_loss(recent60_labels, recent60_probs)), 4),
        "recent20_log_loss": round(safe_log_loss(recent20_labels, recent20_probs), 4),
        "recent60_log_loss": round(safe_log_loss(recent60_labels, recent60_probs), 4),
        "ece": round(expected_calibration_error(probs, labels), 4),
        "stability": round(stability, 4),
        "trend_stability": round(trend_stability, 4),
        "strategy_return": round(float(strategy_path[-1] - 1.0), 4),
        "bah_return": round(float(bah_path[-1] - 1.0), 4),
        "alpha": round(float((strategy_path[-1] - 1.0) - (bah_path[-1] - 1.0)), 4),
        "sharpe": round(sharpe_ratio(strategy_returns, horizon), 4),
        "sortino": round(sortino_ratio(strategy_returns, horizon), 4),
        "max_drawdown": round(max_drawdown(strategy_path), 4),
        "n_trades": int(np.count_nonzero(np.abs(strategy_returns) > 0)),
        "portfolio_values": [round(float(value), 4) for value in strategy_path[-120:]],
        "bah_values": [round(float(value), 4) for value in bah_path[-120:]],
        "regime_split": regime_split,
        "trend_regime_split": trend_regime_split,
    }


def score_family(metrics: dict[str, Any]) -> float:
    return float(
        -1.8 * metrics["brier"]
        - 0.7 * metrics["log_loss"]
        + 0.55 * metrics["recent20_accuracy"]
        + 0.30 * metrics["recent60_accuracy"]
        + 0.20 * metrics["stability"]
        + 0.12 * metrics.get("trend_stability", 0.0)
        - 0.25 * metrics["ece"]
        + 0.15 * metrics["alpha"]
    )


def permutation_driver_importance(estimator: Any, X: pd.DataFrame, y: pd.Series, top_n: int, light_mode: bool = False) -> list[dict[str, Any]]:
    if light_mode:
        return []
    if len(X) < 40:
        return []
    sample = X.tail(min(len(X), 160))
    sample_y = y.tail(len(sample))
    try:
        result = permutation_importance(estimator, sample, sample_y, scoring="neg_brier_score", n_repeats=5, random_state=42)
    except Exception:
        return []
    order = np.argsort(result.importances_mean)[::-1][:top_n]
    return [
        {"feature": str(sample.columns[idx]), "importance": round(float(result.importances_mean[idx]), 5)}
        for idx in order
    ]


def subset_features(features: list[str], subset: str) -> list[str]:
    if subset == "price_only":
        blocked = (
            "spy_",
            "qqq_",
            "tlt_",
            "gld_",
            "macro_",
            "company_",
            "evt_",
            "premkt_",
            "days_to_",
            "news_",
            "relative_strength_",
            "rel_",
            "market_alignment_flag",
            "correlation_trend_",
            "trend_component_relative",
            "trend_component_news",
        )
        return [feature for feature in features if not feature.startswith(blocked)]
    if subset == "structured_no_news":
        blocked = ("premkt_", "news_", "trend_component_news")
        return [feature for feature in features if not feature.startswith(blocked)]
    if subset == "no_trend":
        blocked = (
            "close_slope_",
            "higher_high_",
            "lower_low_",
            "up_streak_",
            "down_streak_",
            "price_above_ma",
            "ema_stack_",
            "ma_stack_",
            "breakout_distance_",
            "breakdown_distance_",
            "drawdown_recovery_",
            "trend_",
            "relative_strength_",
            "market_alignment_flag",
            "correlation_trend_",
            "rel_spy_trend_",
            "rel_qqq_trend_",
            "rel_tlt_trend_",
            "rel_gld_trend_",
        )
        return [feature for feature in features if not feature.startswith(blocked)]
    return list(features)


def evaluate_ablation(frame: pd.DataFrame, horizon: int, family: str, features: list[str], config: ResearchConfig) -> dict[str, Any]:
    if config.light_mode:
        return {
            "price_only": {"feature_count": len(subset_features(features, "price_only")), "brier": None, "log_loss": None, "accuracy": None},
            "structured_no_news": {"feature_count": len(subset_features(features, "structured_no_news")), "brier": None, "log_loss": None, "accuracy": None},
            "no_trend": {"feature_count": len(subset_features(features, "no_trend")), "brier": None, "log_loss": None, "accuracy": None},
            "full": {"feature_count": len(features), "brier": None, "log_loss": None, "accuracy": None},
            "structured_events_delta_brier": None,
            "premarket_news_delta_brier": None,
            "trend_delta_brier": None,
            "meta_policy_used": True,
            "meta_policy_note": "Light mode skips extra ablation reruns for speed; full nightly runs persist full ablation reports.",
        }
    reports: dict[str, Any] = {}
    for subset_name in ("price_only", "structured_no_news", "no_trend", "full"):
        subset = subset_features(features, subset_name if subset_name != "full" else "full")
        prediction_frame = build_walkforward_predictions(frame, subset, horizon, family, config)
        metrics = summarise_prediction_frame(prediction_frame, horizon)
        reports[subset_name] = {
            "feature_count": len(subset),
            "brier": metrics["brier"],
            "log_loss": metrics["log_loss"],
            "accuracy": metrics["accuracy"],
        }
    reports["structured_events_delta_brier"] = round(reports["price_only"]["brier"] - reports["structured_no_news"]["brier"], 4)
    reports["premarket_news_delta_brier"] = round(reports["structured_no_news"]["brier"] - reports["full"]["brier"], 4)
    reports["trend_delta_brier"] = round(reports["no_trend"]["brier"] - reports["full"]["brier"], 4)
    reports["meta_policy_used"] = True
    reports["meta_policy_note"] = "Meta-policy adjusts trust and sizing only; it is not allowed to override direction."
    return reports


def confidence_supported_probability(probability: float, bucket_table: list[dict[str, Any]], config: ResearchConfig) -> tuple[float, bool]:
    supported = confidence_bucket_supported(bucket_table, probability, config.min_bucket_samples)
    if probability > config.probability_cap_without_support and not supported:
        return config.probability_cap_without_support, False
    return probability, supported


def recent_one_day_edge_is_weak(metrics: dict[str, Any]) -> bool:
    recent20_accuracy = float(metrics.get("recent20_accuracy", metrics.get("accuracy", 0.5)))
    recent60_accuracy = float(metrics.get("recent60_accuracy", metrics.get("accuracy", 0.5)))
    recent20_brier = float(metrics.get("recent20_brier", metrics.get("brier", 1.0)))
    recent20_log_loss = float(metrics.get("recent20_log_loss", metrics.get("log_loss", 1.0)))
    return (
        recent20_accuracy < 0.55
        or recent60_accuracy < 0.54
        or recent20_brier > 0.245
        or recent20_log_loss > 0.685
    )


def build_one_day_edge_assessment(horizon_payload: dict[str, Any], live_context: dict[str, Any] | None) -> dict[str, Any]:
    metrics = horizon_payload.get("recent_performance", {})
    recent20_accuracy = float(metrics.get("recent20_accuracy", metrics.get("accuracy", 0.5)))
    recent60_accuracy = float(metrics.get("recent60_accuracy", metrics.get("accuracy", 0.5)))
    recent20_brier = float(metrics.get("recent20_brier", metrics.get("brier", 1.0)))
    recent20_log_loss = float(metrics.get("recent20_log_loss", metrics.get("log_loss", 1.0)))
    supported = bool(horizon_payload.get("supported_probability", True))
    used_fallback = bool((live_context or {}).get("used_recent_fallback"))
    low_edge = recent_one_day_edge_is_weak(metrics)

    if low_edge:
        status = "low_edge"
        label = "TACTICAL / LOW EDGE"
        tone = "bad"
        summary = (
            f"Recent 1d walk-forward accuracy is {recent20_accuracy * 100:.1f}% / "
            f"{recent60_accuracy * 100:.1f}% over the last 20 / 60 matured calls, "
            "so the next-day read is being treated as a low-edge tactical signal."
        )
    elif used_fallback:
        status = "fallback"
        label = "TACTICAL / FALLBACK NEWS"
        tone = "warn"
        summary = (
            "The 1d model is usable, but the current live context is fallback-driven rather than "
            "a true pre-open headline window, so the next-day read remains tactical."
        )
    elif not supported:
        status = "capped"
        label = "TACTICAL / CAPPED"
        tone = "warn"
        summary = (
            "The 1d probability is not fully supported by its historical confidence bucket, "
            "so the live next-day read is capped and treated tactically."
        )
    else:
        status = "calibrated"
        label = "CALIBRATED 1D"
        tone = "good"
        summary = (
            f"Recent 1d walk-forward accuracy is {recent20_accuracy * 100:.1f}% / "
            f"{recent60_accuracy * 100:.1f}% with Brier {recent20_brier:.3f} and "
            f"log loss {recent20_log_loss:.3f}, so the next-day read is calibrated enough "
            "to use as a normal tactical input."
        )

    return {
        "status": status,
        "label": label,
        "tone": tone,
        "summary": summary,
        "recent20_accuracy": round(recent20_accuracy, 4),
        "recent60_accuracy": round(recent60_accuracy, 4),
        "recent20_brier": round(recent20_brier, 4),
        "recent20_log_loss": round(recent20_log_loss, 4),
        "supported_probability": supported,
        "used_recent_fallback": used_fallback,
        "is_low_edge": low_edge,
    }


def apply_one_day_reliability_guard(
    probability: float,
    metrics: dict[str, Any],
    horizon: int,
    live_context: dict[str, Any] | None,
) -> float:
    if horizon != 1:
        return float(np.clip(probability, 0.001, 0.999))

    recent20_accuracy = float(metrics.get("recent20_accuracy", metrics.get("accuracy", 0.5)))
    recent60_accuracy = float(metrics.get("recent60_accuracy", metrics.get("accuracy", 0.5)))
    recent20_brier = float(metrics.get("recent20_brier", metrics.get("brier", 1.0)))
    recent20_log_loss = float(metrics.get("recent20_log_loss", metrics.get("log_loss", 1.0)))

    shrink = 1.0
    if recent20_accuracy < 0.55:
        shrink -= min(0.30, (0.55 - recent20_accuracy) * 1.6)
    if recent60_accuracy < 0.55:
        shrink -= min(0.14, (0.55 - recent60_accuracy) * 1.1)
    if recent20_brier > 0.24:
        shrink -= min(0.12, (recent20_brier - 0.24) * 6.0)
    if recent20_log_loss > 0.67:
        shrink -= min(0.10, (recent20_log_loss - 0.67) * 1.4)
    if live_context and live_context.get("used_recent_fallback"):
        shrink -= 0.08

    shrink = float(np.clip(shrink, 0.35, 1.0))
    adjusted = 0.5 + (probability - 0.5) * shrink
    return float(np.clip(adjusted, 0.001, 0.999))


def expected_return_from_bucket(probability: float, bucket_table: list[dict[str, Any]], fallback_scale: float) -> float:
    bucket = bucket_for_probability(bucket_table, probability)
    if bucket and int(bucket["count"]) >= 3:
        return float(bucket["avg_return_pct"]) / 100.0
    return float(np.clip((probability - 0.5) * fallback_scale, -0.05, 0.05))


def apply_trend_prior(probability: float, latest_row: pd.Series, live_context: dict[str, Any] | None) -> float:
    trend_score = float(latest_row.get("trend_score", 0.0))
    trend_direction = trend_direction_from_score(trend_score)
    if trend_direction == 0:
        return float(np.clip(probability, 0.001, 0.999))
    material_count = int((live_context or {}).get("material_count", 0))
    categories = {str(cat).lower() for article in (live_context or {}).get("articles", []) for cat in article.get("categories", [])}
    impactful_window = material_count >= 2 or bool(categories.intersection({"earnings", "contract", "macro"}))
    anchor = 0.5 + trend_score * 0.18
    anchor_weight = 0.12 + min(0.08, abs(trend_score) * 0.12)
    news_score = float((live_context or {}).get("net_score", 0.0))
    probability_penalty = 0.0
    if news_score * trend_direction < -0.10 and not impactful_window:
        anchor_weight += 0.03
        probability_penalty += min(0.015, abs(news_score) * 0.04)
    elif news_score * trend_direction > 0.10:
        anchor_weight += 0.06
    elif impactful_window:
        anchor_weight = max(0.06, anchor_weight - 0.05)
    adjusted = probability * (1.0 - anchor_weight) + anchor * anchor_weight - probability_penalty
    return float(np.clip(adjusted, 0.001, 0.999))


def probability_aligns_with_trend(probability: float, trend_score: float) -> bool:
    trend_direction = trend_direction_from_score(trend_score)
    forecast_direction = 1 if probability >= 0.5 else -1
    if trend_direction == 0:
        return abs(probability - 0.5) <= 0.08
    return forecast_direction == trend_direction


def build_trend_snapshot(latest_row: pd.Series, probability: float) -> dict[str, Any]:
    trend_score = float(latest_row.get("trend_score", 0.0))
    return {
        "state": trend_state_from_score(trend_score),
        "score": round(trend_score, 4),
        "close_slope_5d": round(float(latest_row.get("close_slope_5d", 0.0)) * 100.0, 3),
        "close_slope_20d": round(float(latest_row.get("close_slope_20d", 0.0)) * 100.0, 3),
        "relative_strength_20d": round(float(latest_row.get("relative_strength_20d", 0.0)) * 100.0, 3),
        "streak_up_days": int(latest_row.get("up_streak_days", 0.0)),
        "streak_down_days": int(latest_row.get("down_streak_days", 0.0)),
        "price_above_ma20": bool(float(latest_row.get("price_above_ma20", 0.0)) >= 0.5),
        "price_above_ma50": bool(float(latest_row.get("price_above_ma50", 0.0)) >= 0.5),
        "price_above_ma200": bool(float(latest_row.get("price_above_ma200", 0.0)) >= 0.5),
        "trend_vol_ratio": round(float(latest_row.get("trend_vol_ratio", 0.0)), 4),
        "breakout_distance_20d": round(float(latest_row.get("breakout_distance_20d", 0.0)) * 100.0, 3),
        "news_trend_5d": round(float(latest_row.get("premkt_hist_net_5d", 0.0)), 4),
        "trend_supported": probability_aligns_with_trend(probability, trend_score),
    }


def meta_policy_adjustment(latest_row: pd.Series, live_context: dict[str, Any] | None, probability: float) -> dict[str, float]:
    drawdown = float(latest_row.get("drawdown_60", 0.0))
    volatility = float(latest_row.get("vol_10", 0.0))
    news_weight = news_weight_from_context(live_context)
    trend_score = float(latest_row.get("trend_score", 0.0))
    trend_direction = trend_direction_from_score(trend_score)
    news_score = float((live_context or {}).get("net_score", 0.0))
    risk_penalty = 0.0
    conviction_bonus = 0.0
    if drawdown < -0.12 and volatility > 0.45:
        risk_penalty += 0.08
    if live_context and live_context.get("used_recent_fallback"):
        risk_penalty += 0.06
    if news_weight > 0.6 and int((live_context or {}).get("material_count", 0)) >= 1:
        conviction_bonus += 0.03
    if float(latest_row.get("days_to_next_earnings", 90.0)) <= 3:
        risk_penalty += 0.06
    if probability_aligns_with_trend(probability, trend_score):
        conviction_bonus += 0.02 + min(0.03, abs(trend_score) * 0.04)
    elif trend_direction != 0:
        risk_penalty += 0.03 + min(0.04, abs(trend_score) * 0.05)
    if trend_direction != 0 and news_score * trend_direction > 0.12:
        conviction_bonus += 0.02
    if trend_direction != 0 and news_score * trend_direction < -0.12:
        risk_penalty += 0.03 if int((live_context or {}).get("material_count", 0)) < 2 else 0.015
    return {"risk_penalty": risk_penalty, "conviction_bonus": conviction_bonus}


def compute_live_trust(
    latest_row: pd.Series,
    probability: float,
    metrics: dict[str, Any],
    supported_probability: bool,
    live_context: dict[str, Any] | None,
    horizon: int = 1,
) -> float:
    base = 0.25
    base += 0.22 * float(metrics["recent20_accuracy"])
    base += 0.18 * float(metrics["recent60_accuracy"])
    base += 0.15 * float(metrics["stability"])
    base += 0.08 * float(metrics.get("trend_stability", 0.5))
    base += 0.10 * (1.0 - float(metrics["ece"]))
    base += 0.05 * min(1.0, abs(probability - 0.5) / 0.15)
    if supported_probability:
        base += 0.05
    adjustment = meta_policy_adjustment(latest_row, live_context, probability)
    base += adjustment["conviction_bonus"]
    base -= adjustment["risk_penalty"]
    if horizon == 1:
        if recent_one_day_edge_is_weak(metrics):
            base -= 0.06
        if live_context and live_context.get("used_recent_fallback"):
            base -= 0.04
    return float(np.clip(base, 0.20, 0.85))


def derive_signal(probability: float, trust: float, horizon: int, metrics: dict[str, Any], live_context: dict[str, Any] | None) -> str:
    buy_threshold = 0.58
    sell_threshold = 0.42
    trust_threshold = 0.55
    if horizon == 1:
        if recent_one_day_edge_is_weak(metrics):
            buy_threshold += 0.03
            sell_threshold -= 0.03
            trust_threshold += 0.04
        if live_context and live_context.get("used_recent_fallback"):
            buy_threshold += 0.01
            sell_threshold -= 0.01
            trust_threshold += 0.02
    if probability >= buy_threshold and trust >= trust_threshold:
        return "BUY"
    if probability <= sell_threshold and trust >= trust_threshold:
        return "SELL"
    return "HOLD"


def horizon_label(horizon: int) -> str:
    return f"{horizon}d"


def build_reasoning_drivers(
    ticker: str,
    horizon_payload: dict[str, Any],
    latest_row: pd.Series,
    live_context: dict[str, Any] | None,
    trend_snapshot: dict[str, Any],
) -> list[dict[str, str]]:
    probability = float(horizon_payload["probability_up"])
    expected_return = float(horizon_payload["expected_return_pct"])
    drivers = [
        {
            "title": "Champion model",
            "direction": "bull" if probability > 55 else "bear" if probability < 45 else "neutral",
            "detail": (
                f"{ticker} {horizon_payload['model_family']} champion for {horizon_payload['horizon_days']}d "
                f"outputs {probability:.1f}% up with {expected_return:+.2f}% expected return."
            ),
        },
        {
            "title": "Calibration and trust",
            "direction": "bull" if float(horizon_payload["trust_score"]) >= 55 else "neutral",
            "detail": (
                f"Trust is {horizon_payload['trust_score']:.1f}% with recent walk-forward accuracy "
                f"{horizon_payload['recent_performance']['recent20_accuracy'] * 100:.1f}% / "
                f"{horizon_payload['recent_performance']['recent60_accuracy'] * 100:.1f}%."
                f"{' The 1d edge is currently weak, so the live call is intentionally shrunk back toward neutral.' if horizon_payload['horizon_days'] == 1 and recent_one_day_edge_is_weak(horizon_payload['recent_performance']) else ''}"
            ),
        },
        {
            "title": "Daily trend structure",
            "direction": "bull" if trend_snapshot["state"] == "BULLISH" else "bear" if trend_snapshot["state"] == "BEARISH" else "neutral",
            "detail": (
                f"Trend is {trend_snapshot['state']} ({trend_snapshot['score']:+.2f}) with "
                f"5d slope {trend_snapshot['close_slope_5d']:+.3f}%/day, "
                f"20d slope {trend_snapshot['close_slope_20d']:+.3f}%/day, "
                f"relative strength {trend_snapshot['relative_strength_20d']:+.2f}%, and "
                f"{'trend-supported follow-through' if trend_snapshot['trend_supported'] else 'a signal still held back by conflicting tape/news'}."
            ),
        },
    ]
    if live_context:
        drivers.append(
            {
                "title": "Pre-open news window",
                "direction": "bull" if live_context.get("net_score", 0.0) > 0.35 else "bear" if live_context.get("net_score", 0.0) < -0.35 else "neutral",
                "detail": live_context.get("summary", "No live pre-open context."),
            }
        )
    drivers.append(
        {
            "title": "Technical context",
            "direction": "bull" if float(latest_row.get("ma_20_dist", 0.0)) > 0 and float(latest_row.get("ret_5", 0.0)) > 0 else "bear" if float(latest_row.get("drawdown_60", 0.0)) < -0.10 else "neutral",
            "detail": (
                f"RSI 14 is {(float(latest_row.get('rsi_14', 0.0)) * 50 + 50):.1f}, "
                f"5-day return is {float(latest_row.get('ret_5', 0.0)) * 100:+.2f}%, "
                f"and the stock sits {float(latest_row.get('ma_20_dist', 0.0)) * 100:+.2f}% from the 20-day mean."
            ),
        }
    )
    return drivers


def build_card(signal: dict[str, Any], ticker_payload: dict[str, Any]) -> dict[str, Any]:
    horizons = ticker_payload["horizons"]
    current_price = float(signal["current_price"])
    one_day = float(horizons["1d"]["expected_return_pct"])
    five_day = float(horizons["5d"]["expected_return_pct"])
    ten_day = float(horizons["10d"]["expected_return_pct"])
    live_news = ticker_payload["news"]
    net_score = float(live_news.get("net_score", 0.0))
    trend_snapshot = ticker_payload.get("trend_snapshot", {})
    return {
        "sig": signal["signal"],
        "sc": signal["signal"].lower(),
        "conf": round(float(signal["confidence"]) / 100.0, 3),
        "px": round(current_price, 2),
        "chg": round(float(ticker_payload["technical_snapshot"]["ret_1d_pct"]), 2),
        "l1h": round(net_score * 0.45, 2),
        "l4h": round(net_score, 2),
        "l1d": round(one_day, 2),
        "l5d": round(five_day, 2),
        "l10d": round(ten_day, 2),
        "l20d": round(float(np.clip(ten_day * 1.8, -18.0, 18.0)), 2),
        "tgt": round(float(signal["target_price"]), 2),
        "up": round(((float(signal["target_price"]) / current_price) - 1.0) * 100.0 if current_price else 0.0, 1),
        "trend_state": trend_snapshot.get("state", "MIXED"),
        "trend_score": trend_snapshot.get("score", 0.0),
        "trend_supported": trend_snapshot.get("trend_supported", True),
        "edge_label": ticker_payload.get("one_day_edge", {}).get("label"),
        "edge_status": ticker_payload.get("one_day_edge", {}).get("status"),
        "headline_summary": ticker_payload["summary"],
    }


def build_news_feed_items(ticker_payload: dict[str, Any]) -> list[dict[str, Any]]:
    signal = ticker_payload["signal"]
    articles = ticker_payload["news"].get("articles", [])[:4]
    if not articles:
        return []
    items: list[dict[str, Any]] = []
    for idx, article in enumerate(articles):
        net_score = float(article.get("net_score", 0.0))
        items.append(
            {
                "id": 12000 + idx + abs(hash((ticker_payload["ticker"], article.get("headline", "")))) % 1000,
                "ticker": ticker_payload["ticker"],
                "dir": "bull" if net_score >= 0 else "bear",
                "impact": article.get("impact", "LOW"),
                "vader": float(np.clip(net_score / 3.0, -1.0, 1.0)),
                "age": "0h",
                "url": article.get("url", ""),
                "headline": article.get("headline", ""),
                "source": article.get("source", "Google News"),
                "news_cat": ((article.get("categories") or ["premarket"])[0]).replace("_", " "),
                "summary": article.get("rationale") or ticker_payload["news"].get("summary", ""),
                "body": (
                    f"{article.get('description') or article.get('headline', '')}\n\n"
                    f"{article.get('rationale') or 'Premarket context incorporated into the live forecast.'}"
                ),
                "px_impact": f"{float(signal['pred_return_pct']) + net_score * 0.18:+.1f}% next-day bias",
                "horizons": [
                    {"l": "1D", "v": f"{float(ticker_payload['horizons']['1d']['expected_return_pct']) + net_score * 0.18:+.1f}%"},
                    {"l": "1W", "v": f"{float(ticker_payload['horizons']['5d']['expected_return_pct']) + net_score * 0.24:+.1f}%"},
                    {"l": "2W", "v": f"{float(ticker_payload['horizons']['10d']['expected_return_pct']) + net_score * 0.30:+.1f}%"},
                ],
            }
        )
    return items


def artifact_status_for_run(run_type: str) -> str:
    return "experimental" if run_type == "nightly_experimental" else "champion"


def pages_publish_mode_for_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    status = artifact_status_for_run(str(artifact.get("run_type", "")))
    return {
        "surface": "github_pages",
        "policy": "latest_available",
        "backend_policy": "champion_only",
        "artifact_status": status,
        "label": "LATEST EXPERIMENTAL" if status == "experimental" else "CHAMPION",
    }


def format_compact_volume(value: float) -> str:
    amount = float(value)
    if abs(amount) >= 1_000_000_000:
        return f"{amount / 1_000_000_000:.2f}B"
    if abs(amount) >= 1_000_000:
        return f"{amount / 1_000_000:.2f}M"
    if abs(amount) >= 1_000:
        return f"{amount / 1_000:.1f}K"
    return f"{amount:.0f}"


def build_chart_timeframes(history: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    candles = history.copy()
    candles["date"] = pd.to_datetime(candles["date"])
    outputs: dict[str, list[dict[str, Any]]] = {}
    for key, window in TIMEFRAME_WINDOWS.items():
        subset = candles if window is None else candles.tail(window)
        outputs[key] = [
            {
                "d": row["date"].strftime("%Y-%m-%d"),
                "o": round(float(row["open"]), 2),
                "h": round(float(row["high"]), 2),
                "l": round(float(row["low"]), 2),
                "c": round(float(row["close"]), 2),
                "v": float(row["volume"]),
            }
            for _, row in subset.iterrows()
        ]
    return outputs


def build_quote_snapshot_from_history(history: pd.DataFrame) -> dict[str, Any]:
    latest = history.iloc[-1]
    prev_close = float(history["close"].iloc[-2]) if len(history) > 1 else float(latest["close"])
    close_px = float(latest["close"])
    change_pct = ((close_px / prev_close) - 1.0) * 100.0 if prev_close else 0.0
    return {
        "market_date": pd.Timestamp(latest["date"]).date().isoformat(),
        "open": round(float(latest["open"]), 2),
        "high": round(float(latest["high"]), 2),
        "low": round(float(latest["low"]), 2),
        "close": round(close_px, 2),
        "prev_close": round(prev_close, 2),
        "change_pct": round(change_pct, 2),
        "volume": int(float(latest["volume"])),
        "volume_label": format_compact_volume(float(latest["volume"])),
        "range_pct": round(((float(latest["high"]) - float(latest["low"])) / (close_px + 1e-10)) * 100.0, 2),
    }


def build_macro_snapshot(symbol: str) -> dict[str, Any]:
    history = load_price_history(symbol, is_macro=True)
    if history.empty:
        fallback_value = latest_monthly_macro_value("vix_avg") if symbol == "VIX" else 0.0
        return {
            "symbol": "^VIX" if symbol == "VIX" else symbol,
            "display_symbol": "VIX" if symbol == "VIX" else symbol,
            "price": round(float(fallback_value), 2),
            "change_pct": 0.0,
        }
    snapshot = build_quote_snapshot_from_history(history)
    return {
        "symbol": "^VIX" if symbol == "VIX" else symbol,
        "display_symbol": "VIX" if symbol == "VIX" else symbol,
        "price": snapshot["close"],
        "change_pct": snapshot["change_pct"],
        "market_date": snapshot["market_date"],
    }


def build_forecast_overlay(ticker_payload: dict[str, Any], latest_row: pd.Series) -> dict[str, Any]:
    signal = ticker_payload["signal"]
    current_price = float(signal["current_price"])
    trust = float(signal["trust_score"])
    vol = float(latest_row.get("vol_20", 0.25))
    one_day_edge = ticker_payload.get("one_day_edge", {})
    points = []
    for horizon_key, label in (("1d", "1D"), ("5d", "5D"), ("10d", "10D")):
        payload = ticker_payload["horizons"][horizon_key]
        base_move = abs(float(payload["expected_return_pct"]))
        band_pct = max(0.8, min(5.5, base_move * 0.7 + vol * 100.0 * 0.08 + (100.0 - trust) * 0.04))
        target = float(payload["target_price"])
        points.append(
            {
                "key": horizon_key,
                "label": label,
                "days": int(payload["horizon_days"]),
                "probability_up": float(payload["probability_up"]),
                "expected_return_pct": float(payload["expected_return_pct"]),
                "target_price": round(target, 2),
                "trust_score": float(payload["trust_score"]),
                "upper_price": round(current_price * (1.0 + (float(payload["expected_return_pct"]) + band_pct) / 100.0), 2),
                "lower_price": round(current_price * (1.0 + (float(payload["expected_return_pct"]) - band_pct) / 100.0), 2),
                "edge_label": one_day_edge.get("label") if horizon_key == "1d" else label,
                "edge_status": one_day_edge.get("status") if horizon_key == "1d" else "swing",
            }
        )
    return {
        "current_price": round(current_price, 2),
        "now_divider_label": "NOW",
        "points": points,
        "scenario_ladder": [
            {
                "label": point["label"],
                "probability_up": point["probability_up"],
                "expected_return_pct": point["expected_return_pct"],
                "target_price": point["target_price"],
                "trust_score": point["trust_score"],
                "edge_label": point["edge_label"],
                "edge_status": point["edge_status"],
            }
            for point in points
        ],
    }


def infer_news_section(article: dict[str, Any], used_fallback: bool) -> str:
    categories = {str(cat).lower() for cat in article.get("categories", [])}
    if used_fallback:
        return "Fallback"
    if article.get("in_window"):
        return "Premarket"
    if "macro" in categories:
        return "Macro"
    if "analyst" in categories:
        return "Analyst"
    if "earnings" in categories or "contract" in categories:
        return "Company"
    return "Today"


def build_news_monitor(ticker_payload: dict[str, Any]) -> dict[str, Any]:
    news = ticker_payload.get("news", {})
    used_fallback = bool(news.get("used_recent_fallback"))
    items = []
    for article in news.get("articles", []):
        section = infer_news_section(article, used_fallback)
        items.append(
            {
                "headline": article.get("headline", ""),
                "source": article.get("source", "Google News"),
                "url": article.get("url", ""),
                "published_at_et": article.get("published_at_et") or article.get("published"),
                "section": section,
                "impact": article.get("impact", "LOW"),
                "sentiment": article.get("sentiment", "NEUTRAL"),
                "net_score": round(float(article.get("net_score", 0.0)), 4),
                "categories": article.get("categories", []),
                "reason": article.get("rationale") or news.get("summary", ""),
                "description": article.get("description") or article.get("headline", ""),
            }
        )
    return {
        "summary": news.get("summary", ""),
        "article_count": int(news.get("article_count", len(items))),
        "material_count": int(news.get("material_count", 0)),
        "net_score": round(float(news.get("net_score", 0.0)), 4),
        "used_recent_fallback": used_fallback,
        "status_label": "FALLBACK NEWS" if used_fallback else "PREMARKET LIVE",
        "items": items,
    }


def build_catalyst_lane(ticker: str, ticker_payload: dict[str, Any], latest_row: pd.Series) -> list[dict[str, Any]]:
    snapshot = COMPANY_SNAPSHOTS[ticker]
    days_to_earnings = int(round(float(latest_row.get("days_to_next_earnings", 999.0))))
    next_earnings = None
    for event in EVENT_LEDGER.get(ticker, []):
        if event.get("category") == "earnings":
            evt_date = pd.Timestamp(event["date"])
            if evt_date >= pd.Timestamp(ticker_payload["market_date"]):
                next_earnings = evt_date.date().isoformat()
                break
    catalysts = [
        {
            "label": "Next earnings",
            "value": next_earnings or "TBD",
            "detail": f"{days_to_earnings} trading days",
            "tone": "neutral" if days_to_earnings > 7 else "risk",
        },
        {
            "label": "Analyst upside",
            "value": f"{float(snapshot['analyst_upside']):+.1f}%",
            "detail": "catalog snapshot",
            "tone": "bull" if float(snapshot["analyst_upside"]) >= 0 else "bear",
        },
        {
            "label": "Macro regime",
            "value": str(latest_row.get("macro_regime", "NEUTRAL")),
            "detail": f"Fed {float(latest_row.get('macro_fed', 0.0)):.2f}% · 10Y {float(latest_row.get('macro_y10', 0.0)):.2f}%",
            "tone": "bear" if str(latest_row.get("macro_regime", "NEUTRAL")) == "BEAR" else "bull" if str(latest_row.get("macro_regime", "NEUTRAL")) == "BULL" else "neutral",
        },
        {
            "label": "Relative strength",
            "value": f"{float(ticker_payload['trend_snapshot']['relative_strength_20d']):+.2f}%",
            "detail": "vs SPY over 20d",
            "tone": "bull" if float(ticker_payload["trend_snapshot"]["relative_strength_20d"]) >= 0 else "bear",
        },
    ]
    return catalysts


def build_levels_snapshot(history: pd.DataFrame) -> dict[str, Any]:
    close = history["close"].astype(float)
    latest = float(close.iloc[-1])
    return {
        "support_20d": round(float(history["low"].tail(20).min()), 2),
        "support_60d": round(float(history["low"].tail(60).min()), 2),
        "resistance_20d": round(float(history["high"].tail(20).max()), 2),
        "resistance_60d": round(float(history["high"].tail(60).max()), 2),
        "ma20": round(float(close.tail(20).mean()), 2),
        "ma50": round(float(close.tail(50).mean()), 2),
        "ma200": round(float(close.tail(200).mean()), 2),
        "distance_to_20d_high_pct": round((latest / (float(history["high"].tail(20).max()) + 1e-10) - 1.0) * 100.0, 2),
        "distance_to_20d_low_pct": round((latest / (float(history["low"].tail(20).min()) + 1e-10) - 1.0) * 100.0, 2),
    }


def build_market_context_snapshot(latest_row: pd.Series) -> dict[str, Any]:
    return {
        "macro_regime": str(latest_row.get("macro_regime", "NEUTRAL")),
        "spy_ret_1_pct": round(float(latest_row.get("spy_ret_1", 0.0)) * 100.0, 2),
        "qqq_ret_1_pct": round(float(latest_row.get("qqq_ret_1", 0.0)) * 100.0, 2),
        "tlt_ret_5_pct": round(float(latest_row.get("tlt_ret_5", 0.0)) * 100.0, 2),
        "gld_ret_5_pct": round(float(latest_row.get("gld_ret_5", 0.0)) * 100.0, 2),
        "risk_on": bool(float(latest_row.get("risk_on", 0.0)) >= 0.5),
    }


def build_terminal_ticker_bundle(ticker: str, ticker_payload: dict[str, Any]) -> dict[str, Any]:
    history = load_price_history(ticker)
    if history.empty:
        raise ValueError(f"Missing price history for {ticker}")
    frame = build_feature_frame(ticker)
    latest_row = frame.iloc[-1]
    quote_snapshot = build_quote_snapshot_from_history(history)
    market_date = quote_snapshot["market_date"]
    freshness = dict(ticker_payload.get("data_freshness", {}))
    freshness["quote_market_date"] = market_date
    freshness["is_stale"] = freshness.get("market_date") != market_date
    return {
        "ticker": ticker,
        "company_name": ticker_payload["company_name"],
        "market_date": market_date,
        "forecast_for_date": ticker_payload["forecast_for_date"],
        "quote_snapshot": quote_snapshot,
        "chart": {
            "timeframes": build_chart_timeframes(history),
            "default_timeframe": "3m",
        },
        "forecast_overlay": build_forecast_overlay(ticker_payload, latest_row),
        "news_monitor": build_news_monitor(ticker_payload),
        "catalysts": build_catalyst_lane(ticker, ticker_payload, latest_row),
        "levels": build_levels_snapshot(history),
        "market_context": build_market_context_snapshot(latest_row),
        "trend_snapshot": ticker_payload["trend_snapshot"],
        "one_day_edge": ticker_payload.get("one_day_edge", {}),
        "signal": ticker_payload["signal"],
        "card": ticker_payload["card"],
        "summary": ticker_payload["summary"],
        "reasoning": ticker_payload["reasoning"],
        "drivers": ticker_payload["drivers"],
        "recent_performance": ticker_payload["recent_performance"],
        "technical_snapshot": ticker_payload["technical_snapshot"],
        "component_scores": ticker_payload["component_scores"],
        "champion_model": ticker_payload["champion_model"],
        "data_freshness": freshness,
        "horizons": ticker_payload["horizons"],
        "news": ticker_payload["news"],
    }


def build_terminal_live_bundle(artifact: dict[str, Any], config: ResearchConfig) -> dict[str, Any]:
    macro_strip = [build_macro_snapshot(symbol) for symbol in ("SPY", "QQQ", "TLT", "GLD", "VIX")]
    tickers = {ticker: build_terminal_ticker_bundle(ticker, payload) for ticker, payload in artifact["tickers"].items()}
    ticker_strip = [
        {
            "ticker": ticker,
            "price": payload["quote_snapshot"]["close"],
            "change_pct": payload["quote_snapshot"]["change_pct"],
            "signal": payload["signal"]["signal"],
            "trust_score": payload["signal"]["trust_score"],
            "trend_state": payload["trend_snapshot"]["state"],
            "edge_label": payload.get("one_day_edge", {}).get("label"),
        }
        for ticker, payload in tickers.items()
    ]
    pages_meta = pages_publish_mode_for_artifact(artifact)
    latest_market_date = max(payload["market_date"] for payload in tickers.values())
    return {
        "artifact_version": artifact["artifact_version"],
        "artifact_status": pages_meta["artifact_status"],
        "generated_at": artifact["generated_at"],
        "market_date": latest_market_date,
        "forecast_for_date": artifact["forecast_for_date"],
        "run_id": artifact["run_id"],
        "run_type": artifact["run_type"],
        "pages_publish_mode": pages_meta,
        "default_ticker": "PLTR",
        "timeframes": list(TIMEFRAME_WINDOWS.keys()),
        "ticker_strip": ticker_strip,
        "macro_strip": macro_strip,
        "tickers": tickers,
        "news_feed": artifact.get("news_feed", []),
    }


def export_pages_terminal_bundle(artifact: dict[str, Any], config: ResearchConfig) -> None:
    if not config.publish_live_outputs:
        return
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    BLOOMBERG_DIR.mkdir(parents=True, exist_ok=True)
    bundle = build_terminal_live_bundle(artifact, config)
    dump_json(PAGES_TERMINAL_BUNDLE, bundle)
    dump_json(PAGES_TERMINAL_MANIFEST, bundle["pages_publish_mode"] | {"default_ticker": bundle["default_ticker"], "timeframes": bundle["timeframes"]})
    dump_json(BLOOMBERG_DIR / "terminal_live_bundle.json", bundle)


def train_ticker_horizon(
    ticker: str,
    frame: pd.DataFrame,
    horizon: int,
    config: ResearchConfig,
    live_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    features = feature_columns_for_frame(frame)
    if config.light_mode:
        features = reduce_features_for_light_mode(features)
    working = frame.copy()
    working["future_return"] = working["close"].shift(-horizon) / working["close"] - 1.0
    working["target_up"] = (working["future_return"] > 0).astype(int)
    matured = working.dropna(subset=["future_return"]).reset_index(drop=True)

    families = {}
    for family in candidate_families(config):
        prediction_frame = build_walkforward_predictions(matured, features, horizon, family, config)
        metrics = summarise_prediction_frame(prediction_frame, horizon)
        bucket_table = bucket_table_from_predictions(
            prediction_frame.get("probability", pd.Series(dtype=float)).to_numpy(),
            prediction_frame.get("label", pd.Series(dtype=float)).to_numpy(),
            prediction_frame.get("realized_return", pd.Series(dtype=float)).to_numpy(),
        ) if not prediction_frame.empty else []
        families[family] = {
            "family": family,
            "score": score_family(metrics),
            "prediction_frame": prediction_frame,
            "metrics": metrics,
            "bucket_table": bucket_table,
        }

    champion = max(families.values(), key=lambda item: item["score"])
    estimator = fit_estimator(
        champion["family"],
        matured[features],
        matured["target_up"],
        config.random_state,
        light_mode=config.light_mode,
    )
    calibrator = fit_calibrator(
        champion["prediction_frame"].get("raw_probability", pd.Series([matured["target_up"].mean()])).to_numpy(),
        champion["prediction_frame"].get("label", pd.Series([round(matured["target_up"].mean())])).to_numpy(),
    )

    latest_row = frame.iloc[[-1]]
    raw_probability = float(estimator.predict_proba(latest_row[features])[:, 1][0])
    probability = float(apply_calibrator(calibrator, np.array([raw_probability]))[0])
    probability = apply_trend_prior(probability, frame.iloc[-1], live_context)
    probability = apply_one_day_reliability_guard(probability, champion["metrics"], horizon, live_context)
    probability, supported_probability = confidence_supported_probability(probability, champion["bucket_table"], config)
    expected_return = expected_return_from_bucket(probability, champion["bucket_table"], max(0.025, float(frame["vol_20"].iloc[-1]) * 0.45))
    trust = compute_live_trust(frame.iloc[-1], probability, champion["metrics"], supported_probability, live_context, horizon)
    signal = derive_signal(probability, trust, horizon, champion["metrics"], live_context)
    current_price = float(frame["close"].iloc[-1])
    target_price = round(current_price * (1.0 + expected_return), 2)
    drivers = permutation_driver_importance(
        estimator,
        matured[features],
        matured["target_up"],
        config.top_driver_count,
        light_mode=config.light_mode,
    )
    ablation = evaluate_ablation(matured, horizon, champion["family"], features, config)
    bundle = {
        "ticker": ticker,
        "horizon": horizon,
        "family": champion["family"],
        "features": features,
        "estimator": estimator,
        "calibrator": calibrator,
        "metrics": champion["metrics"],
        "bucket_table": champion["bucket_table"],
        "drivers": drivers,
        "ablation": ablation,
        "model_version": f"{config.artifact_version}-{ticker.lower()}-{horizon_label(horizon)}-{champion['family']}",
        "market_date": frame["market_date"].iloc[-1],
    }
    horizon_payload = {
        "horizon_days": horizon,
        "model_family": champion["family"],
        "model_version": bundle["model_version"],
        "probability_up": round(probability * 100.0, 1),
        "expected_return_pct": round(expected_return * 100.0, 2),
        "target_price": target_price,
        "trust_score": round(trust * 100.0, 1),
        "supported_probability": supported_probability,
        "trend_supported": probability_aligns_with_trend(probability, float(frame["trend_score"].iloc[-1])),
        "evaluation_mode": "walk_forward_champion",
        "calibration": {
            "brier": champion["metrics"]["brier"],
            "log_loss": champion["metrics"]["log_loss"],
            "ece": champion["metrics"]["ece"],
            "bucket_table": champion["bucket_table"],
        },
        "recent_performance": {
            "accuracy": champion["metrics"]["accuracy"],
            "recent20_accuracy": champion["metrics"]["recent20_accuracy"],
            "recent60_accuracy": champion["metrics"]["recent60_accuracy"],
            "recent20_brier": champion["metrics"]["recent20_brier"],
            "recent60_brier": champion["metrics"]["recent60_brier"],
            "recent20_log_loss": champion["metrics"]["recent20_log_loss"],
            "recent60_log_loss": champion["metrics"]["recent60_log_loss"],
            "stability": champion["metrics"]["stability"],
            "trend_stability": champion["metrics"]["trend_stability"],
        },
        "backtest": {
            key: champion["metrics"][key]
            for key in (
                "strategy_return",
                "bah_return",
                "alpha",
                "sharpe",
                "sortino",
                "max_drawdown",
                "n_trades",
                "portfolio_values",
                "bah_values",
            )
        },
        "regime_split": champion["metrics"]["regime_split"],
        "trend_regime_split": champion["metrics"]["trend_regime_split"],
        "ablation": ablation,
        "top_drivers": drivers,
        "signal": signal,
    }
    if horizon == 1:
        horizon_payload["edge_assessment"] = build_one_day_edge_assessment(horizon_payload, live_context)
    return horizon_payload, bundle


def build_ticker_payload(
    ticker: str,
    frame: pd.DataFrame,
    live_context: dict[str, Any] | None,
    horizon_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    primary = horizon_payloads["1d"]
    market_date = frame["market_date"].iloc[-1]
    forecast_for_date = (live_context or {}).get("forecast_for_date", get_upcoming_session_window()["forecast_date"].isoformat())
    current_price = float(frame["close"].iloc[-1])
    trend_snapshot = build_trend_snapshot(frame.iloc[-1], float(primary["probability_up"]) / 100.0)
    one_day_edge = build_one_day_edge_assessment(primary, live_context)
    signal = {
        "signal": primary["signal"],
        "probability_up": primary["probability_up"],
        "confidence": primary["probability_up"] if primary["signal"] != "HOLD" else max(35.0, primary["trust_score"] * 0.9),
        "pred_return_pct": primary["expected_return_pct"],
        "current_price": round(current_price, 2),
        "target_price": primary["target_price"],
        "trust_score": primary["trust_score"],
        "trend_score": round(trend_snapshot["score"], 4),
        "trend_state": trend_snapshot["state"],
        "trend_supported": trend_snapshot["trend_supported"],
        "edge_status": one_day_edge["status"],
        "edge_label": one_day_edge["label"],
        "forecast_for_date": forecast_for_date,
        "date": market_date,
        "summary": (
            f"{ticker} is {primary['signal']} for {forecast_for_date} with "
            f"{primary['probability_up']:.1f}% odds up, {primary['expected_return_pct']:+.2f}% expected return, "
            f"{primary['trust_score']:.1f}% trust, a {trend_snapshot['state'].lower()} daily trend, "
            f"and a {one_day_edge['label'].lower()} 1d setup."
        ),
    }
    technical_snapshot = {
        "rsi14": round(float(frame["rsi_14"].iloc[-1]) * 50.0 + 50.0, 2),
        "ret_1d_pct": round(float(frame["ret_1"].iloc[-1]) * 100.0, 2),
        "ret_5d_pct": round(float(frame["ret_5"].iloc[-1]) * 100.0, 2),
        "ret_20d_pct": round(float(frame["ret_20"].iloc[-1]) * 100.0, 2),
        "pct_from_ma20": round(float(frame["ma_20_dist"].iloc[-1]) * 100.0, 2),
        "pct_from_ma50": round(float(frame["ma_50_dist"].iloc[-1]) * 100.0, 2),
        "volume_ratio": round(float(frame["volume_ratio_1_20"].iloc[-1]), 2),
    }
    data_freshness = {
        "market_date": market_date,
        "generated_at": datetime.now().astimezone().isoformat(),
        "rows": int(len(frame)),
        "used_live_premarket_news": bool(live_context),
        "used_recent_fallback": bool((live_context or {}).get("used_recent_fallback")),
    }
    ticker_payload = {
        "ticker": ticker,
        "company_name": COMPANIES[ticker],
        "market_date": market_date,
        "forecast_for_date": forecast_for_date,
        "primary_horizon": "1d",
        "signal": signal,
        "summary": signal["summary"],
        "one_day_edge": one_day_edge,
        "horizons": horizon_payloads,
        "champion_model": {
            "family": primary["model_family"],
            "model_version": primary["model_version"],
            "selection": "lowest Brier / log loss with recent stability and calibration support",
        },
        "calibration": primary["calibration"],
        "recent_performance": primary["recent_performance"],
        "data_freshness": data_freshness,
        "news": live_context
        or {
            "ticker": ticker,
            "forecast_for_date": forecast_for_date,
            "article_count": 0,
            "material_count": 0,
            "net_score": 0.0,
            "summary": "No live premarket refresh was requested for this artifact.",
            "articles": [],
            "used_recent_fallback": False,
            "feature_values": {},
        },
        "technical_snapshot": technical_snapshot,
        "trend_snapshot": trend_snapshot,
        "drivers": build_reasoning_drivers(ticker, primary, frame.iloc[-1], live_context, trend_snapshot),
        "reasoning": {
            "summary": (
                f"{ticker} carries a {trend_snapshot['state'].lower()} day-to-day structure "
                f"with score {trend_snapshot['score']:+.2f}; the live forecast is "
                f"{'aligned with' if trend_snapshot['trend_supported'] else 'leaning against'} that backdrop. "
                f"1d edge status: {one_day_edge['label']}."
            ),
            "drivers": build_reasoning_drivers(ticker, primary, frame.iloc[-1], live_context, trend_snapshot),
            "top_model_drivers": primary["top_drivers"],
        },
        "component_scores": {
            "probability_centered": round((primary["probability_up"] / 100.0) - 0.5, 4),
            "trust": round(primary["trust_score"] / 100.0, 4),
            "trend": round(trend_snapshot["score"], 4),
            "news_weight": round(news_weight_from_context(live_context), 4),
            "one_day_edge": -0.4 if one_day_edge["status"] == "low_edge" else -0.15 if one_day_edge["status"] in {"fallback", "capped"} else 0.25,
        },
    }
    ticker_payload["card"] = build_card(signal, ticker_payload)
    return ticker_payload


def save_bundle(bundle: dict[str, Any], config: ResearchConfig) -> None:
    champion_dir = champion_dir_for(config)
    champion_dir.mkdir(parents=True, exist_ok=True)
    path = champion_dir / f"{bundle['ticker']}_{bundle['horizon']}d_bundle.pkl"
    with open(path, "wb") as handle:
        pickle.dump(bundle, handle)


def load_bundle(ticker: str, horizon: int, config: ResearchConfig) -> dict[str, Any]:
    path = champion_dir_for(config) / f"{ticker}_{horizon}d_bundle.pkl"
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_existing_live_artifact(config: ResearchConfig) -> dict[str, Any] | None:
    if not config.publish_live_outputs:
        return None
    for path in (SIGNALS_DIR / "research_forecasts.json", champion_latest_path_for(config)):
        if not path.exists():
            continue
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            continue
        if payload.get("tickers"):
            return payload
    return None


def evaluate_artifact_promotion(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> tuple[bool, dict[str, Any]]:
    if not baseline or not baseline.get("tickers"):
        return True, {"reason": "No live baseline found; promoting new trend-aware champion by default.", "improved_tickers": list(candidate["tickers"].keys())}

    comparisons: dict[str, dict[str, float | bool]] = {}
    improved_tickers: list[str] = []
    for ticker, payload in candidate["tickers"].items():
        baseline_payload = baseline.get("tickers", {}).get(ticker, {})
        candidate_horizon = payload.get("horizons", {}).get("1d", {})
        baseline_horizon = baseline_payload.get("horizons", {}).get("1d", {})
        candidate_recent = candidate_horizon.get("recent_performance", {})
        baseline_recent = baseline_horizon.get("recent_performance", {})
        candidate_brier = float(candidate_recent.get("recent20_brier", candidate_horizon.get("calibration", {}).get("brier", 1.0)))
        baseline_brier = float(baseline_recent.get("recent20_brier", baseline_horizon.get("calibration", {}).get("brier", 1.0)))
        candidate_log_loss = float(candidate_recent.get("recent20_log_loss", candidate_horizon.get("calibration", {}).get("log_loss", 1.0)))
        baseline_log_loss = float(baseline_recent.get("recent20_log_loss", baseline_horizon.get("calibration", {}).get("log_loss", 1.0)))
        improved = candidate_brier < baseline_brier or candidate_log_loss < baseline_log_loss
        comparisons[ticker] = {
            "candidate_recent20_brier": round(candidate_brier, 4),
            "baseline_recent20_brier": round(baseline_brier, 4),
            "candidate_recent20_log_loss": round(candidate_log_loss, 4),
            "baseline_recent20_log_loss": round(baseline_log_loss, 4),
            "improved": improved,
        }
        if improved:
            improved_tickers.append(ticker)

    peer_improvements = [ticker for ticker in improved_tickers if ticker in {"AAPL", "NVDA", "TSLA"}]
    promote = "PLTR" in improved_tickers and len(peer_improvements) >= 2
    return promote, {
        "reason": "Promotion requires PLTR plus at least two peers to improve recent matured Brier or log loss.",
        "baseline_artifact_version": baseline.get("artifact_version"),
        "improved_tickers": improved_tickers,
        "comparisons": comparisons,
    }


def write_namespace_artifacts(artifact: dict[str, Any], config: ResearchConfig, run_type: str) -> None:
    root = namespace_root(config)
    root.mkdir(parents=True, exist_ok=True)
    dump_json(latest_experimental_path_for(config), artifact)
    if run_type == "nightly_champion":
        dump_json(champion_latest_path_for(config), artifact)
    if run_type == "premarket_refresh":
        dump_json(premarket_latest_path_for(config), artifact)


def export_compatibility_files(artifact: dict[str, Any], config: ResearchConfig) -> None:
    if not config.publish_live_outputs:
        return
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    PLTR_DIR.mkdir(parents=True, exist_ok=True)
    DRL_DIR.mkdir(parents=True, exist_ok=True)
    BLOOMBERG_DIR.mkdir(parents=True, exist_ok=True)

    dump_json(SIGNALS_DIR / "research_forecasts.json", artifact)
    dump_json(DOCS_DIR / "research_forecasts.json", artifact)
    write_json(SIGNALS_DIR / "tomorrow_premarket_forecast.json", artifact)
    write_json(DOCS_DIR / "live_tomorrow_forecasts.json", artifact)
    write_json(BLOOMBERG_DIR / "live_tomorrow_forecasts.json", artifact)
    dump_json(latest_experimental_path_for(config), artifact)
    if artifact.get("run_type") == "nightly_champion":
        dump_json(champion_latest_path_for(config), artifact)

    pltr_payload = artifact["tickers"]["PLTR"]
    write_json(DOCS_DIR / "pltr_live_context.json", pltr_payload)
    write_json(BLOOMBERG_DIR / "pltr_live_context.json", pltr_payload)
    dump_json(PLTR_DIR / "pltr_signal.json", pltr_payload["signal"])
    dump_json(
        PLTR_DIR / "pltr_deep_results.json",
        {
            "ticker": "PLTR",
            "generated_at": artifact["generated_at"],
            "market_date": pltr_payload["market_date"],
            "forecast_for_date": pltr_payload["forecast_for_date"],
            "primary_horizon": pltr_payload["primary_horizon"],
            "signal": pltr_payload["signal"],
            "horizons": pltr_payload["horizons"],
            "reasoning": pltr_payload["reasoning"],
            "news": pltr_payload["news"],
        },
    )
    dump_json(
        DRL_DIR / "drl_v2_results.json",
        {
            "generated_at": artifact["generated_at"],
            "evaluation_mode": "research_challenger_overlay",
            "signals": {
                ticker: {
                    "signal": payload["signal"]["signal"],
                    "xgb_prob_up": payload["horizons"]["1d"]["probability_up"],
                    "drl_proba_buy": round(
                        max(0.0, payload["horizons"]["1d"]["probability_up"] - (100.0 - payload["horizons"]["1d"]["trust_score"])) / 2.0,
                        1,
                    ),
                    "drl_proba_sell": round(
                        max(0.0, (100.0 - payload["horizons"]["1d"]["probability_up"]) - (100.0 - payload["horizons"]["1d"]["trust_score"])) / 2.0,
                        1,
                    ),
                    "combined_conf": payload["horizons"]["1d"]["trust_score"],
                    "current_price": payload["signal"]["current_price"],
                    "target_price": payload["signal"]["target_price"],
                }
                for ticker, payload in artifact["tickers"].items()
            },
        },
    )


def write_run_registry(run_id: str, artifact: dict[str, Any], bundles: dict[str, dict[str, Any]], config: ResearchConfig, promote_to_champion: bool = True) -> None:
    run_dir = runs_dir_for(config) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_json(run_dir / "artifact.json", artifact)
    dump_json(run_dir / "config.json", asdict(config))
    manifest = {
        "run_id": run_id,
        "generated_at": artifact["generated_at"],
        "run_type": artifact["run_type"],
        "tickers": {},
    }
    for bundle_key, bundle in bundles.items():
        with open(run_dir / f"{bundle['ticker']}_{bundle['horizon']}d_bundle.pkl", "wb") as handle:
            pickle.dump(bundle, handle)
        if promote_to_champion:
            save_bundle(bundle, config)
        dump_json(
            run_dir / f"{bundle['ticker']}_{bundle['horizon']}d_metrics.json",
            {
                "ticker": bundle["ticker"],
                "horizon": bundle["horizon"],
                "model_version": bundle["model_version"],
                "family": bundle["family"],
                "metrics": bundle["metrics"],
                "bucket_table": bundle["bucket_table"],
                "drivers": bundle["drivers"],
                "ablation": bundle["ablation"],
            },
        )
        manifest["tickers"].setdefault(bundle["ticker"], {})[horizon_label(bundle["horizon"])] = {
            "model_version": bundle["model_version"],
            "family": bundle["family"],
        }
    if promote_to_champion:
        dump_json(champion_manifest_path_for(config), manifest)
    dump_json(run_dir / "manifest.json", manifest)


def build_artifact_header(run_id: str, run_type: str, config: ResearchConfig, tickers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    generated_at = datetime.now().astimezone()
    forecast_for_date = next(iter(tickers.values()))["forecast_for_date"]
    market_date = max(payload["market_date"] for payload in tickers.values())
    news_feed = []
    for ticker_payload in tickers.values():
        news_feed.extend(build_news_feed_items(ticker_payload))
    return {
        "artifact_version": config.artifact_version,
        "artifact_status": artifact_status_for_run(run_type),
        "run_id": run_id,
        "run_type": run_type,
        "generated_at": generated_at.isoformat(),
        "market_date": market_date,
        "forecast_for_date": forecast_for_date,
        "methodology": {
            "summary": "Calibrated daily forecasting platform with anchored walk-forward evaluation, trend-first daily path modeling, trust scoring, and premarket inference refresh.",
            "selection": "Champion chosen by Brier score, log loss, recent walk-forward hit rate, regime stability, trend stability, and calibration.",
            "policy": "Signals are derived from probabilities plus trust; trend provides a structural prior, and meta-policy can shrink confidence but not override direction.",
        },
        "config_snapshot": asdict(config),
        "promotion": {
            "promoted": run_type != "nightly_experimental",
            "reason": "Premarket artifacts reuse the current champion; nightly runs may downgrade to experimental if promotion criteria fail.",
        },
        "pages_publish_mode": pages_publish_mode_for_artifact({"run_type": run_type}),
        "tickers": tickers,
        "news_feed": news_feed,
    }


def run_nightly_retrain(config: ResearchConfig | None = None) -> dict[str, Any]:
    config = config or ResearchConfig()
    baseline_artifact = load_existing_live_artifact(config)
    run_id = f"nightly-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    bundles: dict[str, dict[str, Any]] = {}
    tickers_payload: dict[str, dict[str, Any]] = {}
    for ticker in config.tickers:
        frame = build_feature_frame(ticker)
        horizon_payloads: dict[str, dict[str, Any]] = {}
        for horizon in config.horizons:
            payload, bundle = train_ticker_horizon(ticker, frame, horizon, config, live_context=None)
            horizon_payloads[horizon_label(horizon)] = payload
            bundles[f"{ticker}_{horizon}"] = bundle
        tickers_payload[ticker] = build_ticker_payload(ticker, frame, None, horizon_payloads)
    artifact = build_artifact_header(run_id, "nightly_champion", config, tickers_payload)
    promote, promotion = evaluate_artifact_promotion(artifact, baseline_artifact)
    if not promote:
        artifact["run_type"] = "nightly_experimental"
        artifact["artifact_status"] = "experimental"
    artifact["promotion"] = {"promoted": promote, **promotion}
    artifact["pages_publish_mode"] = pages_publish_mode_for_artifact(artifact)
    write_namespace_artifacts(artifact, config, artifact["run_type"])
    export_pages_terminal_bundle(artifact, config)
    if promote:
        export_compatibility_files(artifact, config)
    write_run_registry(run_id, artifact, bundles, config, promote_to_champion=promote)
    return artifact


def infer_from_bundle(bundle: dict[str, Any], frame: pd.DataFrame, live_context: dict[str, Any] | None, config: ResearchConfig) -> dict[str, Any]:
    features = bundle["features"]
    estimator = bundle["estimator"]
    calibrator = bundle["calibrator"]
    latest_row = frame.iloc[[-1]]
    raw_probability = float(estimator.predict_proba(latest_row[features])[:, 1][0])
    probability = float(apply_calibrator(calibrator, np.array([raw_probability]))[0])
    probability = apply_trend_prior(probability, frame.iloc[-1], live_context)
    probability = apply_one_day_reliability_guard(probability, bundle["metrics"], int(bundle["horizon"]), live_context)
    probability, supported_probability = confidence_supported_probability(probability, bundle["bucket_table"], config)
    expected_return = expected_return_from_bucket(probability, bundle["bucket_table"], max(0.025, float(frame["vol_20"].iloc[-1]) * 0.45))
    trust = compute_live_trust(frame.iloc[-1], probability, bundle["metrics"], supported_probability, live_context, int(bundle["horizon"]))
    signal = derive_signal(probability, trust, int(bundle["horizon"]), bundle["metrics"], live_context)
    current_price = float(frame["close"].iloc[-1])
    metrics = bundle.get("metrics", {})
    return {
        "horizon_days": bundle["horizon"],
        "model_family": bundle["family"],
        "model_version": bundle["model_version"],
        "probability_up": round(probability * 100.0, 1),
        "expected_return_pct": round(expected_return * 100.0, 2),
        "target_price": round(current_price * (1.0 + expected_return), 2),
        "trust_score": round(trust * 100.0, 1),
        "supported_probability": supported_probability,
        "trend_supported": probability_aligns_with_trend(probability, float(frame["trend_score"].iloc[-1])),
        "evaluation_mode": "premarket_refresh",
        "calibration": {
            "brier": bundle["metrics"]["brier"],
            "log_loss": bundle["metrics"]["log_loss"],
            "ece": bundle["metrics"]["ece"],
            "bucket_table": bundle["bucket_table"],
        },
        "recent_performance": {
            "accuracy": metrics.get("accuracy", 0.5),
            "recent20_accuracy": metrics.get("recent20_accuracy", metrics.get("accuracy", 0.5)),
            "recent60_accuracy": metrics.get("recent60_accuracy", metrics.get("accuracy", 0.5)),
            "recent20_brier": metrics.get("recent20_brier", metrics.get("brier", 1.0)),
            "recent60_brier": metrics.get("recent60_brier", metrics.get("brier", 1.0)),
            "recent20_log_loss": metrics.get("recent20_log_loss", metrics.get("log_loss", 1.0)),
            "recent60_log_loss": metrics.get("recent60_log_loss", metrics.get("log_loss", 1.0)),
            "stability": metrics.get("stability", 0.5),
            "trend_stability": metrics.get("trend_stability", 0.5),
        },
        "backtest": {
            key: metrics.get(key)
            for key in (
                "strategy_return",
                "bah_return",
                "alpha",
                "sharpe",
                "sortino",
                "max_drawdown",
                "n_trades",
                "portfolio_values",
                "bah_values",
            )
        },
        "regime_split": metrics.get("regime_split", {}),
        "trend_regime_split": metrics.get("trend_regime_split", {}),
        "ablation": bundle["ablation"],
        "top_drivers": bundle["drivers"],
        "signal": signal,
        "edge_assessment": build_one_day_edge_assessment(
            {
                "recent_performance": {
                    "accuracy": metrics.get("accuracy", 0.5),
                    "recent20_accuracy": metrics.get("recent20_accuracy", metrics.get("accuracy", 0.5)),
                    "recent60_accuracy": metrics.get("recent60_accuracy", metrics.get("accuracy", 0.5)),
                    "recent20_brier": metrics.get("recent20_brier", metrics.get("brier", 1.0)),
                    "recent60_brier": metrics.get("recent60_brier", metrics.get("brier", 1.0)),
                    "recent20_log_loss": metrics.get("recent20_log_loss", metrics.get("log_loss", 1.0)),
                    "recent60_log_loss": metrics.get("recent60_log_loss", metrics.get("log_loss", 1.0)),
                },
                "supported_probability": supported_probability,
            },
            live_context,
        ) if int(bundle["horizon"]) == 1 else None,
    }


def run_premarket_refresh(
    config: ResearchConfig | None = None,
    live_context_overrides: dict[str, dict[str, Any]] | None = None,
    reference_dt: datetime | str | None = None,
) -> dict[str, Any]:
    config = config or ResearchConfig()
    parsed_reference_dt = parse_reference_dt(reference_dt)
    manifest_path = champion_manifest_path_for(config)
    if not manifest_path.exists():
        run_nightly_retrain(config)
    run_id = f"premarket-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tickers_payload: dict[str, dict[str, Any]] = {}
    bundles_for_registry: dict[str, dict[str, Any]] = {}
    for ticker in config.tickers:
        live_context = (live_context_overrides or {}).get(ticker) or fetch_live_premarket_context(
            ticker=ticker,
            company_name=COMPANIES[ticker],
            reference_dt=parsed_reference_dt,
        )
        frame = build_feature_frame(ticker, live_context=live_context)
        horizon_payloads: dict[str, dict[str, Any]] = {}
        for horizon in config.horizons:
            bundle = load_bundle(ticker, horizon, config)
            bundles_for_registry[f"{ticker}_{horizon}"] = bundle
            horizon_payloads[horizon_label(horizon)] = infer_from_bundle(bundle, frame, live_context, config)
        tickers_payload[ticker] = build_ticker_payload(ticker, frame, live_context, horizon_payloads)
    artifact = build_artifact_header(run_id, "premarket_refresh", config, tickers_payload)
    artifact["pages_publish_mode"] = pages_publish_mode_for_artifact(artifact)
    write_namespace_artifacts(artifact, config, "premarket_refresh")
    export_pages_terminal_bundle(artifact, config)
    export_compatibility_files(artifact, config)
    return artifact
