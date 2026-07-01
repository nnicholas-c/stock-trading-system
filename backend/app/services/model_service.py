"""
ModelService — loads all trained models at startup and provides
fast inference without re-loading on every request.
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from copy import deepcopy
import torch
import torch.nn as nn

from app.core.config import settings

# ── LSTM Architecture (must match training) ───────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden=128, layers=3, dropout=0.2, forecast_steps=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, forecast_steps)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attn(out), dim=1)
        ctx    = (attn_w * out).sum(dim=1)
        return self.head(ctx)


class ModelService:
    """Singleton — load once, serve forever."""

    _models:      dict = {}
    _signals:     dict = {}
    _artifact:    dict = {}
    _initialized: bool = False
    _startup_time: datetime = datetime.now()
    _signals_source: Optional[Path] = None

    @classmethod
    def _normalize_signal(cls, signal: dict) -> dict:
        normalized = dict(signal)

        if "current_price" in normalized and "price" not in normalized:
            normalized["price"] = normalized["current_price"]
        if "rsi_14" in normalized and "rsi14" not in normalized:
            normalized["rsi14"] = normalized["rsi_14"]
        if "macd_h" in normalized and "macd_hist" not in normalized:
            normalized["macd_hist"] = normalized["macd_h"]
        if "lstm_forecast_4w" in normalized and "lstm_forecast" not in normalized:
            normalized["lstm_forecast"] = normalized["lstm_forecast_4w"]
        if "lstm_forecast" in normalized and "lstm_forecast_4w" not in normalized:
            normalized["lstm_forecast_4w"] = normalized["lstm_forecast"]
        if "signal_label" in normalized and "signal" not in normalized:
            normalized["signal"] = normalized["signal_label"]

        return normalized

    @classmethod
    def _load_signal_cache(cls):
        cls._artifact = {}
        research_candidates = [
            settings.research_artifact_path,
            settings.research_dir / "champion_latest.json",
            settings.signals_dir / "tomorrow_premarket_forecast.json",
        ]
        selected_research = next((path for path in research_candidates if path.exists()), None)
        if selected_research is not None:
            with open(selected_research) as handle:
                payload = json.load(handle)
            if payload.get("tickers"):
                cls._artifact = payload
                cls._signals_source = selected_research
                cls._signals = {
                    "signals": {
                        ticker: cls._flatten_forecast(forecast)
                        for ticker, forecast in payload.get("tickers", {}).items()
                    }
                }
                return

        candidates = [
            settings.signals_dir / "current_signals_v2.json",
            settings.signals_dir / "current_signals.json",
            settings.signals_dir / "current_signals_v4.json",
            settings.signals_dir / "current_signals_v3.json",
        ]
        selected = next((path for path in candidates if path.exists()), None)
        if selected is None:
            cls._signals = {}
            cls._signals_source = None
            return

        with open(selected) as f:
            payload = json.load(f)

        signals = payload.get("signals", {})
        payload["signals"] = {
            ticker: cls._normalize_signal(signal)
            for ticker, signal in signals.items()
        }
        cls._signals = payload
        cls._signals_source = selected

    @classmethod
    def _flatten_forecast(cls, forecast: dict) -> dict:
        signal = dict(forecast.get("signal", {}))
        primary_horizon = str(forecast.get("primary_horizon", "1d"))
        horizon = dict(forecast.get("horizons", {}).get(primary_horizon, {}))
        trend_snapshot = dict(forecast.get("trend_snapshot", {}))
        signal_int = 1 if signal.get("signal") == "BUY" else -1 if signal.get("signal") == "SELL" else 0
        return cls._normalize_signal(
            {
                "ticker": forecast.get("ticker"),
                "generated_at": forecast.get("data_freshness", {}).get("generated_at"),
                "price": signal.get("current_price"),
                "current_price": signal.get("current_price"),
                "signal": signal.get("signal"),
                "signal_int": signal_int,
                "confidence": float(signal.get("confidence", 0.0)) / 100.0,
                "probability_up": signal.get("probability_up"),
                "target_price": signal.get("target_price"),
                "trust_score": signal.get("trust_score"),
                "trend_score": trend_snapshot.get("score", signal.get("trend_score", 0.0)),
                "trend_state": trend_snapshot.get("state", signal.get("trend_state")),
                "trend_supported": trend_snapshot.get("trend_supported", signal.get("trend_supported")),
                "edge_label": signal.get("edge_label") or forecast.get("one_day_edge", {}).get("label"),
                "edge_status": signal.get("edge_status") or forecast.get("one_day_edge", {}).get("status"),
                "forecast_for_date": signal.get("forecast_for_date"),
                "summary": signal.get("summary"),
                "horizons": forecast.get("horizons", {}),
                "card": forecast.get("card", {}),
                "recent_performance": forecast.get("recent_performance", {}),
                "data_freshness": forecast.get("data_freshness", {}),
                "news": forecast.get("news", {}),
                "trend_snapshot": trend_snapshot,
                "one_day_edge": forecast.get("one_day_edge", {}),
                "top_drivers": horizon.get("top_drivers", []),
                "backtest": horizon.get("backtest", {}),
            }
        )

    @classmethod
    async def initialize(cls):
        """Load all models for all tickers at startup."""
        if cls._initialized:
            return

        for ticker in settings.tickers:
            try:
                m = {}

                # RF + scaler
                rf_path = settings.models_dir / f"{ticker}_v2_rf.pkl"
                if rf_path.exists():
                    with open(rf_path, "rb") as f:
                        m["rf_bundle"] = pickle.load(f)

                # XGBoost
                xgb_path = settings.models_dir / f"{ticker}_v2_xgb.pkl"
                if xgb_path.exists():
                    with open(xgb_path, "rb") as f:
                        m["xgb"] = pickle.load(f)

                # LightGBM
                lgb_path = settings.models_dir / f"{ticker}_v2_lgb.pkl"
                if lgb_path.exists():
                    with open(lgb_path, "rb") as f:
                        m["lgb"] = pickle.load(f)

                # Meta-ensemble
                meta_path = settings.models_dir / f"{ticker}_v2_meta.pkl"
                if meta_path.exists():
                    with open(meta_path, "rb") as f:
                        m["meta"] = pickle.load(f)

                # LSTM
                lstm_path = settings.models_dir / f"{ticker}_lstm.pt"
                if lstm_path.exists():
                    model = LSTMForecaster()
                    model.load_state_dict(torch.load(lstm_path, map_location="cpu", weights_only=True))
                    model.eval()
                    m["lstm"] = model

                cls._models[ticker] = m
                print(f"  {ticker}: {list(m.keys())}")

            except Exception as e:
                print(f"  {ticker} model load error: {e}")

        cls._load_signal_cache()

        cls._initialized = True

    @classmethod
    def get_models(cls, ticker: str) -> dict:
        return cls._models.get(ticker, {})

    @classmethod
    def get_cached_signal(cls, ticker: str) -> Optional[dict]:
        signal = cls._signals.get("signals", {}).get(ticker)
        return deepcopy(signal) if signal else None

    @classmethod
    def get_forecast(cls, ticker: str) -> Optional[dict]:
        forecast = cls._artifact.get("tickers", {}).get(ticker)
        return deepcopy(forecast) if forecast else None

    @classmethod
    def get_all_forecasts(cls) -> dict:
        return deepcopy(cls._artifact)

    @classmethod
    def get_uptime(cls) -> float:
        return (datetime.now() - cls._startup_time).total_seconds()

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._initialized and (bool(cls._artifact) or len(cls._models) > 0)
