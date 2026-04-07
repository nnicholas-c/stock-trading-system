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
    _initialized: bool = False
    _startup_time: datetime = datetime.now()

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
                    model.load_state_dict(torch.load(lstm_path, map_location="cpu"))
                    model.eval()
                    m["lstm"] = model

                cls._models[ticker] = m
                print(f"  ✓ {ticker}: {list(m.keys())}")

            except Exception as e:
                print(f"  ⚠ {ticker} model load error: {e}")

        # Load cached signals
        sig_path = settings.signals_dir / "current_signals_v2.json"
        if sig_path.exists():
            with open(sig_path) as f:
                cls._signals = json.load(f)

        cls._initialized = True

    @classmethod
    def get_models(cls, ticker: str) -> dict:
        return cls._models.get(ticker, {})

    @classmethod
    def get_cached_signal(cls, ticker: str) -> Optional[dict]:
        return cls._signals.get("signals", {}).get(ticker)

    @classmethod
    def get_uptime(cls) -> float:
        return (datetime.now() - cls._startup_time).total_seconds()

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._initialized and len(cls._models) > 0
