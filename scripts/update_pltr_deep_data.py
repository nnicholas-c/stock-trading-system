#!/usr/bin/env python3
"""
Refresh the PLTR deep-model local datasets from Yahoo Finance.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "trading_system" / "pltr_deep"


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.reset_index()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            str(level_0).lower() if level_0 else str(level_1).lower()
            for level_0, level_1 in frame.columns.to_flat_index()
        ]
    else:
        frame = frame.rename(columns=str.lower)
    return frame


def download(symbol: str, end_date: date) -> pd.DataFrame:
    data = yf.download(
        symbol,
        start="2023-01-01",
        end=(end_date + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    return flatten_columns(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh PLTR deep data files")
    parser.add_argument(
        "--as-of",
        default=date.today().isoformat(),
        help="Fetch data through this date",
    )
    args = parser.parse_args()

    end_date = date.fromisoformat(args.as_of)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pltr = download("PLTR", end_date)[["date", "open", "high", "low", "close", "volume"]].copy()
    pltr["date"] = pd.to_datetime(pltr["date"]).dt.strftime("%Y-%m-%d")
    pltr.to_csv(OUT_DIR / "pltr_ohlcv.csv", index=False)

    for symbol, filename in [("SPY", "spy_close.csv"), ("QQQ", "qqq_close.csv")]:
        frame = download(symbol, end_date)[["date", "close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        frame.to_csv(OUT_DIR / filename, index=False)

    print(f"Refreshed PLTR deep datasets through {end_date.isoformat()}")


if __name__ == "__main__":
    main()
