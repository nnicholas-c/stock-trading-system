#!/usr/bin/env python3
"""
Refresh daily market CSVs from Yahoo Finance for the DRL pipeline.
Updates the repo's local `data/daily` and `data/macro` files in place.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "data" / "daily"
MACRO_DIR = ROOT / "data" / "macro"

EQUITIES = ["PLTR", "AAPL", "NVDA", "TSLA"]
MACRO = ["SPY", "QQQ", "TLT", "GLD"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.reset_index()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            str(level_0).lower() if level_0 else str(level_1).lower()
            for level_0, level_1 in frame.columns.to_flat_index()
        ]
    else:
        frame = frame.rename(columns=str.lower)
    keep = ["date", "open", "high", "low", "close", "volume"]
    frame = frame[keep].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    frame["volume"] = frame["volume"].fillna(0).astype(float)
    return frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")


def refresh_symbol(symbol: str, out_dir: Path, end_date: date) -> tuple[str, str]:
    path = out_dir / f"{symbol}_daily.csv"
    if path.exists():
        existing = pd.read_csv(path)
        start = pd.to_datetime(existing["date"]).max().date() - timedelta(days=7)
    else:
        existing = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        start = date(2022, 1, 1)

    downloaded = yf.download(
        symbol,
        start=start.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if downloaded.empty:
        return symbol, "no_new_data"

    latest = normalize(downloaded)
    merged = pd.concat([existing, latest], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    merged.to_csv(path, index=False)
    return symbol, merged["date"].iloc[-1]


def main():
    parser = argparse.ArgumentParser(description="Refresh local market CSVs")
    parser.add_argument(
        "--as-of",
        default=date.today().isoformat(),
        help="Target date. For weekend runs, use the current date and the script will fetch the latest available close.",
    )
    args = parser.parse_args()

    end_date = date.fromisoformat(args.as_of)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    MACRO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Refreshing market data through {end_date.isoformat()}...")
    for symbol in EQUITIES:
        sym, last_date = refresh_symbol(symbol, DAILY_DIR, end_date)
        print(f"  {sym}: {last_date}")
    for symbol in MACRO:
        sym, last_date = refresh_symbol(symbol, MACRO_DIR, end_date)
        print(f"  {sym}: {last_date}")


if __name__ == "__main__":
    main()
