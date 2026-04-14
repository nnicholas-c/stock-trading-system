#!/usr/bin/env python3
"""
Refresh local market data and rebuild the live DRL signals sequentially.

This wrapper exists because multi-ticker SB3 runs are unstable on this local
macOS/PyTorch stack. Running one ticker at a time has been reliable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python"
UPDATE_DATA = ROOT / "scripts" / "update_market_data.py"
TRAIN_DRL = ROOT / "train_drl_v2.py"
RESULTS = ROOT / "trading_system" / "drl" / "drl_v2_results.json"
DEFAULT_TICKERS = ["PLTR", "NVDA", "AAPL", "TSLA"]


def run_command(cmd: list[str]) -> None:
    env = dict(os.environ)
    env["PYTHONFAULTHANDLER"] = "1"
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh live DRL signals locally")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Tickers to refresh sequentially",
    )
    parser.add_argument(
        "--as-of",
        default=date.today().isoformat(),
        help="Refresh data through this date before training",
    )
    parser.add_argument(
        "--schedule",
        default="100",
        help="Comma-separated PPO timestep schedule for the per-ticker refresh",
    )
    parser.add_argument(
        "--oracle-trees",
        type=int,
        default=40,
        help="Estimator count for the live oracle fits",
    )
    parser.add_argument(
        "--oracle-jobs",
        type=int,
        default=1,
        help="Thread count for the live oracle fits",
    )
    parser.add_argument(
        "--skip-data-refresh",
        action="store_true",
        help="Skip refreshing the Yahoo Finance CSVs first",
    )
    args = parser.parse_args()

    tickers = [ticker.upper() for ticker in args.tickers]

    if not args.skip_data_refresh:
        run_command([str(PYTHON), str(UPDATE_DATA), "--as-of", args.as_of])

    base = None
    combined_backtest = {}
    combined_signals = {}

    for ticker in tickers:
        run_command(
            [
                str(PYTHON),
                str(TRAIN_DRL),
                "--tickers",
                ticker,
                "--schedule",
                args.schedule,
                "--max-iters",
                "1",
                "--fast-live",
                "--oracle-trees",
                str(args.oracle_trees),
                "--oracle-jobs",
                str(args.oracle_jobs),
            ]
        )
        data = json.loads(RESULTS.read_text())
        base = data
        combined_backtest[ticker] = data["backtest"][ticker]
        combined_signals[ticker] = data["signals"][ticker]

    if base is None:
        raise SystemExit("No DRL output was generated.")

    latest_signal_date = max((signal.get("date", "") for signal in combined_signals.values()), default="")
    combined = dict(base)
    combined["generated"] = datetime.now().isoformat()
    combined["tickers"] = tickers
    combined["latest_signal_date"] = latest_signal_date
    combined["backtest"] = combined_backtest
    combined["signals"] = combined_signals
    RESULTS.write_text(json.dumps(combined, indent=2))

    print(json.dumps({"latest_signal_date": latest_signal_date, "signals": combined_signals}, indent=2))


if __name__ == "__main__":
    main()
