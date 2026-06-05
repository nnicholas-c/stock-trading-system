#!/usr/bin/env python3
"""Thin runner for the shared research pipeline with multi-ticker output."""

from __future__ import annotations

import argparse
import json

from research.pipeline import ResearchConfig, run_nightly_retrain, run_premarket_refresh


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the shared research pipeline and print the cross-ticker 1d view.")
    parser.add_argument("--premarket", action="store_true", help="Refresh live premarket inference without retraining weights.")
    parser.add_argument("--fast", action="store_true", help="Use a lighter walk-forward profile for local iteration.")
    parser.add_argument(
        "--reference-datetime",
        help="Pin premarket inference to an ISO timestamp or YYYY-MM-DD date, for example 2026-04-14T08:15:00-04:00.",
    )
    args = parser.parse_args()

    config = ResearchConfig(
        min_train_size=180 if args.fast else 252,
        validation_size=12 if args.fast else 20,
        step_size=90 if args.fast else 20,
        light_mode=args.fast,
    )
    artifact = (
        run_premarket_refresh(config, reference_dt=args.reference_datetime)
        if args.premarket
        else run_nightly_retrain(config)
    )
    summary = {
        "generated_at": artifact["generated_at"],
        "run_type": artifact["run_type"],
        "forecast_for_date": artifact["forecast_for_date"],
        "signals": {ticker: payload["signal"] for ticker, payload in artifact["tickers"].items()},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
