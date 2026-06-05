#!/usr/bin/env python3
"""Build the canonical premarket forecast artifact for backend and Pages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.pipeline import ResearchConfig, run_premarket_refresh


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the shared premarket forecast artifact.")
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
    artifact = run_premarket_refresh(config, reference_dt=args.reference_datetime)
    print(json.dumps({"forecast_for_date": artifact["forecast_for_date"], "market_date": artifact["market_date"]}, indent=2))


if __name__ == "__main__":
    main()
