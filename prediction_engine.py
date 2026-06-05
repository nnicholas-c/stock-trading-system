#!/usr/bin/env python3
"""Unified forecast entrypoint for nightly retrains and premarket refreshes."""

from __future__ import annotations

import argparse
import json

from research.pipeline import ResearchConfig, run_nightly_retrain, run_premarket_refresh


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AXIOM's shared research forecast engine.")
    parser.add_argument(
        "--mode",
        choices=("nightly", "premarket"),
        default="nightly",
        help="Nightly retrain or premarket inference refresh.",
    )
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
        if args.mode == "premarket"
        else run_nightly_retrain(config)
    )
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
