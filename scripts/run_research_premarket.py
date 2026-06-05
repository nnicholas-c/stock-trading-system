#!/usr/bin/env python3
"""Run the research-grade premarket inference refresh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.pipeline import run_premarket_refresh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the shared research premarket refresh.")
    parser.add_argument(
        "--reference-datetime",
        help="Pin premarket inference to an ISO timestamp or YYYY-MM-DD date, for example 2026-04-14T08:15:00-04:00.",
    )
    args = parser.parse_args()
    print(json.dumps(run_premarket_refresh(reference_dt=args.reference_datetime), indent=2))
