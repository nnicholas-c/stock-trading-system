#!/usr/bin/env python3
"""Run the research-grade nightly retrain pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.pipeline import run_nightly_retrain


if __name__ == "__main__":
    print(json.dumps(run_nightly_retrain(), indent=2))
