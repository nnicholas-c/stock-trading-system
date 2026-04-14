#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "trading_system" / "pltr_deep" / "pltr_prediction_audit.json"
sys.path.insert(0, str(ROOT))

import train_pltr_deep as model


def build_audit(tail_rows: int = 12) -> dict:
    prices = model.build_price_matrix()
    features = model.build_daily_features(prices)
    pltr_close = prices["PLTR"]

    audit: dict[str, dict] = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "latest_market_date": pltr_close.index[-1].strftime("%Y-%m-%d"),
        "horizons": {},
    }

    for horizon in (1, 5, 10):
        wf = model.PLTRWalkForward(horizon=horizon)
        metrics = wf.run(features, pltr_close)
        df = pd.DataFrame(wf.results)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        df["target_date"] = df["date"].apply(
            lambda d: pltr_close.index[pltr_close.index.get_loc(d) + horizon].strftime("%Y-%m-%d")
        )
        df["pred_label"] = df["pred_dir"].map({1: "UP", 0: "DOWN"})
        df["actual_label"] = df["actual_dir"].map({1: "UP", 0: "DOWN"})
        df["prob_up_pct"] = (df["prob_up"] * 100).round(1)
        df["conf_pct"] = (df["confidence"] * 100).round(1)
        df["pred_ret_pct"] = (df["pred_ret"] * 100).round(2)
        df["actual_ret_pct"] = (df["actual_ret"] * 100).round(2)

        recent = df.tail(tail_rows).copy()
        audit["horizons"][f"{horizon}d"] = {
            "metrics": metrics,
            "recent_accuracy_last_5": float(df.tail(5)["correct"].mean()),
            "recent_accuracy_last_10": float(df.tail(10)["correct"].mean()),
            "recent_accuracy_last_20": float(df.tail(20)["correct"].mean()),
            "recent_predictions": recent[
                [
                    "date",
                    "target_date",
                    "price",
                    "pred_label",
                    "actual_label",
                    "prob_up_pct",
                    "conf_pct",
                    "pred_ret_pct",
                    "actual_ret_pct",
                    "correct",
                ]
            ].to_dict(orient="records"),
        }

    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit matured PLTR deep-model predictions.")
    parser.add_argument("--tail", type=int, default=12, help="Number of recent matured calls to keep per horizon.")
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="Where to write the JSON report.")
    args = parser.parse_args()

    audit = build_audit(tail_rows=args.tail)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(audit, handle, indent=2, default=str)

    print(f"Wrote audit to {args.out}")
    for horizon, payload in audit["horizons"].items():
        print(
            f"{horizon}: overall={payload['metrics']['overall_accuracy']*100:.1f}% | "
            f"last10={payload['recent_accuracy_last_10']*100:.1f}% | "
            f"last5={payload['recent_accuracy_last_5']*100:.1f}%"
        )


if __name__ == "__main__":
    main()
