from __future__ import annotations

import json
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.pipeline import (
    RESEARCH_DIR,
    ROOT,
    apply_one_day_reliability_guard,
    build_one_day_edge_assessment,
    build_terminal_live_bundle,
    champion_latest_path_for,
    champion_manifest_path_for,
    derive_signal,
    latest_experimental_path_for,
    namespace_root,
    premarket_latest_path_for,
    ResearchConfig,
    apply_calibrator,
    apply_trend_prior,
    asof_join,
    build_feature_frame,
    compute_live_trust,
    fit_calibrator,
    is_in_premarket_window,
    news_weight_from_context,
    run_nightly_retrain,
    run_premarket_refresh,
    trend_state_from_score,
)


class ResearchPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ResearchConfig(
            min_train_size=120,
            validation_size=8,
            step_size=320,
            light_mode=True,
            publish_live_outputs=False,
            output_namespace="test_suite",
        )
        cls.nightly = run_nightly_retrain(cls.config)
        cls.override_contexts = {
            ticker: {
                "ticker": ticker,
                "forecast_for_date": cls.nightly["forecast_for_date"],
                "window_start_et": "2026-04-13T16:00:00-04:00",
                "window_end_et": "2026-04-14T08:15:00-04:00",
                "next_open_et": "2026-04-14T09:30:00-04:00",
                "analysis_mode": "heuristic",
                "article_count": 1,
                "material_count": 1,
                "overall_sentiment": "NEUTRAL",
                "intraday_bias": "FLAT",
                "net_score": 0.1 if ticker != "TSLA" else -0.4,
                "used_recent_fallback": ticker == "AAPL",
                "feature_values": {
                    "premkt_live_net": 0.1 if ticker != "TSLA" else -0.4,
                    "premkt_live_article_count": 1.0,
                    "premkt_live_material_count": 1.0,
                    "premkt_live_competition_risk": 0.0,
                    "premkt_live_contract_signal": 0.0,
                    "premkt_live_earnings_signal": 0.0,
                },
                "summary": f"Override context for {ticker}",
                "articles": [
                    {
                        "headline": f"{ticker} premarket test headline",
                        "description": "Synthetic test headline",
                        "source": "Unit Test",
                        "url": "",
                        "published": "Tue, 14 Apr 2026 08:00:00 -0400",
                        "published_at_et": "2026-04-14T08:00:00-04:00",
                        "in_window": True,
                        "sentiment": "NEUTRAL",
                        "impact": "MEDIUM",
                        "net_score": 0.1 if ticker != "TSLA" else -0.4,
                        "is_material": True,
                        "categories": ["premarket"],
                        "rationale": f"Override rationale for {ticker}",
                    }
                ],
            }
            for ticker in ("PLTR", "AAPL", "NVDA", "TSLA")
        }
        cls.premarket = run_premarket_refresh(cls.config, live_context_overrides=cls.override_contexts)
        cls.pages_bundle = build_terminal_live_bundle(cls.premarket, cls.config)

    def test_asof_join_never_uses_future_rows(self) -> None:
        left = pd.DataFrame({"date": pd.to_datetime(["2026-04-10", "2026-04-11", "2026-04-12"])})
        right = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-09", "2026-04-11", "2026-04-13"]),
                "value": [1, 2, 3],
            }
        )
        joined = asof_join(left, right)
        self.assertEqual(list(joined["value"]), [1, 2, 2])

    def test_premarket_window_respects_post_close_to_next_open(self) -> None:
        eastern = ZoneInfo("America/New_York")
        reference = datetime(2026, 4, 14, 8, 20, tzinfo=eastern)
        inside = datetime(2026, 4, 13, 16, 30, tzinfo=eastern)
        outside = datetime(2026, 4, 13, 15, 59, tzinfo=eastern)
        self.assertTrue(is_in_premarket_window(inside, reference_dt=reference))
        self.assertFalse(is_in_premarket_window(outside, reference_dt=reference))

    def test_fallback_news_is_downweighted(self) -> None:
        fresh = news_weight_from_context({"article_count": 2, "material_count": 1, "used_recent_fallback": False})
        fallback = news_weight_from_context({"article_count": 2, "material_count": 1, "used_recent_fallback": True})
        self.assertGreater(fresh, fallback)

    def test_calibration_is_monotonic(self) -> None:
        raw = np.array([0.1, 0.2, 0.25, 0.4, 0.6, 0.65, 0.8, 0.9])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        calibrator = fit_calibrator(raw, labels)
        calibrated = apply_calibrator(calibrator, np.array([0.15, 0.3, 0.5, 0.7, 0.85]))
        self.assertTrue(np.all(np.diff(calibrated) >= -1e-9))

    def test_trend_features_are_generated_without_future_leakage(self) -> None:
        frame = build_feature_frame("PLTR")
        self.assertIn("close_slope_5d", frame.columns)
        self.assertIn("trend_score", frame.columns)
        self.assertIn("premkt_hist_net_5d", frame.columns)
        self.assertIn("trend_state", frame.columns)
        self.assertTrue(((frame["trend_score"] >= -1.0) & (frame["trend_score"] <= 1.0)).all())
        self.assertIn(frame["trend_state"].iloc[-1], {"BULLISH", "BEARISH", "MIXED"})

    def test_trend_state_thresholds_are_stable(self) -> None:
        self.assertEqual(trend_state_from_score(0.40), "BULLISH")
        self.assertEqual(trend_state_from_score(-0.35), "BEARISH")
        self.assertEqual(trend_state_from_score(0.05), "MIXED")

    def test_trend_prior_and_trust_respect_alignment_and_conflict(self) -> None:
        latest_row = pd.Series(
            {
                "trend_score": 0.58,
                "drawdown_60": -0.04,
                "vol_10": 0.22,
                "days_to_next_earnings": 12,
            }
        )
        metrics = {
            "recent20_accuracy": 0.60,
            "recent60_accuracy": 0.58,
            "stability": 0.82,
            "trend_stability": 0.76,
            "ece": 0.08,
        }
        aligned_news = {"article_count": 1, "material_count": 1, "used_recent_fallback": False, "net_score": 0.35}
        conflicting_news = {"article_count": 1, "material_count": 0, "used_recent_fallback": False, "net_score": -0.22}
        fallback_conflict = {"article_count": 1, "material_count": 0, "used_recent_fallback": True, "net_score": -0.22}

        aligned_probability = apply_trend_prior(0.52, latest_row, aligned_news)
        conflicting_probability = apply_trend_prior(0.52, latest_row, conflicting_news)
        fallback_probability = apply_trend_prior(0.52, latest_row, fallback_conflict)
        aligned_trust = compute_live_trust(latest_row, aligned_probability, metrics, True, aligned_news)
        conflicting_trust = compute_live_trust(latest_row, conflicting_probability, metrics, True, conflicting_news)

        self.assertGreaterEqual(aligned_probability, conflicting_probability)
        self.assertGreater(aligned_trust, conflicting_trust)
        self.assertGreater(fallback_probability, 0.50)

    def test_one_day_reliability_guard_shrinks_weak_forecasts(self) -> None:
        weak_metrics = {
            "accuracy": 0.50,
            "recent20_accuracy": 0.40,
            "recent60_accuracy": 0.53,
            "recent20_brier": 0.252,
            "recent20_log_loss": 0.698,
        }
        strong_metrics = {
            "accuracy": 0.58,
            "recent20_accuracy": 0.65,
            "recent60_accuracy": 0.62,
            "recent20_brier": 0.214,
            "recent20_log_loss": 0.603,
        }
        fallback_context = {"used_recent_fallback": True}
        raw_probability = 0.70

        weak_adjusted = apply_one_day_reliability_guard(raw_probability, weak_metrics, 1, fallback_context)
        strong_adjusted = apply_one_day_reliability_guard(raw_probability, strong_metrics, 1, fallback_context)
        five_day_adjusted = apply_one_day_reliability_guard(raw_probability, weak_metrics, 5, fallback_context)

        self.assertLess(weak_adjusted, strong_adjusted)
        self.assertLess(weak_adjusted, raw_probability)
        self.assertEqual(five_day_adjusted, raw_probability)

    def test_one_day_signal_gate_gets_stricter_when_edge_is_weak(self) -> None:
        weak_metrics = {
            "accuracy": 0.50,
            "recent20_accuracy": 0.42,
            "recent60_accuracy": 0.52,
            "recent20_brier": 0.251,
            "recent20_log_loss": 0.695,
        }
        strong_metrics = {
            "accuracy": 0.58,
            "recent20_accuracy": 0.64,
            "recent60_accuracy": 0.62,
            "recent20_brier": 0.219,
            "recent20_log_loss": 0.612,
        }
        fallback_context = {"used_recent_fallback": True}

        weak_signal = derive_signal(0.61, 0.62, 1, weak_metrics, fallback_context)
        strong_signal = derive_signal(0.61, 0.62, 1, strong_metrics, None)

        self.assertEqual(weak_signal, "HOLD")
        self.assertEqual(strong_signal, "BUY")

    def test_one_day_edge_assessment_labels_low_edge_and_fallback(self) -> None:
        low_edge = build_one_day_edge_assessment(
            {
                "recent_performance": {
                    "accuracy": 0.50,
                    "recent20_accuracy": 0.40,
                    "recent60_accuracy": 0.53,
                    "recent20_brier": 0.252,
                    "recent20_log_loss": 0.698,
                },
                "supported_probability": True,
            },
            {"used_recent_fallback": False},
        )
        fallback = build_one_day_edge_assessment(
            {
                "recent_performance": {
                    "accuracy": 0.58,
                    "recent20_accuracy": 0.63,
                    "recent60_accuracy": 0.60,
                    "recent20_brier": 0.216,
                    "recent20_log_loss": 0.601,
                },
                "supported_probability": True,
            },
            {"used_recent_fallback": True},
        )
        calibrated = build_one_day_edge_assessment(
            {
                "recent_performance": {
                    "accuracy": 0.58,
                    "recent20_accuracy": 0.63,
                    "recent60_accuracy": 0.60,
                    "recent20_brier": 0.216,
                    "recent20_log_loss": 0.601,
                },
                "supported_probability": True,
            },
            {"used_recent_fallback": False},
        )
        self.assertEqual(low_edge["label"], "TACTICAL / LOW EDGE")
        self.assertEqual(fallback["label"], "TACTICAL / FALLBACK NEWS")
        self.assertEqual(calibrated["label"], "CALIBRATED 1D")

    def test_nightly_artifact_contains_all_tickers_and_horizons(self) -> None:
        tickers = self.nightly["tickers"]
        self.assertEqual(set(tickers.keys()), {"PLTR", "AAPL", "NVDA", "TSLA"})
        for ticker, payload in tickers.items():
            self.assertEqual(set(payload["horizons"].keys()), {"1d", "5d", "10d"})
            self.assertIn("trust_score", payload["signal"])
            self.assertEqual(payload["primary_horizon"], "1d")
            self.assertIn("trend_snapshot", payload)
            self.assertIn("one_day_edge", payload)
            self.assertIn("edge_label", payload["signal"])
            self.assertIn("edge_assessment", payload["horizons"]["1d"])
            self.assertIn("trend", payload["component_scores"])
            self.assertIn("one_day_edge", payload["component_scores"])
            self.assertIn("Daily trend structure", [driver["title"] for driver in payload["drivers"]])

    def test_backend_and_pages_artifacts_share_same_contract(self) -> None:
        nightly_keys = set(self.nightly.keys())
        premarket_keys = set(self.premarket.keys())
        self.assertEqual(nightly_keys, premarket_keys)
        for ticker in ("PLTR", "AAPL", "NVDA", "TSLA"):
            self.assertEqual(
                set(self.nightly["tickers"][ticker].keys()),
                set(self.premarket["tickers"][ticker].keys()),
            )

    def test_terminal_bundle_contains_live_pages_contract(self) -> None:
        bundle = self.pages_bundle
        self.assertEqual(bundle["artifact_status"], "champion")
        self.assertEqual(bundle["pages_publish_mode"]["policy"], "latest_available")
        self.assertEqual(set(bundle["tickers"].keys()), {"PLTR", "AAPL", "NVDA", "TSLA"})
        self.assertEqual(bundle["default_ticker"], "PLTR")
        self.assertIn("VIX", [item["display_symbol"] for item in bundle["macro_strip"]])
        for ticker, payload in bundle["tickers"].items():
            self.assertIn("quote_snapshot", payload)
            self.assertIn("chart", payload)
            self.assertIn("forecast_overlay", payload)
            self.assertIn("news_monitor", payload)
            self.assertIn("catalysts", payload)
            self.assertIn("levels", payload)
            self.assertIn("one_day_edge", payload)
            self.assertEqual(payload["market_date"], payload["quote_snapshot"]["market_date"])
            self.assertEqual(payload["chart"]["timeframes"]["all"][-1]["d"], payload["market_date"])
            self.assertIn("edge_label", payload["forecast_overlay"]["points"][0])

    def test_frontend_no_longer_fetches_stale_ohlcv_or_static_news_stamp(self) -> None:
        html = (ROOT / "docs" / "index.html").read_text()
        self.assertNotIn("fetch('ohlcv.json')", html)
        self.assertNotIn("Yahoo Finance · Apr 8 2026", html)
        self.assertIn("terminal_app.js", html)

    def test_premarket_refresh_keeps_model_versions(self) -> None:
        for ticker in ("PLTR", "AAPL", "NVDA", "TSLA"):
            nightly_versions = {
                horizon: self.nightly["tickers"][ticker]["horizons"][horizon]["model_version"]
                for horizon in ("1d", "5d", "10d")
            }
            refreshed_versions = {
                horizon: self.premarket["tickers"][ticker]["horizons"][horizon]["model_version"]
                for horizon in ("1d", "5d", "10d")
            }
            self.assertEqual(nightly_versions, refreshed_versions)
            self.assertEqual(
                self.premarket["tickers"][ticker]["news"]["used_recent_fallback"],
                self.override_contexts[ticker]["used_recent_fallback"],
            )
            self.assertIn("trend_snapshot", self.premarket["tickers"][ticker])
            self.assertIn("trend_supported", self.premarket["tickers"][ticker]["horizons"]["1d"])

    def test_run_registry_written(self) -> None:
        sandbox_root = namespace_root(self.config)
        self.assertTrue(sandbox_root.exists())
        self.assertTrue(champion_manifest_path_for(self.config).exists())
        self.assertTrue(champion_latest_path_for(self.config).exists())
        self.assertTrue(latest_experimental_path_for(self.config).exists())
        self.assertTrue(premarket_latest_path_for(self.config).exists())
        self.assertTrue(self.nightly["promotion"]["promoted"])


if __name__ == "__main__":
    unittest.main()
