#!/usr/bin/env python3
"""
run_signal.py — Quick signal runner called by OpenClaw skill.

Usage:
    python scripts/run_signal.py                    # run all tickers
    python scripts/run_signal.py --tickers NVDA PLTR  # run specific tickers
    python scripts/run_signal.py --format telegram  # output formatted for Telegram

OpenClaw reads stdout JSON and formats the alert.
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def load_cached_signals():
    """Load most recently generated signals."""
    path = ROOT / "trading_system" / "signals" / "current_signals.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        # Check freshness — if > 6 hours old, flag for re-run
        generated = datetime.fromisoformat(data.get("generated_at", "2000-01-01"))
        age_hours = (datetime.now() - generated).total_seconds() / 3600
        data["cache_age_hours"] = round(age_hours, 1)
        data["is_stale"] = age_hours > 6
        return data
    return None


def format_telegram_message(signals: dict) -> str:
    """Format signals as a Telegram-ready message."""
    lines = ["📊 *TRADING SIGNAL UPDATE*", f"_{datetime.now().strftime('%b %d, %Y %H:%M')} PDT_", ""]

    signal_emoji = {"STRONG BUY": "🚀", "BUY": "✅", "HOLD": "⏸", "SELL": "🔴"}
    risk_label = lambda r: "Low" if r < 3 else "Moderate" if r < 6 else "High" if r < 8 else "Very High"

    for ticker, sig in signals.get("signals", {}).items():
        emoji = signal_emoji.get(sig["signal"], "⚪")
        conf = sig["confidence"] * 100
        upside = sig["analyst_upside"]
        risk = sig["risk_score"]
        rsi = sig.get("rsi_14", 0)
        macd = sig.get("macd_hist", 0)
        macd_dir = "↑ Bullish" if macd > 0 else "↓ Bearish"

        lines.append(f"{emoji} *{ticker}* — {sig['signal']} ({conf:.0f}% confidence)")
        lines.append(f"   💰 ${sig['current_price']:.2f} → Target ${sig['analyst_target']:.0f} (*{upside:+.1f}%*)")
        lines.append(f"   📊 RSI: {rsi:.1f} | MACD: {macd_dir} | Risk: {risk_label(risk)} ({risk:.1f}/10)")
        lines.append(f"   👥 Analyst consensus: {sig['bull_pct']:.0f}% bullish")

        # Top ML features
        top_feats = [f[0] for f in sig.get("top_features", [])[:3]]
        if top_feats:
            lines.append(f"   🧠 Top signals: `{'`, `'.join(top_feats)}`")
        lines.append("")

    # Backtest reminder
    lines.append("─────────────────────────")
    lines.append("_⚠️ Not financial advice. Always do your own research._")

    return "\n".join(lines)


def format_discord_message(signals: dict) -> str:
    """Format signals as Discord embed-ready."""
    lines = ["**📊 TRADING SIGNAL UPDATE**",
             f"> Generated: {datetime.now().strftime('%b %d %Y %H:%M')} PDT", ""]

    for ticker, sig in signals.get("signals", {}).items():
        conf = sig["confidence"] * 100
        lines.append(f"**{ticker}** — `{sig['signal']}` ({conf:.0f}% conf)")
        lines.append(f"> Price: **${sig['current_price']:.2f}** → Target **${sig['analyst_target']:.0f}** ({sig['analyst_upside']:+.1f}%)")
        lines.append(f"> RSI {sig.get('rsi_14', 0):.1f} | Risk {sig['risk_score']:.1f}/10 | {sig['bull_pct']:.0f}% bulls")
        lines.append("")

    lines.append("*Not financial advice.*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run ML trading signals")
    parser.add_argument("--tickers", nargs="+", default=["PLTR", "AAPL", "NVDA", "TSLA"])
    parser.add_argument("--format", choices=["json", "telegram", "discord", "plain"], default="json")
    parser.add_argument("--refresh", action="store_true", help="Force re-run ML model")
    args = parser.parse_args()

    # Load cached or re-run
    data = load_cached_signals()

    if data is None or args.refresh or data.get("is_stale"):
        print("⟳ Signals stale or missing — re-running ML engine...", file=sys.stderr)
        result = subprocess.run(
            [sys.executable, str(ROOT / "ml_trading_system.py")],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        if result.returncode != 0:
            print(f"ERROR: ML engine failed\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
        data = load_cached_signals()

    if data is None:
        print("ERROR: Could not load signals", file=sys.stderr)
        sys.exit(1)

    # Filter to requested tickers
    if args.tickers:
        data["signals"] = {k: v for k, v in data.get("signals", {}).items() if k in args.tickers}

    # Output
    if args.format == "json":
        print(json.dumps(data, indent=2, default=str))
    elif args.format == "telegram":
        print(format_telegram_message(data))
    elif args.format == "discord":
        print(format_discord_message(data))
    else:
        # Plain text
        for ticker, sig in data.get("signals", {}).items():
            conf = sig["confidence"] * 100
            print(f"{ticker}: {sig['signal']} ({conf:.0f}% confidence) | "
                  f"${sig['current_price']:.2f} → ${sig['analyst_target']:.0f} "
                  f"({sig['analyst_upside']:+.1f}%)")


if __name__ == "__main__":
    main()
