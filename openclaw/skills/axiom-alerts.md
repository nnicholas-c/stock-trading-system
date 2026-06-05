# AXIOM Research Audit - OpenClaw Note

This legacy OpenClaw alert skill has been disabled during the honest rewrite.

The old version presented experimental model signals as scheduled trading intelligence. That framing is no longer valid after the leakage audit. Do not use this skill for live trade alerts or performance claims.

## Current Status

- The repository is now a signal-research and backtest-audit project.
- The validated headline is the fixed-specification harness in `honest_backtest/`.
- Legacy model scripts live in `experiments/` and are retained as research history only.

## Safer Use

If OpenClaw integration is restored later, it should send research-status notifications only:

- latest honest backtest summary location,
- audit warnings,
- data refresh status,
- test/CI status,
- explicit "not financial advice" language.

It should not send BUY/SELL alerts unless a new audited live-signal process is built and documented.
