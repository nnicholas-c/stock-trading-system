# SECURITY FINDINGS

Scan date: 2026-06-05

## Tools Available

- `gitleaks`: not installed in this environment.
- `trufflehog`: not installed in this environment.
- Fallback scans used `rg`, `git grep` over `git rev-list --all`, and `git log --all --name-only`.

## Commands Run

```bash
command -v gitleaks || true
command -v trufflehog || true
git log --all --name-only --pretty=format: | rg '(^|/)\.env$' || true
rg -n -I '(sk-[A-Za-z0-9_-]{20,}|OPENAI_API_KEY\s*=\s*[^\s#]+|API_KEY\s*=\s*[^\s#]+|SECRET\s*=\s*[^\s#]+|TOKEN\s*=\s*[^\s#]+|AXIOM_OPENAI_API_KEY)' .
git grep -n -I -E '(sk-[A-Za-z0-9_-]{20,}|OPENAI_API_KEY\s*=\s*[^[:space:]#]+|API_KEY\s*=\s*[^[:space:]#]+|SECRET\s*=\s*[^[:space:]#]+|TOKEN\s*=\s*[^[:space:]#]+|AXIOM_OPENAI_API_KEY)' $(git rev-list --all)
rg -n -I '[A-Za-z0-9+/]{48,}={0,2}' --glob '!.venv/**' --glob '!data/**' --glob '!docs/*.json' --glob '!bloomberg/*.json' .
```

## Findings

No live-looking API keys, bearer tokens, private keys, or `.env` files were found by the available scans.

Observed matches were placeholders or false positives:

- `backend/.env.example` documents `AXIOM_OPENAI_API_KEY` but does not contain a value.
- `backend/app/core/config.py` accepts `AXIOM_OPENAI_API_KEY` and `OPENAI_API_KEY` as environment variable names.
- `ios/AXIOM/Services/APIService.swift` and `ios/README-iOS.md` now default to `http://localhost:8000`; no deployed API placeholder or credential is embedded.
- Historical `docs/index.html` contained a Yahoo Finance URL with the substring `sk-...`; this is not an OpenAI key.
- High-entropy checks produced Google News/RSS URLs and other public encoded URL fragments, not credentials.

## Repository Hygiene Risks

These are not secret leaks, but they are worth cleaning up in a future artifact-purge branch:

- Tracked model binaries exist under `trading_system/models/`, `trading_system/models/v3/`, and `trading_system/research/champion/`.
- Tracked generated data/artifacts exist under `data/`, `docs/*.json`, `bloomberg/*.json`, and `trading_system/research/runs/`.

This branch updates `.gitignore` so new local data, binary model artifacts, and caches are ignored going forward. Already tracked files remain tracked until explicitly removed with `git rm --cached` and a history/artifact strategy.

## Recommendations

- Install and run `gitleaks detect --source . --no-git` and `gitleaks detect --source .` before merging.
- Add a secret-scan CI job.
- Rotate any credential that was ever copied into a local `.env`, even though no committed `.env` was found here.
- Move large generated artifacts to release assets, object storage, or Git LFS after deciding which historical outputs should remain reproducible.
