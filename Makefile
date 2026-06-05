PYTHON := .venv/bin/python
PIP := .venv/bin/pip
UVICORN := .venv/bin/uvicorn

.PHONY: setup run-backend run-dashboard smoke

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt -r backend/requirements.txt

run-backend:
	cd backend && ../.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

run-dashboard:
	$(PYTHON) -m http.server 8080 -d docs

smoke:
	curl -fsS http://127.0.0.1:8000/health/
	curl -fsS http://127.0.0.1:8000/signals/PLTR
	curl -fsS http://127.0.0.1:8000/predict/PLTR/weekly
