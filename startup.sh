#!/bin/bash
set -e

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Starting FastAPI server ==="
exec uvicorn code.api:app --host 0.0.0.0 --port "${PORT:-8000}"
