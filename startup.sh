#!/bin/bash
set -e

echo "--- Installing pip deps ---"
pip install --no-cache-dir -r requirements-f1.txt 2>&1 | tail -3

echo "--- Starting server ---"
exec python -m uvicorn api:app --app-dir code --host 0.0.0.0 --port 8000
