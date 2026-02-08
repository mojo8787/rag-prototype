#!/bin/bash
# Azure App Service: use PORT from environment (default 8000)
PORT=${PORT:-8000}
exec python -m uvicorn api:app --host 0.0.0.0 --port $PORT
