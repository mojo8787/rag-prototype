#!/bin/bash
# Azure App Service: use PORT from environment (default 8000)
PORT=${PORT:-8000}
exec gunicorn -w 2 -k uvicorn.workers.UvicornWorker api:app --bind 0.0.0.0:$PORT
