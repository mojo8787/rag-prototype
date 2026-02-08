#!/bin/bash
# Deploy RAG prototype to Azure App Service
# IMPORTANT: Wait 5 min after any config/restart before deploying.
set -e

RG="rg-rag-west"
APP="rag-prototype-7be4dfe5"

echo "Deploying to $APP..."
az webapp up --resource-group $RG --name $APP --runtime "PYTHON:3.11" --plan plan-rag

echo ""
echo "Deploy complete. Wait 2–3 min for app to start, then test:"
echo "  curl https://$APP.azurewebsites.net/health"
echo "  curl -X POST https://$APP.azurewebsites.net/ingest -F 'files=@data/sample_contract.txt'"
