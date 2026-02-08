#!/bin/bash
# Test script for RAG API (local or Azure)
# Usage: ./test_api.sh [BASE_URL]
# Example: ./test_api.sh
# Example: ./test_api.sh https://rag-prototype-7be4dfe5.azurewebsites.net

set -e

BASE_URL="${1:-http://localhost:8000}"
SAMPLE_FILE="${2:-data/sample_contract.txt}"

echo "=========================================="
echo "RAG API Test"
echo "Base URL: $BASE_URL"
echo "Sample file: $SAMPLE_FILE"
echo "=========================================="

# 1. Health check
echo ""
echo "[1/4] Health check..."
curl -s "$BASE_URL/health" | jq . 2>/dev/null || curl -s "$BASE_URL/health"
echo ""

# 2. Ingest
echo ""
echo "[2/4] Ingesting sample contract..."
if [ ! -f "$SAMPLE_FILE" ]; then
  echo "ERROR: Sample file not found: $SAMPLE_FILE"
  echo "Run from project root or provide full path."
  exit 1
fi
INGEST_RESPONSE=$(curl -s -X POST "$BASE_URL/ingest" -F "files=@$SAMPLE_FILE")
echo "$INGEST_RESPONSE" | jq . 2>/dev/null || echo "$INGEST_RESPONSE"
echo ""

# 3. Extract
echo ""
echo "[3/4] Extracting structured data..."
EXTRACT_RESPONSE=$(curl -s -X POST "$BASE_URL/extract" -H "Content-Type: application/json" -d '{}')
echo "$EXTRACT_RESPONSE" | jq . 2>/dev/null || echo "$EXTRACT_RESPONSE"
echo ""

# 4. Q&A
echo ""
echo "[4/4] Q&A: What is the total contract value?"
QA_RESPONSE=$(curl -s -X POST "$BASE_URL/qa" -H "Content-Type: application/json" -d '{"question": "What is the total contract value?"}')
echo "$QA_RESPONSE" | jq . 2>/dev/null || echo "$QA_RESPONSE"
echo ""

echo "=========================================="
echo "Test complete."
echo "=========================================="
