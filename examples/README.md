# Test Examples

## Quick test (curl)

**Local (API running on port 8000):**
```bash
cd ..
./examples/test_api.sh
# or
./examples/test_api.sh http://localhost:8000
```

**Azure:**
```bash
./examples/test_api.sh https://rag-prototype-7be4dfe5.azurewebsites.net
```

**With custom sample file:**
```bash
./examples/test_api.sh https://rag-prototype-7be4dfe5.azurewebsites.net /path/to/contract.pdf
```

## Python test (requests)

```bash
pip install requests
python examples/test_api.py
# or
python examples/test_api.py https://rag-prototype-7be4dfe5.azurewebsites.net
```

## Manual curl commands

```bash
# Set base URL
BASE_URL="https://rag-prototype-7be4dfe5.azurewebsites.net"
# or for local: BASE_URL="http://localhost:8000"

# 1. Health
curl $BASE_URL/health

# 2. Ingest
curl -X POST $BASE_URL/ingest -F "files=@data/sample_contract.txt"

# 3. Extract
curl -X POST $BASE_URL/extract -H "Content-Type: application/json" -d '{}'

# 4. Q&A
curl -X POST $BASE_URL/qa -H "Content-Type: application/json" \
  -d '{"question": "What is the total contract value?"}'
```
