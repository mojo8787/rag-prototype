#!/usr/bin/env python3
"""Python test script for RAG API (local or Azure)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

BASE_URL = "http://localhost:8000"
SAMPLE_FILE = Path(__file__).parent.parent / "data" / "sample_contract.txt"


def main(base_url: str = BASE_URL) -> None:
    base_url = base_url.rstrip("/")
    print("=" * 50)
    print("RAG API Test")
    print(f"Base URL: {base_url}")
    print("=" * 50)

    # 1. Health
    print("\n[1/4] Health check...")
    r = requests.get(f"{base_url}/health")
    print(json.dumps(r.json(), indent=2))

    # 2. Ingest
    print("\n[2/4] Ingesting sample contract...")
    if not SAMPLE_FILE.exists():
        print(f"ERROR: {SAMPLE_FILE} not found")
        sys.exit(1)
    with open(SAMPLE_FILE, "rb") as f:
        r = requests.post(f"{base_url}/ingest", files={"files": ("sample_contract.txt", f)})
    print(json.dumps(r.json(), indent=2))

    # 3. Extract
    print("\n[3/4] Extracting structured data...")
    r = requests.post(f"{base_url}/extract", json={})
    data = r.json()
    print("Record:", json.dumps(data.get("record", {}), indent=2))
    print("Needs review:", data.get("needs_review"))

    # 4. Q&A
    print("\n[4/4] Q&A: What is the total contract value?")
    r = requests.post(f"{base_url}/qa", json={"question": "What is the total contract value?"})
    data = r.json()
    print("Answer:", data.get("answer", "")[:200], "...")
    print("Confidence:", data.get("confidence"))
    print("Needs review:", data.get("needs_review"))

    print("\n" + "=" * 50)
    print("Test complete.")
    print("=" * 50)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else BASE_URL
    main(url)
