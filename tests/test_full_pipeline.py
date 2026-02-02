#!/usr/bin/env python3
"""Full pipeline test: ingest, Q&A, extraction."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()


def test_full_pipeline():
    """Run ingest → Q&A → extraction and assert results."""
    from langchain_openai import ChatOpenAI

    from src.extraction import DefaultExtractionSchema, run_extraction
    from src.ingest import ingest_documents
    from src.qa import run_qa

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY not set. Add to .env and retry.")

    data_dir = Path(__file__).resolve().parent.parent / "data"
    sample_file = data_dir / "sample_contract.txt"
    if not sample_file.exists():
        raise SystemExit(f"Sample file not found: {sample_file}")

    print("=" * 60)
    print("1. INGEST")
    print("=" * 60)

    def on_progress(msg: str, p: float) -> None:
        print(f"  [{p*100:.0f}%] {msg}")

    vector_store = ingest_documents(
        [str(sample_file)],
        progress_callback=on_progress,
    )
    print("  OK: Ingest complete\n")

    print("=" * 60)
    print("2. Q&A")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_result = run_qa("What is the total contract value?", vector_store, llm)

    print(f"  Question: What is the total contract value?")
    print(f"  Answer: {qa_result['answer'][:200]}...")
    print(f"  Confidence: {qa_result.get('confidence', 'N/A')}")
    print(f"  Needs review: {qa_result['needs_review']} ({qa_result['review_reason']})")
    print(f"  Source chunks: {len(qa_result['source_chunks'])}")
    assert qa_result["answer"], "Q&A should return an answer"
    print("  OK: Q&A complete\n")

    print("=" * 60)
    print("3. EXTRACTION")
    print("=" * 60)

    ext_result = run_extraction(vector_store, DefaultExtractionSchema, llm)

    print(f"  Record: {ext_result['record']}")
    print(f"  Uncertain fields: {ext_result.get('uncertain_fields', [])}")
    print(f"  Validation errors: {ext_result.get('validation_errors', [])}")
    print(f"  Needs review: {ext_result['needs_review']} ({ext_result['review_reason']})")
    assert "record" in ext_result, "Extraction should return a record"
    print("  OK: Extraction complete\n")

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()
