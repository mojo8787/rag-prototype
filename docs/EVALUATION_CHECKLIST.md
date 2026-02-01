# Evaluation Checklist for RAG Prototype

Use this checklist to judge prototype quality: retrieval, answer faithfulness, extraction accuracy, and human-review gate behavior.

## 1. Retrieval

| # | Criterion | Yes / No | Notes |
|---|-----------|----------|--------|
| 1 | Retrieved chunks are relevant to the query | | |
| 2 | Top-k is sufficient (not too few / too many chunks) | | |
| 3 | Chunk boundaries don’t cut critical information mid-sentence | | |
| 4 | Source metadata (e.g. file name, chunk index) is correct | | |

## 2. Document Q&A

| # | Criterion | Yes / No | Notes |
|---|-----------|----------|--------|
| 5 | Answer is grounded in the retrieved context | | |
| 6 | Answer does not hallucinate information outside context | | |
| 7 | Citations (e.g. “Chunk 1”) match the actual sources | | |
| 8 | Confidence score (when present) is plausible | | |
| 9 | For out-of-scope questions, model says “I don’t know” or equivalent | | |

## 3. Document Extraction

| # | Criterion | Yes / No | Notes |
|---|-----------|----------|--------|
| 10 | Extracted fields match the document content | | |
| 11 | Data types and formats are correct (e.g. dates, amounts) | | |
| 12 | Uncertain fields are flagged when appropriate | | |
| 13 | Validation errors are accurate and actionable | | |

## 4. Human-Review Gates

| # | Criterion | Yes / No | Notes |
|---|-----------|----------|--------|
| 14 | Q&A: “Needs human review” triggers when confidence is low | | |
| 15 | Q&A: “Needs human review” triggers when no/few chunks retrieved | | |
| 16 | Q&A: “Needs human review” triggers on “I don’t know” style answers | | |
| 17 | Extraction: “Needs human review” triggers on uncertain fields | | |
| 18 | Extraction: “Needs human review” triggers on validation errors | | |
| 19 | Review reason (e.g. low_confidence, no_context) is correct | | |
| 20 | Gate does not over-flag (precision) or under-flag (recall) | | |

## 5. End-to-End

| # | Criterion | Yes / No | Notes |
|---|-----------|----------|--------|
| 21 | Ingest → Q&A → “Needs human review” flow works | | |
| 22 | Ingest → Extraction → “Needs human review” flow works | | |
| 23 | UI shows answer/sources and review status clearly | | |
| 24 | Config (chunk size, top-k, thresholds) is easy to change | | |

## How to use

1. Run the app, ingest one or more sample documents.
2. Run several Q&A and extraction examples (in-scope and out-of-scope).
3. For each criterion, mark Yes/No and add short notes.
4. Optionally run canned examples (if added) and record pass/fail and gate outcomes for manual review.

## Optional: Gate precision/recall

- **Precision**: Of all items marked “Needs human review”, how many truly needed review? (Avoid over-flagging.)
- **Recall**: Of all items that truly needed review, how many were marked “Needs human review”? (Avoid under-flagging.)
- Tune thresholds (e.g. `GATE_CONFIDENCE_THRESHOLD`, `GATE_MIN_CHUNKS`) based on this checklist and your use case.
