# Demo Results

Sample outputs and performance metrics for the RAG prototype.

## Sample Input

Using `data/sample_contract.txt` (service agreement between Acme Corporation and Tech Solutions Inc.).

## Sample Extraction Output

```json
{
  "record": {
    "dates": "January 15, 2025; February 1, 2025; March 15, 2025; April 30, 2025; May 31, 2025",
    "parties": "Acme Corporation; Tech Solutions Inc.",
    "amounts": "$125,000 USD; $62,500; $31,250",
    "terms": "Software development services; payment schedule 50/50; milestone-based delivery",
    "summary": "Agreement for custom CRM development between Acme and Tech Solutions."
  },
  "uncertain_fields": [],
  "validation_errors": [],
  "needs_review": false,
  "review_reason": "ok"
}
```

## Sample Q&A Output

**Question:** "What is the total contract value?"

```json
{
  "answer": "The total contract value is $125,000 USD.",
  "confidence": 0.92,
  "needs_review": false,
  "review_reason": "ok",
  "sources": [...]
}
```

## Performance Metrics (Template)

| Metric | Value | Notes |
|--------|-------|-------|
| Ingest latency (1 doc) | ~5–15 s | Depends on embedding API |
| Extraction latency | ~2–5 s | Per run |
| Q&A latency | ~2–4 s | Per question |
| Chunk size | 512 chars | Configurable |
| Top-k retrieval | 4 | Configurable |
| Gate precision | TBD | Run EVALUATION_CHECKLIST |
| Gate recall | TBD | Run EVALUATION_CHECKLIST |

## MLFlow Metrics (when enabled)

With `MLFLOW_TRACKING_URI` set, each run logs:

- `needs_review` (0/1)
- `num_uncertain_fields`
- `confidence` (Q&A)
- `chunk_size`, `top_k`, `model` (params)

## Human Review Gate Behavior

| Scenario | needs_review | Reason |
|----------|---------------|--------|
| Low confidence (<0.7) | Yes | low_confidence |
| Few chunks retrieved | Yes | no_context |
| "I don't know" phrasing | Yes | uncertain_phrasing |
| Uncertain extraction fields | Yes | uncertain_fields |
| Validation errors | Yes | validation_errors |
| Clean result | No | ok |
