# LLM / RAG Prototype

A small RAG prototype with document Q&A and extraction workflows, configurable chunking, evaluation checklists, and "needs human review" decision gates.

## What’s included

- **Document Q&A**: Ask questions over ingested documents; answers are grounded in retrieved chunks with optional confidence and source citations. A gate flags results that need human review (e.g. low confidence, no context).
- **Document extraction**: Extract structured fields (e.g. dates, parties, amounts) from documents. A gate flags results with uncertain fields or validation errors.
- **Chunking**: Fixed-size-with-overlap and by-paragraph strategies; see [docs/CHUNKING.md](docs/CHUNKING.md).
- **Evaluation checklist**: Criteria for judging retrieval, Q&A, extraction, and gates; see [docs/EVALUATION_CHECKLIST.md](docs/EVALUATION_CHECKLIST.md).

## Setup

**Python**: Use **3.11 or 3.12** for a quick, reliable install. Python 3.14 can cause very long `pip install` times (dependency backtracking); if install hangs, use 3.11/3.12 instead.

1. **Clone / open** this repo and create a virtualenv (recommended):

   ```bash
   python3.12 -m venv .venv   # or python3.11
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment variables** (copy `.env.example` to `.env` and fill in):

   - `OPENAI_API_KEY` — required for embeddings and LLM (e.g. `sk-...`).

   Optional overrides (defaults in [src/config.py](src/config.py)):

   - `CHUNK_SIZE`, `CHUNK_OVERLAP` — chunking
   - `RETRIEVAL_TOP_K` — number of chunks retrieved
   - `GATE_CONFIDENCE_THRESHOLD`, `GATE_MIN_CHUNKS` — human-review gate thresholds

## Run the app

```bash
streamlit run app.py
# or
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
2. Click **Create app** and fill in:
   - **Repository**: `mojo8787/rag-prototype` (or your fork)
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
3. Open **Advanced settings** and add your secrets (paste as TOML):
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
4. Click **Deploy**.

Then:

1. **Ingest**: Upload one or more `.txt` or `.pdf` files and click **Ingest**.
2. **Q&A**: Enter a question and click **Run Q&A**. View answer, sources, and **Needs human review** (Yes/No + reason).
3. **Extraction**: Click **Run extraction** to extract the default schema (dates, parties, amounts, summary). View the record and **Needs human review** (Yes/No + reason).

## Project layout

| Path | Purpose |
|------|--------|
| [app.py](app.py) | Streamlit UI (ingest, Q&A, extraction) |
| [src/config.py](src/config.py) | Chunk size, top-k, gate thresholds |
| [src/chunking.py](src/chunking.py) | Chunking strategies |
| [src/ingest.py](src/ingest.py) | Load, chunk, embed, store (Chroma) |
| [src/retrieval.py](src/retrieval.py) | Top-k retrieval from vector store |
| [src/qa.py](src/qa.py) | Q&A workflow + gate |
| [src/extraction.py](src/extraction.py) | Extraction workflow + gate |
| [src/gates.py](src/gates.py) | Human-review decision logic |
| [docs/CHUNKING.md](docs/CHUNKING.md) | Chunking options and config |
| [docs/EVALUATION_CHECKLIST.md](docs/EVALUATION_CHECKLIST.md) | Evaluation checklist for the prototype |
| [data/](data/) | Optional folder for sample documents |

## Workflows and gates

- **Q&A**: `run_qa(question, vector_store, llm, ...)` in [src/qa.py](src/qa.py). Gate: [src/gates.py](src/gates.py) `qa_needs_review()` (low confidence, few chunks, uncertain phrasing).
- **Extraction**: `run_extraction(vector_store, schema, llm, ...)` in [src/extraction.py](src/extraction.py). Gate: `extraction_needs_review()` (uncertain fields, validation errors).

Thresholds are configurable via env or [src/config.py](src/config.py).
