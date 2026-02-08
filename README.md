# LLM / RAG Prototype

A small RAG prototype for contract/document extraction: upload PDFs, extract key fields (dates, parties, terms, amounts), run Q&A, and flag uncertain results for human review.

## Problem

Contract and document review is manual and error-prone. This prototype demonstrates automated extraction and Q&A over contract documents, with human-review gates for uncertain or low-confidence results.

## Approach

- **RAG pipeline**: Ingest documents, chunk, embed, store in Chroma. Retrieve top-k chunks for each query.
- **Extraction**: LLM extracts structured fields (dates, parties, amounts, terms, summary). Pydantic schema validation.
- **Human-review gates**: Q&A and extraction workflows flag results that need human review (low confidence, uncertain fields, validation errors).
- **Multi-agent workflow** (LangGraph): Extraction agent, Validation agent, Summary agent for richer extraction flows.
- **Experiment tracking**: MLFlow logs runs, params, and metrics for extraction and Q&A.

## Tech stack

- **Python**, **FastAPI** (async API), **Streamlit** (UI)
- **LangChain**, **LangGraph** (multi-agent)
- **Chroma** (vector store), **OpenAI** or **Azure OpenAI**
- **MLFlow** (experiment tracking), **Docker**

## Results

- Extracts dates, parties, amounts, terms, and summary from contracts.
- Flags uncertain results for human review.
- Supports Q&A with source citations and confidence scores.
- Multi-agent flow adds a structured summary for review.

**Demo developed by:** Almotasem Bellah Younis  
**Contact:** [motasem.youniss@gmail.com](mailto:motasem.youniss@gmail.com)  
**Location:** Brno, CZ

## What’s included

- **Document Q&A**: Ask questions over ingested documents; answers are grounded in retrieved chunks with optional confidence and source citations. A gate flags results that need human review (e.g. low confidence, no context).
- **Document extraction**: Extract structured fields (e.g. dates, parties, amounts, terms) from documents. A gate flags results with uncertain fields or validation errors.
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
   - For Azure OpenAI: set `LLM_PROVIDER=azure` and `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
   - For MLFlow: set `MLFLOW_TRACKING_URI` (default: local `./mlruns`). Set `MLFLOW_DISABLED=true` to disable.

   Optional overrides (defaults in [src/config.py](src/config.py)):

   - `CHUNK_SIZE`, `CHUNK_OVERLAP` — chunking
   - `RETRIEVAL_TOP_K` — number of chunks retrieved
   - `GATE_CONFIDENCE_THRESHOLD`, `GATE_MIN_CHUNKS` — human-review gate thresholds

## Run the app

**Streamlit UI:**

```bash
streamlit run app.py
# or
streamlit run streamlit_app.py
```

**FastAPI (async):**

```bash
uvicorn api:app --reload --port 8000
```

**Docker:**

```bash
docker compose up --build
# API: http://localhost:8000
# Streamlit (optional): docker compose --profile streamlit up
```

## API

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/ingest` | POST | Upload PDF/txt files (multipart). Returns `collection_id`. |
| `/extract` | POST | Extract structured fields (dates, parties, amounts, terms) from ingested docs. |
| `/extract-agents` | POST | Multi-agent extraction (Extraction → Validation → Summary) via LangGraph. |
| `/qa` | POST | Ask a question. Body: `{"question": "..."}`. |
| `/health` | GET | Health check. |

**Example:**

```bash
# Ingest
curl -X POST http://localhost:8000/ingest -F "files=@data/sample_contract.txt"

# Extract
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{}'

# Q&A
curl -X POST http://localhost:8000/qa -H "Content-Type: application/json" -d '{"question": "What is the contract value?"}'
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
| [api.py](api.py) | FastAPI (ingest, extract, extract-agents, qa) |
| [src/config.py](src/config.py) | Chunk size, top-k, gate thresholds |
| [src/chunking.py](src/chunking.py) | Chunking strategies |
| [src/ingest.py](src/ingest.py) | Load, chunk, embed, store (Chroma) |
| [src/retrieval.py](src/retrieval.py) | Top-k retrieval from vector store |
| [src/qa.py](src/qa.py) | Q&A workflow + gate |
| [src/extraction.py](src/extraction.py) | Extraction workflow + gate |
| [src/gates.py](src/gates.py) | Human-review decision logic |
| [src/llm_factory.py](src/llm_factory.py) | LLM/embeddings factory (OpenAI, Azure) |
| [src/agents/graph.py](src/agents/graph.py) | LangGraph multi-agent flow |
| [src/mlflow_logging.py](src/mlflow_logging.py) | MLFlow experiment tracking |
| [docs/CHUNKING.md](docs/CHUNKING.md) | Chunking options and config |
| [docs/EVALUATION_CHECKLIST.md](docs/EVALUATION_CHECKLIST.md) | Evaluation checklist for the prototype |
| [data/](data/) | Optional folder for sample documents |

## Workflows and gates

- **Q&A**: `run_qa(question, vector_store, llm, ...)` in [src/qa.py](src/qa.py). Gate: [src/gates.py](src/gates.py) `qa_needs_review()` (low confidence, few chunks, uncertain phrasing).
- **Extraction**: `run_extraction(vector_store, schema, llm, ...)` in [src/extraction.py](src/extraction.py). Gate: `extraction_needs_review()` (uncertain fields, validation errors).
- **Multi-agent extraction**: `run_extraction_agents(...)` in [src/agents/graph.py](src/agents/graph.py) — Extraction → Validation → Summary agents.

Thresholds are configurable via env or [src/config.py](src/config.py).
