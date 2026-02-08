# RAG Prototype — User Guide

How to set up and use the RAG prototype (Streamlit UI or API).

**Live demo:** [Try the app](https://rag-prototype-ezwo58dutziwjacwqxyhqn.streamlit.app/) — run in the browser (add `OPENAI_API_KEY` in Streamlit secrets).

---

## 1. Set your OpenAI API key

The app needs `OPENAI_API_KEY` for embeddings and the LLM.

### Local (Streamlit or API)

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
```

Or export it before running:

```bash
export OPENAI_API_KEY=sk-your-key-here
streamlit run streamlit_app.py
```

### Streamlit Community Cloud

1. Open your app on [share.streamlit.io](https://share.streamlit.io)
2. Click **Settings** (⋮) → **Secrets**
3. Add:

```toml
OPENAI_API_KEY = "sk-your-key-here"
```

4. Save. The app will reload with the key.

---

## 2. Run the app

**Streamlit UI:**
```bash
streamlit run streamlit_app.py
```

**API only:**
```bash
uvicorn api:app --port 8000
```

**Docker (API):**
```bash
docker compose up --build api
```

---

## 3. Use the Streamlit UI

### Step 1: Ingest

1. Go to **📥 Ingest**
2. Upload one or more `.txt` or `.pdf` files (drag & drop or click)
3. Click **🚀 Ingest**
4. Wait for "Documents loaded" in the sidebar

### Step 2: Q&A

1. Go to **💬 Q&A**
2. Enter a question (e.g. "What is the total contract value?")
3. Click **Run Q&A**
4. View the answer, confidence, sources, and whether it needs human review

### Step 3: Extraction

1. Go to **📋 Extraction**
2. Click **Run extraction**
3. View extracted fields (dates, parties, amounts, terms, summary)
4. Check **Needs human review** and the reason

---

## 4. Use the API (curl)

```bash
# Health
curl http://localhost:8000/health

# Ingest
curl -X POST http://localhost:8000/ingest -F "files=@data/sample_contract.txt"

# Extract
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" -d '{}'

# Q&A
curl -X POST http://localhost:8000/qa -H "Content-Type: application/json" \
  -d '{"question": "What is the total contract value?"}'
```

---

## 5. Troubleshooting

| Issue | Fix |
|-------|-----|
| **API Key not set** | Add `OPENAI_API_KEY` to `.env` or Streamlit secrets (see above) |
| **No documents yet** | Run Ingest first; upload files and click Ingest |
| **ModuleNotFoundError** | Run `pip install -r requirements.txt` |
| **Port in use** | Use a different port: `streamlit run streamlit_app.py --server.port 8502` |

---

## 6. Sample documents

Use these to test without your own files:

- `data/sample_contract.txt` — Service agreement (dates, parties, amounts)
- `data/sample_nda.txt` — NDA (confidentiality terms)

Upload either file in Ingest to get started.
