"""FastAPI app for RAG prototype: ingest, extract, Q&A."""
from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src import config
from src.extraction import DefaultExtractionSchema, run_extraction
from src.ingest import ingest_documents
from src.llm_factory import get_llm
from src.mlflow_logging import log_extraction_run, log_qa_run
from src.qa import run_qa

app = FastAPI(
    title="RAG Contract Extraction API",
    description="Document ingestion, extraction, and Q&A for contracts.",
    version="1.0.0",
)

# Single-tenant: one active vector store per process
_vector_store = None
_collection_id: str | None = None


def _get_vector_store():
    """Return current vector store or raise."""
    global _vector_store
    if _vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested. POST /ingest first.",
        )
    return _vector_store


class IngestResponse(BaseModel):
    status: str = "ok"
    collection_id: str


class ExtractRequest(BaseModel):
    query: str | None = Field(default=None, description="Optional retrieval query")


class QaRequest(BaseModel):
    question: str = Field(..., description="Question to ask")


def _serialize_chunks(chunks):
    """Serialize Document chunks for JSON response."""
    return [
        {"content": doc.page_content[:500], "metadata": doc.metadata}
        for doc in chunks
    ]


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)):
    """Upload PDF/text files, chunk, embed, and store. Returns collection_id."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    valid_suffixes = {".txt", ".pdf"}
    paths = []
    display_names = []

    try:
        for f in files:
            suffix = Path(f.filename or "").suffix.lower()
            if suffix not in valid_suffixes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {suffix}. Use .txt or .pdf.",
                )
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            content = await f.read()
            tmp.write(content)
            tmp.close()
            paths.append(tmp.name)
            display_names.append(f.filename or tmp.name)

        def _do_ingest():
            global _vector_store, _collection_id
            collection_id = str(uuid.uuid4())
            _vector_store = ingest_documents(
                paths,
                file_display_names=display_names,
                collection_name=f"rag_{collection_id}",
            )
            _collection_id = collection_id
            return collection_id

        collection_id = await asyncio.to_thread(_do_ingest)
        return IngestResponse(status="ok", collection_id=collection_id)

    except Exception as e:
        detail = str(e)
        if "OPENAI_API_KEY" in detail.upper() or "api_key" in detail.lower():
            detail = "OPENAI_API_KEY not set or invalid. Add it in Azure App Settings."
        raise HTTPException(status_code=500, detail=detail)

    finally:
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/extract-agents")
async def extract_agents(req: ExtractRequest | None = None):
    """Multi-agent extraction (Extraction -> Validation -> Summary) via LangGraph."""
    vs = _get_vector_store()
    llm = get_llm()

    def _do_extract():
        from src.agents import run_extraction_agents
        return run_extraction_agents(vs, DefaultExtractionSchema, llm)

    result = await asyncio.to_thread(_do_extract)
    log_extraction_run(
        result,
        run_type="extraction_agents",
        chunk_size=config.CHUNK_SIZE,
        top_k=config.RETRIEVAL_TOP_K,
    )
    return {
        "record": result["record"],
        "uncertain_fields": result["uncertain_fields"],
        "validation_errors": result["validation_errors"],
        "needs_review": result["needs_review"],
        "review_reason": result["review_reason"],
        "summary": result["summary"],
        "sources": _serialize_chunks(result["source_chunks"]),
    }


@app.post("/extract")
async def extract(req: ExtractRequest | None = None):
    """Extract structured fields (dates, parties, amounts, terms) from ingested docs."""
    vs = _get_vector_store()
    llm = get_llm()
    query = req.query if req and req.query else None

    def _do_extract():
        return run_extraction(
            vs,
            DefaultExtractionSchema,
            llm,
            query=query,
        )

    result = await asyncio.to_thread(_do_extract)
    log_extraction_run(
        result,
        chunk_size=config.CHUNK_SIZE,
        top_k=config.RETRIEVAL_TOP_K,
    )
    return {
        "record": result["record"],
        "uncertain_fields": result["uncertain_fields"],
        "validation_errors": result["validation_errors"],
        "needs_review": result["needs_review"],
        "review_reason": result["review_reason"],
        "sources": _serialize_chunks(result["source_chunks"]),
    }


@app.post("/qa")
async def qa(req: QaRequest):
    """Ask a question over ingested documents. Returns answer with sources and review status."""
    vs = _get_vector_store()
    llm = get_llm()

    def _do_qa():
        return run_qa(req.question, vs, llm)

    result = await asyncio.to_thread(_do_qa)
    log_qa_run(result, top_k=config.RETRIEVAL_TOP_K)
    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "needs_review": result["needs_review"],
        "review_reason": result["review_reason"],
        "sources": _serialize_chunks(result["source_chunks"]),
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "documents_loaded": _vector_store is not None}
