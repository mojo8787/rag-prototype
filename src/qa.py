"""Document Q&A workflow: retrieve, prompt LLM, apply human-review gate."""
from __future__ import annotations

import re
from typing import Any, List

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore

from . import config
from .gates import qa_needs_review
from .retrieval import retrieve

QA_SYSTEM = """You answer questions using ONLY the provided context. If the answer is not in the context, say "I don't know" or that the information is not in the document.
Cite which chunk(s) support your answer (e.g. "Chunk 1").
At the end, on a new line, write your confidence as a number from 0.0 to 1.0, e.g. "Confidence: 0.85"."""

QA_USER_TEMPLATE = """Context (chunks from the document):

{context}

Question: {question}

Answer based only on the context above. End with "Confidence: X.XX"."""


def _format_context(chunks: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(chunks, 1):
        parts.append(f"[Chunk {i}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _parse_confidence(content: str) -> tuple[str, float | None]:
    """Extract confidence line and return (answer_without_confidence, confidence)."""
    content = content.strip()
    match = re.search(r"\s*Confidence:\s*([0-9]*\.?[0-9]+)\s*$", content, re.IGNORECASE)
    if match:
        try:
            conf = float(match.group(1))
            conf = max(0.0, min(1.0, conf))
            answer = content[: match.start()].strip()
            return answer, conf
        except ValueError:
            pass
    return content, None


def run_qa(
    question: str,
    vector_store: VectorStore,
    llm: BaseChatModel,
    top_k: int | None = None,
    threshold_low_confidence: float | None = None,
    min_chunks: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run document Q&A: retrieve chunks, prompt LLM, apply human-review gate.

    Returns:
        {
            "answer": str,
            "confidence": float | None,
            "source_chunks": list of Document,
            "needs_review": bool,
            "review_reason": str,
        }
    """
    chunks = retrieve(vector_store, question, top_k=top_k)
    context = _format_context(chunks)

    user_msg = QA_USER_TEMPLATE.format(context=context, question=question)
    messages = [
        SystemMessage(content=QA_SYSTEM),
        HumanMessage(content=user_msg),
    ]
    response = llm.invoke(messages)
    content = getattr(response, "content", str(response))
    answer, confidence = _parse_confidence(content)

    needs_review, reason = qa_needs_review(
        answer=answer,
        confidence=confidence,
        num_chunks=len(chunks),
        threshold_low_confidence=threshold_low_confidence or config.GATE_CONFIDENCE_THRESHOLD,
        min_chunks=min_chunks or config.GATE_MIN_CHUNKS,
    )

    return {
        "answer": answer,
        "confidence": confidence,
        "source_chunks": chunks,
        "needs_review": needs_review,
        "review_reason": reason,
    }
