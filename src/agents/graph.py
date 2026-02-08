"""LangGraph multi-agent flow: Extraction -> Validation -> Summary."""
from __future__ import annotations

from typing import Any, List, Type

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel
from typing_extensions import TypedDict

from ..extraction import (
    EXTRACTION_SYSTEM,
    EXTRACTION_USER_TEMPLATE,
    DefaultExtractionSchema,
    _format_context,
    _parse_extraction_response,
    _schema_description,
    _validate_record,
)
from ..gates import extraction_needs_review
from ..retrieval import retrieve


class ExtractionState(TypedDict, total=False):
    """State schema for extraction agent graph."""

    chunks: List[Document]
    raw_extraction: str
    record: dict[str, Any]
    uncertain_fields: List[str]
    validation_errors: List[str]
    needs_review: bool
    review_reason: str
    summary: str


SUMMARY_SYSTEM = """You write a very brief human-readable summary (2-4 sentences) of the extracted contract data. Focus on the key parties, amounts, dates, and main terms. Be concise."""

SUMMARY_USER_TEMPLATE = """Extracted contract data:

{record}

Write a short summary for a human reviewer."""


def _build_extraction_node(
    vector_store: VectorStore,
    llm: BaseChatModel,
    schema: Type[BaseModel],
):
    """Build extraction agent node (closure over deps)."""

    def extraction_node(state: dict) -> dict:
        query = " ".join(schema.model_fields.keys())
        chunks = retrieve(vector_store, query)
        context = _format_context(chunks)
        schema_desc = _schema_description(schema)

        user_msg = EXTRACTION_USER_TEMPLATE.format(
            context=context, schema_desc=schema_desc
        )
        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM),
            HumanMessage(content=user_msg),
        ]
        response = llm.invoke(messages)
        content = getattr(response, "content", str(response))
        record, uncertain = _parse_extraction_response(content, schema)

        return {
            "chunks": chunks,
            "raw_extraction": content,
            "record": record,
            "uncertain_fields": uncertain,
        }

    return extraction_node


def _build_validation_node(schema: Type[BaseModel]):
    """Build validation agent node."""

    def validation_node(state: dict) -> dict:
        record = state.get("record", {})
        uncertain = state.get("uncertain_fields", [])

        validation_errors = _validate_record(record, schema)
        needs_review, reason = extraction_needs_review(
            record=record,
            uncertain_fields=uncertain,
            validation_errors=validation_errors,
        )

        return {
            "validation_errors": validation_errors,
            "needs_review": needs_review,
            "review_reason": reason,
        }

    return validation_node


def _build_summary_node(llm: BaseChatModel):
    """Build summary agent node."""

    def summary_node(state: dict) -> dict:
        record = state.get("record", {})
        record_str = "\n".join(
            f"{k}: {v}" for k, v in record.items() if v is not None
        )
        if not record_str:
            return {"summary": "No data extracted."}

        user_msg = SUMMARY_USER_TEMPLATE.format(record=record_str)
        messages = [
            SystemMessage(content=SUMMARY_SYSTEM),
            HumanMessage(content=user_msg),
        ]
        response = llm.invoke(messages)
        summary = getattr(response, "content", str(response)).strip()

        return {"summary": summary}

    return summary_node


def run_extraction_agents(
    vector_store: VectorStore,
    schema: Type[BaseModel],
    llm: BaseChatModel,
    query: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """
    Run multi-agent extraction: Extraction -> Validation -> Summary.

    Returns same shape as run_extraction plus "summary" key.
    """
    from langgraph.graph import END, START, StateGraph

    extraction_node = _build_extraction_node(vector_store, llm, schema)
    validation_node = _build_validation_node(schema)
    summary_node = _build_summary_node(llm)

    graph = StateGraph(ExtractionState)

    graph.add_node("extraction", extraction_node)
    graph.add_node("validation", validation_node)
    graph.add_node("summary", summary_node)

    graph.add_edge(START, "extraction")
    graph.add_edge("extraction", "validation")
    graph.add_edge("validation", "summary")
    graph.add_edge("summary", END)

    compiled = graph.compile()
    result = compiled.invoke({})

    return {
        "record": result.get("record", {}),
        "uncertain_fields": result.get("uncertain_fields", []),
        "validation_errors": result.get("validation_errors", []),
        "source_chunks": result.get("chunks", []),
        "needs_review": result.get("needs_review", False),
        "review_reason": result.get("review_reason", ""),
        "summary": result.get("summary", ""),
    }
