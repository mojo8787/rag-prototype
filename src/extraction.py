"""Document extraction workflow: retrieve, prompt LLM for structured data, apply gate."""
from __future__ import annotations

import json
import re
from typing import Any, List, Type

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field, ValidationError

from . import config
from .gates import extraction_needs_review
from .retrieval import retrieve


class DefaultExtractionSchema(BaseModel):
    """Preset schema for document extraction: dates, parties, amounts."""

    dates: str | None = Field(default=None, description="Relevant dates mentioned")
    parties: str | None = Field(default=None, description="People or organizations involved")
    amounts: str | None = Field(default=None, description="Monetary amounts or quantities")
    summary: str | None = Field(default=None, description="Brief summary of the document")

EXTRACTION_SYSTEM = """You extract structured data from the provided document chunks. Output valid JSON only.
For each field you are uncertain about, include an "uncertain_fields" array listing those field names.
If a field is missing or unclear, use null for its value and add it to "uncertain_fields"."""

EXTRACTION_USER_TEMPLATE = """Context (chunks from the document):

{context}

Extract the following fields into a JSON object. Use the exact field names. Add an "uncertain_fields" array if any value is uncertain.

Schema / fields:
{schema_desc}

Output only one JSON object with the field names as keys and "uncertain_fields" as an optional array of strings."""


def _schema_description(schema: Type[BaseModel]) -> str:
    """Produce a short description of the Pydantic model for the prompt."""
    lines = []
    for name, field in schema.model_fields.items():
        info = field.description or str(field.annotation)
        lines.append(f"- {name}: {info}")
    return "\n".join(lines) if lines else schema.model_json_schema().get("properties", {}).__str__()


def _format_context(chunks: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(chunks, 1):
        parts.append(f"[Chunk {i}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _parse_extraction_response(content: str, schema: Type[BaseModel]) -> tuple[dict[str, Any], List[str]]:
    """Parse LLM JSON response and return (record, uncertain_fields)."""
    content = content.strip()
    # Try to find a JSON object in the response
    start = content.find("{")
    if start == -1:
        return {}, ["parse_error"]
    end = content.rfind("}") + 1
    if end <= start:
        return {}, ["parse_error"]
    json_str = content[start:end]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return {}, ["parse_error"]

    uncertain = list(data.pop("uncertain_fields", [])) if isinstance(data.get("uncertain_fields"), list) else []
    # Build record with only schema fields
    record = {}
    for name in schema.model_fields:
        record[name] = data.get(name)
    return record, uncertain


def _validate_record(record: dict[str, Any], schema: Type[BaseModel]) -> List[str]:
    """Validate record against schema; return list of validation error messages."""
    try:
        schema.model_validate(record)
        return []
    except ValidationError as e:
        errors = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", str(err))
            errors.append(f"{loc}: {msg}")
        return errors


def run_extraction(
    vector_store: Chroma,
    schema: Type[BaseModel],
    llm: BaseChatModel,
    query: str | None = None,
    top_k: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run document extraction: retrieve chunks, prompt LLM for structured data, validate, apply gate.

    If query is None, uses a generic retrieval query based on schema field names.

    Returns:
        {
            "record": dict (schema fields),
            "uncertain_fields": list of str,
            "validation_errors": list of str,
            "source_chunks": list of Document,
            "needs_review": bool,
            "review_reason": str,
        }
    """
    if query is None:
        query = " ".join(schema.model_fields.keys())
    chunks = retrieve(vector_store, query, top_k=top_k)
    context = _format_context(chunks)
    schema_desc = _schema_description(schema)

    user_msg = EXTRACTION_USER_TEMPLATE.format(context=context, schema_desc=schema_desc)
    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM),
        HumanMessage(content=user_msg),
    ]
    response = llm.invoke(messages)
    content = getattr(response, "content", str(response))
    record, uncertain = _parse_extraction_response(content, schema)
    validation_errors = _validate_record(record, schema)

    needs_review, reason = extraction_needs_review(
        record=record,
        uncertain_fields=uncertain,
        validation_errors=validation_errors,
    )

    return {
        "record": record,
        "uncertain_fields": uncertain,
        "validation_errors": validation_errors,
        "source_chunks": chunks,
        "needs_review": needs_review,
        "review_reason": reason,
    }
