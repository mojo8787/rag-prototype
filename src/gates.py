"""Needs human review decision gates for Q&A and extraction."""
from __future__ import annotations

import re
from typing import Any, List

from . import config

# Phrases that suggest the model is uncertain or refusing to answer
QA_UNCERTAIN_PHRASES = (
    r"i\s+don'?t\s+know",
    r"not\s+(?:found|mentioned|stated|in\s+the\s+document)",
    r"cannot\s+(?:find|determine|answer)",
    r"no\s+information",
    r"insufficient\s+(?:context|information)",
    r"not\s+in\s+the\s+context",
    r"unclear",
    r"cannot\s+be\s+determined",
)


def qa_needs_review(
    answer: str,
    confidence: float | None,
    num_chunks: int,
    threshold_low_confidence: float | None = None,
    min_chunks: int | None = None,
) -> tuple[bool, str]:
    """
    Decide whether a Q&A result needs human review.

    Returns:
        (needs_review: bool, reason: str)
    """
    threshold = (
        threshold_low_confidence
        if threshold_low_confidence is not None
        else config.GATE_CONFIDENCE_THRESHOLD
    )
    min_c = min_chunks if min_chunks is not None else config.GATE_MIN_CHUNKS

    if num_chunks < min_c:
        return True, "no_context"

    if confidence is not None and confidence < threshold:
        return True, "low_confidence"

    answer_lower = (answer or "").strip().lower()
    for pattern in QA_UNCERTAIN_PHRASES:
        if re.search(pattern, answer_lower):
            return True, "uncertain_phrasing"

    return False, "ok"


def extraction_needs_review(
    record: dict[str, Any],
    uncertain_fields: List[str],
    validation_errors: List[str],
) -> tuple[bool, str]:
    """
    Decide whether an extraction result needs human review.

    Returns:
        (needs_review: bool, reason: str)
    """
    if validation_errors:
        return True, "validation_errors"
    if uncertain_fields:
        return True, "uncertain_fields"
    return False, "ok"
