"""MLFlow experiment tracking for RAG extraction, Q&A, and agent runs."""
from __future__ import annotations

import os
from typing import Any

import mlflow

EXPERIMENT_NAME = "rag-contract-extraction"


def _is_mlflow_enabled() -> bool:
    """Check if MLFlow tracking is enabled."""
    if os.getenv("MLFLOW_DISABLED", "").lower() == "true":
        return False
    return True


def _ensure_experiment() -> None:
    """Ensure experiment exists."""
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_extraction_run(
    result: dict[str, Any],
    run_type: str = "extraction",
    chunk_size: int | None = None,
    top_k: int | None = None,
    model: str | None = None,
) -> None:
    """Log extraction result to MLFlow."""
    if not _is_mlflow_enabled():
        return
    try:
        _ensure_experiment()
        with mlflow.start_run(run_name=run_type):
            mlflow.log_param("run_type", run_type)
            if chunk_size is not None:
                mlflow.log_param("chunk_size", chunk_size)
            if top_k is not None:
                mlflow.log_param("top_k", top_k)
            if model is not None:
                mlflow.log_param("model", model)
            mlflow.log_metric("needs_review", 1 if result.get("needs_review") else 0)
            mlflow.log_metric("num_uncertain_fields", len(result.get("uncertain_fields", [])))
            mlflow.log_metric("num_validation_errors", len(result.get("validation_errors", [])))
            record = result.get("record", {})
            if record:
                mlflow.log_dict(record, "extracted_record.json")
    except Exception:
        pass


def log_qa_run(
    result: dict[str, Any],
    top_k: int | None = None,
    model: str | None = None,
) -> None:
    """Log Q&A result to MLFlow."""
    if not _is_mlflow_enabled():
        return
    try:
        _ensure_experiment()
        with mlflow.start_run(run_name="qa"):
            mlflow.log_param("run_type", "qa")
            if top_k is not None:
                mlflow.log_param("top_k", top_k)
            if model is not None:
                mlflow.log_param("model", model)
            mlflow.log_metric("needs_review", 1 if result.get("needs_review") else 0)
            conf = result.get("confidence")
            if conf is not None:
                mlflow.log_metric("confidence", float(conf))
    except Exception:
        pass
