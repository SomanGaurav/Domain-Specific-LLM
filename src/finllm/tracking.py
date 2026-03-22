"""MLflow setup and logging helpers shared by training, RAG, and evaluation."""

from typing import Any

import mlflow


def setup_mlflow(config: dict[str, Any]) -> None:
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, full_key))
        else:
            flat[full_key] = value
    return flat


def log_config(config: dict[str, Any]) -> None:
    """Log the full (merged) experiment config as flattened MLflow params."""
    params = _flatten(config)
    # MLflow caps param values at 500 chars; stringify and truncate defensively.
    mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
