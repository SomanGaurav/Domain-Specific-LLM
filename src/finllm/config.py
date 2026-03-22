"""YAML experiment configs with single-parent inheritance via an `inherits:` key."""

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, recursively merging its `inherits:` parent (child wins)."""
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    parent_path = config.pop("inherits", None)
    if parent_path:
        parent = load_config(parent_path)
        config = _deep_merge(parent, config)
    return config
