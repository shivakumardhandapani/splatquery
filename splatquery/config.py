"""Lightweight config: load YAML into attribute-accessible nested namespaces,
with dotted-key overrides from the CLI (e.g. --set agent.backend=local)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict that also supports attribute access and nested dotted lookups."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_dotted(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_dotted(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self
        for part in parts[:-1]:
            node = node.setdefault(part, Config())
        node[parts[-1]] = _coerce(value)


def _coerce(value: Any) -> Any:
    """Turn CLI string values into bools/ints/floats/None where sensible."""
    if not isinstance(value, str):
        return value
    low = value.lower()
    if low in {"null", "none"}:
        return None
    if low in {"true", "false"}:
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config and apply ``key=value`` dotted overrides."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = Config(copy.deepcopy(raw))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, val = item.split("=", 1)
        cfg.set_dotted(key.strip(), val.strip())
    return cfg
