from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from config.config_schema import AppConfig

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _interpolate(value: Any) -> Any:
    """Recursively replace ${ENV_VAR} with environment variable values."""
    if isinstance(value, str):
        def replacer(match: re.Match[str]) -> str:
            env_name = match.group(1)
            return os.environ.get(env_name, match.group(0))

        return _ENV_PATTERN.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def load_config(config_path: str | Path = "config/config.yaml") -> AppConfig:
    """Load YAML config, interpolate environment variables, and validate."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config must contain a YAML mapping, got {type(raw).__name__}")

    interpolated = _interpolate(raw)
    return AppConfig.model_validate(interpolated)