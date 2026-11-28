from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from dotenv import load_dotenv

from config.config_schema import AppConfig

if TYPE_CHECKING:
    from data.universe import UniverseSnapshot

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
    load_dotenv()
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config must contain a YAML mapping, got {type(raw).__name__}")

    interpolated = _interpolate(raw)
    return AppConfig.model_validate(interpolated)


def load_runtime_config(
    config_path: str | Path = "config/config.yaml",
) -> tuple[AppConfig, UniverseSnapshot | None]:
    """Load config and resolve its configured runtime trading universe."""
    config_path = Path(config_path)
    config = load_config(config_path)
    if config.trading.universe_source == "configured":
        return config, None

    from data.universe import load_nifty500_universe

    project_root = config_path.resolve().parent.parent
    snapshot = load_nifty500_universe(
        project_root / "data" / "nifty500_constituents.csv"
    )
    trading = config.trading.model_copy(
        update={"universe": list(snapshot.symbols)}
    )
    return config.model_copy(update={"trading": trading}), snapshot