from __future__ import annotations

from broker.base import BrokerAdapter, BrokerConfig
from broker.angel import AngelOneAdapter

_PROVIDER_REGISTRY: dict[str, type[BrokerAdapter]] = {
    "angel": AngelOneAdapter,
}


def get_adapter(provider: str, config: BrokerConfig) -> BrokerAdapter:
    """Factory: return broker adapter instance for given provider."""
    if provider not in _PROVIDER_REGISTRY:
        raise ValueError(f"Unsupported broker provider: {provider}. Available: {list(_PROVIDER_REGISTRY.keys())}")
    return _PROVIDER_REGISTRY[provider](config)