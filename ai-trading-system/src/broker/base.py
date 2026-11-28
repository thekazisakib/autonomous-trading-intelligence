from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable


@dataclass
class BrokerConfig:
    provider: str
    api_key: str
    api_secret: str
    base_url: str
    websocket_url: str
    rate_limit_per_second: int = 3
    daily_api_call_limit: int = 1000
    client_id: str = ""
    totp_secret: str = ""
    instrument_master_url: str = ""


@dataclass
class NormalizedTick:
    symbol: str
    timestamp: datetime
    ltp: float
    bid: float | None = None
    ask: float | None = None
    bid_qty: int | None = None
    ask_qty: int | None = None
    volume: int = 0


class BrokerAdapter(ABC):
    """Abstract broker adapter.

    Every broker implementation must:
    1. Connect to WebSocket for realtime ticks
    2. Normalize ticks to NormalizedTick
    3. Handle reconnection with backoff
    4. Report data freshness
    5. Place orders (paper or live)
    """

    def __init__(self, config: BrokerConfig) -> None:
        self.config = config
        self._connected = False
        self._on_tick_callback: Callable[[NormalizedTick], None] | None = None
        self._last_tick_time: dict[str, datetime] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    async def connect(self) -> None:
        """Establish WebSocket connection to broker."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close WebSocket connection."""

    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to realtime ticks for given symbols."""

    @abstractmethod
    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from ticks."""

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        order_type: str = "LIMIT",
    ) -> dict:
        """Place an order through broker API. Returns broker response."""

    @abstractmethod
    async def get_quote(self, symbol: str) -> NormalizedTick | None:
        """Fetch latest quote via REST."""

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """Fetch current positions from broker."""

    def set_on_tick(self, callback: Callable[[NormalizedTick], None]) -> None:
        """Register callback for incoming ticks."""
        self._on_tick_callback = callback

    def _emit_tick(self, tick: NormalizedTick) -> None:
        """Internal: emit tick to registered callback."""
        self._last_tick_time[tick.symbol] = datetime.now(timezone.utc)
        if self._on_tick_callback:
            self._on_tick_callback(tick)

    async def _emit_tick_async(self, tick: NormalizedTick) -> None:
        """Emit a tick and await asynchronous consumers when necessary."""
        import inspect

        self._last_tick_time[tick.symbol] = datetime.now(timezone.utc)
        if self._on_tick_callback:
            result = self._on_tick_callback(tick)
            if inspect.isawaitable(result):
                await result

    def get_data_freshness(self, symbol: str) -> float | None:
        """Return seconds since last tick for symbol, or None if never received."""
        last = self._last_tick_time.get(symbol)
        if not last:
            return None
        return (datetime.now(timezone.utc) - last).total_seconds()