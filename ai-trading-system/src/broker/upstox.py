from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from broker.base import BrokerConfig, BrokerAdapter, NormalizedTick

logger = logging.getLogger(__name__)


class UpstoxAdapter(BrokerAdapter):
    """Upstox broker adapter.

    Upstox offers a free API for retail clients with WebSocket market data.
    Requires app creation at https://developer.upstox.com/

    Note: This is a production-ready skeleton. Actual WebSocket message parsing
    depends on Upstox's current SDK message format. Update _parse_tick() if
    Upstox changes their API.
    """

    def __init__(self, config: BrokerConfig) -> None:
        super().__init__(config)
        self._ws: Any = None
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._subscribed_symbols: set[str] = set()

    async def connect(self) -> None:
        """Connect to Upstox WebSocket."""
        try:
            import websockets  # type: ignore

            logger.info("connecting_to_upstox", url=self.config.websocket_url)
            self._ws = await websockets.connect(self.config.websocket_url)
            self._connected = True
            self._reconnect_delay = 1.0
            logger.info("upstox_connected")
        except Exception:
            logger.exception("upstox_connect_failed")
            self._connected = False
            raise

    async def disconnect(self) -> None:
        """Close WebSocket."""
        if self._ws:
            await self._ws.close()
        self._connected = False
        logger.info("upstox_disconnected")

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to Upstox WebSocket feed."""
        if not self._connected:
            await self.connect()

        self._subscribed_symbols.update(symbols)
        logger.info("subscribing", symbols=list(self._subscribed_symbols))

        # Upstox subscription message format (adjust to current API)
        sub_msg = {
            "guid": "some-guid",
            "method": "sub",
            "data": {
                "instrumentMode": "single",
                "instrumentKeys": list(self._subscribed_symbols),
            },
        }

        if self._ws:
            await self._ws.send(json.dumps(sub_msg))

        # Start receiving loop
        asyncio.create_task(self._receive_loop())

    async def unsubscribe(self, symbols: list[str]) -> None:
        self._subscribed_symbols.difference_update(symbols)
        logger.info("unsubscribing", symbols=symbols)

    async def _receive_loop(self) -> None:
        """Continuously receive ticks from WebSocket."""
        while self._connected and self._ws:
            try:
                raw = await self._ws.recv()
                data = json.loads(raw)
                tick = self._parse_tick(data)
                if tick:
                    self._emit_tick(tick)
            except Exception:
                logger.exception("tick_receive_error")
                await asyncio.sleep(1)
                await self._reconnect()

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        self._connected = False
        delay = self._reconnect_delay
        logger.info("reconnecting", delay=delay)
        await asyncio.sleep(delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

        try:
            await self.connect()
            if self._subscribed_symbols:
                await self.subscribe(list(self._subscribed_symbols))
        except Exception:
            logger.exception("reconnect_failed")

    def _parse_tick(self, data: dict[str, Any]) -> NormalizedTick | None:
        """Parse Upstox WebSocket message into NormalizedTick.

        Upstox message format may vary. Adjust field mapping here.
        """
        try:
            # Example Upstox feed structure (adjust to actual API)
            symbol = data.get("instrument_key", data.get("symbol", ""))
            ltp = float(data.get("last_price", data.get("ltp", 0)))
            bid = float(data.get("bid", 0)) if data.get("bid") else None
            ask = float(data.get("ask", 0)) if data.get("ask") else None
            volume = int(data.get("volume", 0))

            if not symbol or ltp <= 0:
                return None

            return NormalizedTick(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                ltp=ltp,
                bid=bid,
                ask=ask,
                volume=volume,
            )
        except Exception:
            logger.exception("tick_parse_failed", raw=data)
            return None

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        order_type: str = "LIMIT",
    ) -> dict:
        """Place order via Upstox REST API."""
        logger.warning(
            "live_order_attempt",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
        )
        # Implement actual Upstox order placement here
        return {
            "status": "not_implemented",
            "message": "Upstox order placement not yet wired",
        }

    async def get_quote(self, symbol: str) -> NormalizedTick | None:
        """Fetch latest quote via Upstox REST."""
        logger.warning("get_quote_not_implemented", symbol=symbol)
        return None

    async def get_positions(self) -> list[dict]:
        """Fetch positions from Upstox."""
        logger.warning("get_positions_not_implemented")
        return []