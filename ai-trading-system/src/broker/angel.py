from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import os
import struct
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

from broker.base import BrokerConfig, BrokerAdapter, NormalizedTick
import json
import logging

logger = logging.getLogger(__name__)


class AngelOneAdapter(BrokerAdapter):
    """Angel One broker adapter for realtime trading.
    
    Angel One provides free API access for retail algo trading.
    Requires API key, secret, client ID, PIN, and TOTP secret.
    """

    def __init__(self, config: BrokerConfig) -> None:
        super().__init__(config)
        self._ws: Any = None
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._subscribed_symbols: set[str] = set()
        self._jwt_token: str | None = None
        self._feed_token: str | None = None

    @staticmethod
    def _generate_totp(secret: str) -> str:
        normalized = secret.strip().upper().replace(" ", "")
        padding = "=" * ((8 - len(normalized) % 8) % 8)
        key = base64.b32decode(normalized + padding)
        counter = int(time.time()) // 30
        digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
        offset = digest[-1] & 0x0F
        binary = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
        return f"{binary % 1_000_000:06d}"

    def login(self) -> dict[str, Any]:
        api_key = os.getenv("ANGEL_API_KEY", self.config.api_key)
        client_id = os.getenv("ANGEL_CLIENT_ID", "")
        pin = os.getenv("ANGEL_PIN", self.config.api_secret)
        totp_secret = os.getenv("ANGEL_TOTP_SECRET", "")

        missing = [
            name for name, value in (
                ("ANGEL_API_KEY", api_key),
                ("ANGEL_CLIENT_ID", client_id),
                ("ANGEL_PIN", pin),
                ("ANGEL_TOTP_SECRET", totp_secret),
            ) if not value
        ]
        if missing:
            raise RuntimeError(f"Missing Angel One credentials: {', '.join(missing)}")

        payload = json.dumps({
            "clientcode": client_id,
            "password": pin,
            "totp": self._generate_totp(totp_secret),
        }).encode("utf-8")
        request = Request(
            f"{self.config.base_url}/rest/auth/angelbroking/user/v1/loginByPassword",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": "00:00:00:00:00:00",
                "X-PrivateKey": api_key,
                "X-SourceID": "WEB",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=20) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Angel One login failed HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Angel One login network error: {exc.reason}") from exc

        if not result.get("status") or not result.get("data"):
            raise RuntimeError(f"Angel One login rejected: {result}")

        self._jwt_token = result["data"].get("jwtToken")
        self._feed_token = result["data"].get("feedToken")
        return result["data"]

    async def connect(self) -> None:
        """Authenticate and connect to Angel One WebSocket."""
        try:
            import websockets  # type: ignore

            logger.info("connecting_to_angel_one", url=self.config.websocket_url)
            self._ws = await websockets.connect(self.config.websocket_url)
            self._connected = True
            self._reconnect_delay = 1.0
            logger.info("angel_one_connected")
        except Exception:
            logger.exception("angel_one_connect_failed")
            self._connected = False
            raise

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self._ws:
            await self._ws.close()
        self._connected = False
        logger.info("angel_one_disconnected")

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to Angel One WebSocket feed."""
        if not self._connected:
            await self.connect()

        self._subscribed_symbols.update(symbols)
        logger.info("subscribing", symbols=list(self._subscribed_symbols))

        # Angel One subscription message format
        sub_msg = {
            "action": "subscribe",
            "params": {
                "mode": "LTP",
                "tokenList": [
                    {"exchange": "NSE", "tokens": self._symbols_to_tokens(symbols)}
                ],
            },
        }

        if self._ws:
            await self._ws.send(json.dumps(sub_msg))

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from Angel One feed."""
        self._subscribed_symbols.difference_update(symbols)
        logger.info("unsubscribing", symbols=symbols)

    async def _receive_loop(self) -> None:
        """Receive ticks from Angel One WebSocket."""
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

    def _symbols_to_tokens(self, symbols: list[str]) -> list[str]:
        """Map symbol names to Angel One token IDs.
        
        In production, maintain a symbol-to-token mapping table.
        For now, return symbol names as placeholder tokens.
        """
        # Token mapping should be loaded from Angel One master contract
        return symbols

    def _parse_tick(self, data: dict[str, Any]) -> NormalizedTick | None:
        """Parse Angel One WebSocket message into NormalizedTick."""
        try:
            symbol = data.get("symbol", data.get("tradingSymbol", ""))
            ltp = float(data.get("ltp", data.get("lastTradedPrice", 0)))

            if not symbol or ltp <= 0:
                return None

            return NormalizedTick(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                ltp=ltp,
                bid=float(data.get("bid", 0)) if data.get("bid") else None,
                ask=float(data.get("ask", 0)) if data.get("ask") else None,
                volume=int(data.get("volume", 0)),
            )
        except Exception:
            logger.exception("tick_parse_failed raw=%s", data)
            return None

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        order_type: str = "LIMIT",
    ) -> dict:
        """Place order via Angel One REST API."""
        logger.warning(
            "live_order_attempt",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
        )
        return {
            "status": "not_implemented",
            "message": "Angel One order placement not yet wired",
        }

    async def get_quote(self, symbol: str) -> NormalizedTick | None:
        """Fetch latest quote via Angel One REST."""
        logger.warning("get_quote_not_implemented", symbol=symbol)
        return None

    async def get_positions(self) -> list[dict]:
        """Fetch positions from Angel One."""
        logger.warning("get_positions_not_implemented")
        return []