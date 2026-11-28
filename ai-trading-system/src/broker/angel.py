from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import struct
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from typing import Any
from urllib.request import Request, urlopen

from broker.base import BrokerConfig, BrokerAdapter, NormalizedTick

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
        self._instrument_tokens: dict[str, str] = {}
        self._symbols_by_token: dict[str, str] = {}
        self._receive_task: asyncio.Task[None] | None = None

    @staticmethod
    def _generate_totp(secret: str, time_offset_seconds: int = 0) -> str:
        normalized = secret.strip().upper().replace(" ", "")
        padding = "=" * ((8 - len(normalized) % 8) % 8)
        key = base64.b32decode(normalized + padding)
        counter = int(time.time() + time_offset_seconds) // 30
        digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
        offset = digest[-1] & 0x0F
        binary = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
        return f"{binary % 1_000_000:06d}"

    def login(self, totp_time_offset_seconds: int | None = None) -> dict[str, Any]:
        if totp_time_offset_seconds is None:
            totp_time_offset_seconds = int(os.getenv("ANGEL_TOTP_TIME_OFFSET_SECONDS", "0"))
        api_key = os.getenv("ANGEL_API_KEY", self.config.api_key)
        client_id = os.getenv("ANGEL_CLIENT_ID", self.config.client_id)
        pin = os.getenv("ANGEL_PIN", self.config.api_secret)
        totp_secret = os.getenv("ANGEL_TOTP_SECRET", self.config.totp_secret)

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
            "totp": self._generate_totp(totp_secret, totp_time_offset_seconds),
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

            session = await asyncio.to_thread(self.login)
            headers = {
                "Authorization": f"Bearer {session['jwtToken']}",
                "x-api-key": os.getenv("ANGEL_API_KEY", self.config.api_key),
                "x-client-code": os.getenv("ANGEL_CLIENT_ID", self.config.client_id),
                "x-feed-token": session["feedToken"],
            }
            logger.info("connecting_to_angel_one", extra={"url": self.config.websocket_url})
            self._ws = await websockets.connect(
                self.config.websocket_url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=20,
            )
            self._connected = True
            self._reconnect_delay = 1.0
            self._receive_task = asyncio.create_task(self._receive_loop())
            logger.info("angel_one_connected")
        except Exception:
            logger.exception("angel_one_connect_failed")
            self._connected = False
            raise

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._connected = False
        if self._receive_task and self._receive_task is not asyncio.current_task():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("angel_one_disconnected")

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to Angel One WebSocket feed."""
        if not self._connected:
            await self.connect()

        if not self._instrument_tokens:
            self._instrument_tokens, self._symbols_by_token = await asyncio.to_thread(
                self._load_instrument_tokens
            )

        tokens = self._symbols_to_tokens(symbols)
        self._subscribed_symbols.update(symbols)
        logger.info("subscribing", extra={"symbols": list(self._subscribed_symbols)})

        sub_msg = {
            "correlationID": "ati-market",
            "action": 1,
            "params": {
                "mode": 3,
                "tokenList": [
                    {"exchangeType": 1, "tokens": tokens}
                ],
            },
        }

        if self._ws:
            await self._ws.send(json.dumps(sub_msg))

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from Angel One feed."""
        tokens = self._symbols_to_tokens(symbols)
        self._subscribed_symbols.difference_update(symbols)
        if self._ws and self._connected:
            await self._ws.send(json.dumps({
                "correlationID": "ati-market",
                "action": 0,
                "params": {
                    "mode": 3,
                    "tokenList": [{"exchangeType": 1, "tokens": tokens}],
                },
            }))
        logger.info("unsubscribing", extra={"symbols": symbols})

    async def _receive_loop(self) -> None:
        """Receive ticks from Angel One WebSocket."""
        while self._connected and self._ws:
            try:
                raw = await self._ws.recv()
                if isinstance(raw, str):
                    logger.info("angel_websocket_message", extra={"message": raw})
                    continue
                tick = self._parse_tick(raw)
                if tick:
                    await self._emit_tick_async(tick)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("tick_receive_error")
                self._connected = False
                break

    def _load_instrument_tokens(self) -> tuple[dict[str, str], dict[str, str]]:
        if not self.config.instrument_master_url:
            raise RuntimeError("Angel One instrument master URL is not configured")

        request = Request(
            self.config.instrument_master_url,
            headers={"User-Agent": "autonomous-trading-intelligence/1.0"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                instruments = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"Unable to load Angel One instrument master: {exc}") from exc

        by_symbol: dict[str, str] = {}
        by_token: dict[str, str] = {}
        for instrument in instruments:
            if instrument.get("exch_seg") != "NSE":
                continue
            trading_symbol = str(instrument.get("symbol", ""))
            if not trading_symbol.endswith("-EQ"):
                continue
            symbol = trading_symbol[:-3].upper()
            token = str(instrument.get("token", ""))
            if symbol and token:
                by_symbol[symbol] = token
                by_token[token] = symbol
        return by_symbol, by_token

    def _symbols_to_tokens(self, symbols: list[str]) -> list[str]:
        missing = [symbol for symbol in symbols if symbol.upper() not in self._instrument_tokens]
        if missing:
            raise ValueError(f"No Angel One NSE equity token found for: {', '.join(missing)}")
        return [self._instrument_tokens[symbol.upper()] for symbol in symbols]

    def _parse_tick(self, data: bytes | bytearray | dict[str, Any]) -> NormalizedTick | None:
        """Parse Angel One WebSocket message into NormalizedTick."""
        try:
            if isinstance(data, dict):
                symbol = data.get("symbol", data.get("tradingSymbol", ""))
                ltp = float(data.get("ltp", data.get("lastTradedPrice", 0)))
                if not symbol or ltp <= 0:
                    return None
                return NormalizedTick(
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    ltp=ltp,
                    bid=float(data["bid"]) if data.get("bid") else None,
                    ask=float(data["ask"]) if data.get("ask") else None,
                    volume=int(data.get("volume", 0)),
                )

            if len(data) < 51:
                raise ValueError(f"Angel One quote packet is too short: {len(data)} bytes")
            mode = data[0]
            token = data[2:27].split(b"\0", 1)[0].decode("ascii")
            symbol = self._symbols_by_token.get(token)
            ltp = struct.unpack_from("<q", data, 43)[0] / 100.0
            timestamp_ms = struct.unpack_from("<q", data, 35)[0]
            timestamp = (
                datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                if timestamp_ms > 0 else datetime.now(timezone.utc)
            )
            volume = 0
            bid = None
            ask = None
            if mode in (2, 3) and len(data) >= 75:
                volume = int(struct.unpack_from("<q", data, 67)[0])
            if mode == 3 and len(data) >= 167:
                for index in range(5):
                    offset = 147 + index * 20
                    side = struct.unpack_from("<H", data, offset)[0]
                    level_price = struct.unpack_from("<q", data, offset + 10)[0] / 100.0
                    if level_price <= 0:
                        continue
                    if side == 0 and bid is None:
                        bid = level_price
                    elif side != 0 and ask is None:
                        ask = level_price

            if not symbol or ltp <= 0:
                return None
            return NormalizedTick(
                symbol=symbol,
                timestamp=timestamp,
                ltp=ltp,
                bid=bid,
                ask=ask,
                volume=volume,
            )
        except (ValueError, UnicodeDecodeError, struct.error):
            logger.exception("tick_parse_failed")
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