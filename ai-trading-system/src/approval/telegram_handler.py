from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)
TelegramCommandHandler = Callable[[str], Awaitable[None]]


class TelegramAPIError(RuntimeError):
    pass


class TelegramApprovalHandler:
    """Long-polling Telegram adapter restricted to one configured user and chat."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        allowed_user_id: str,
        poll_timeout_seconds: int = 20,
    ) -> None:
        if not bot_token.strip() or not chat_id.strip() or not allowed_user_id.strip():
            raise ValueError("Telegram token, chat ID, and allowed user ID are required")
        try:
            int(chat_id)
            int(allowed_user_id)
        except ValueError as exc:
            raise ValueError("Telegram chat and user IDs must be numeric") from exc
        if not 0 < poll_timeout_seconds <= 50:
            raise ValueError("Telegram poll timeout must be between 1 and 50 seconds")

        self._bot_token = bot_token
        self.chat_id = chat_id
        self.allowed_user_id = allowed_user_id
        self.poll_timeout_seconds = poll_timeout_seconds
        self._offset: int | None = None
        self._stopping = asyncio.Event()

    async def send_message(self, text: str) -> None:
        await self._api_call("sendMessage", {"chat_id": self.chat_id, "text": text})

    async def send_approval(
        self,
        ticket: str,
        proposal: dict[str, Any],
        timeout_seconds: int,
    ) -> None:
        side = str(proposal.get("action", "BUY"))
        message = (
            "PAPER ENTRY APPROVAL REQUIRED\n"
            f"Ticket: {ticket}\n"
            f"Symbol: {proposal.get('symbol', 'unknown')}\n"
            f"Action: {side}\n"
            f"Quantity: {proposal.get('quantity', 0)}\n"
            f"Reference price: {float(proposal.get('price', 0)):.2f}\n"
            f"Order value: INR {float(proposal.get('order_value_inr', 0)):.2f}\n"
            f"Expires in: {timeout_seconds} seconds\n\n"
            f"Approve exactly this proposal: /approve {ticket}\n"
            f"Reject it: /reject {ticket}"
        )
        await self.send_message(message)

    async def poll(self, on_command: TelegramCommandHandler) -> None:
        while not self._stopping.is_set():
            if await self._discard_old_updates():
                break
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=2)
            except TimeoutError:
                pass
        while not self._stopping.is_set():
            try:
                updates = await self._api_call(
                    "getUpdates",
                    {
                        "offset": self._offset,
                        "timeout": self.poll_timeout_seconds,
                        "allowed_updates": ["message"],
                    },
                    timeout_seconds=self.poll_timeout_seconds + 10,
                )
                for update in updates:
                    self._offset = int(update["update_id"]) + 1
                    message = update.get("message") or {}
                    sender = message.get("from") or {}
                    chat = message.get("chat") or {}
                    if sender.get("is_bot"):
                        continue
                    if (
                        str(sender.get("id")) != self.allowed_user_id
                        or str(chat.get("id")) != self.chat_id
                    ):
                        logger.warning("telegram_command_unauthorized")
                        continue
                    text = message.get("text")
                    if isinstance(text, str):
                        try:
                            await on_command(text)
                        except Exception:
                            logger.exception("telegram_command_handler_failed")
                            try:
                                await self.send_message(
                                    "Command failed; check the application logs."
                                )
                            except TelegramAPIError as exc:
                                logger.error(
                                    "telegram_error_notice_failed",
                                    error=str(exc),
                                )
            except asyncio.CancelledError:
                raise
            except TelegramAPIError as exc:
                logger.error("telegram_poll_failed", error=str(exc))
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=2)
                except TimeoutError:
                    pass

    def stop(self) -> None:
        self._stopping.set()

    async def _discard_old_updates(self) -> bool:
        try:
            updates = await self._api_call(
                "getUpdates",
                {"offset": -1, "limit": 1, "timeout": 0, "allowed_updates": ["message"]},
                timeout_seconds=10,
            )
            if updates:
                self._offset = int(updates[-1]["update_id"]) + 1
            return True
        except TelegramAPIError as exc:
            logger.error("telegram_initial_poll_failed", error=str(exc))
            return False

    async def _api_call(
        self,
        method: str,
        payload: dict[str, Any],
        timeout_seconds: int = 15,
    ) -> Any:
        return await asyncio.to_thread(
            self._api_call_sync,
            method,
            payload,
            timeout_seconds,
        )

    def _api_call_sync(
        self,
        method: str,
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> Any:
        url = f"https://api.telegram.org/bot{self._bot_token}/{method}"
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                result = json.loads(response.read())
        except HTTPError as exc:
            raise TelegramAPIError(
                f"Telegram {method} returned HTTP {exc.code}"
            ) from None
        except (URLError, TimeoutError, OSError) as exc:
            raise TelegramAPIError(
                f"Telegram {method} request failed: {type(exc).__name__}"
            ) from None
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TelegramAPIError(
                f"Telegram {method} returned invalid JSON: {type(exc).__name__}"
            ) from None

        if not isinstance(result, dict) or not result.get("ok"):
            error_code = result.get("error_code") if isinstance(result, dict) else None
            raise TelegramAPIError(
                f"Telegram {method} was rejected"
                + (f" with error code {error_code}" if error_code else "")
            )
        return result.get("result")
