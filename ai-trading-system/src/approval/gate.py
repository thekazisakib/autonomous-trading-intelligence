from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from approval.telegram_handler import TelegramApprovalHandler


@dataclass(frozen=True)
class ApprovalResult:
    approved: bool
    reason: str
    ticket: str | None = None


class ApprovalGate:
    """Starts a Telegram approval request; execution awaits a later command."""

    def __init__(self, handler: TelegramApprovalHandler) -> None:
        self.handler = handler

    async def request_approval(
        self,
        ticket: str,
        proposal: dict[str, Any],
        timeout_seconds: int,
    ) -> ApprovalResult:
        await self.handler.send_approval(ticket, proposal, timeout_seconds)
        return ApprovalResult(
            approved=False,
            reason="pending operator decision",
            ticket=ticket,
        )
