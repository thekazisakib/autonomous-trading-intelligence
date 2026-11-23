from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from approval.telegram_handler import TelegramApprovalHandler, QualityResult


@dataclass
class ApprovalResult:
    """Result of the approval process."""
    approved: bool
    reason: str
    proposal: dict | None = None


class ApprovalGate:
    """
    Human approval gate for trade decisions.
    
    When auto_execute is False, trade proposals are sent to a human
    operator via Telegram for approval before execution.
    """
    
    def __init__(self, 
                 approval_handler: TelegramApprovalHandler,
                 auto_execute: bool = False) -> None:
        self.approval_handler = approval_handler
        self.auto_execute = auto_execute
    
    def check_approval(self, proposal: dict, symbol: str) -> ApprovalResult:
        """
        Check if a trade proposal should be approved.
        
        Args:
            proposal: TradeProposal dict
            symbol: Trading symbol
            
        Returns:
            ApprovalResult with approved/rejected decision
        """
        if self.auto_execute:
            return ApprovalResult(
                approved=True,
                reason="auto_execute enabled",
                proposal=proposal
            )
        
        # Send to Telegram for human approval
        loop = asyncio.get_event_loop()
        decision = loop.run_until_complete(
            self.approval_handler.send_approval(proposal, symbol)
        )
        
        if decision == "approve":
            return ApprovalResult(
                approved=True,
                reason="human approved",
                proposal=proposal
            )
        else:
            return ApprovalResult(
                approved=False,
                reason="human rejected or timeout",
                proposal=proposal
            )
    
    def should_execute(self, proposal: dict, symbol: str) -> bool:
        """Convenience method: returns True if trade should execute."""
        result = self.check_approval(proposal, symbol)
        return result.approved