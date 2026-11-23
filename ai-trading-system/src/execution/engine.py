from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class ExecutionResult:
    order_id: str | None
    symbol: str
    side: str
    quantity: int
    status: OrderStatus
    filled_quantity: int = 0
    average_fill_price: float | None = None
    slippage_bps: float = 0.0
    timestamp: datetime = datetime.now(timezone.utc)
    error: str | None = None


class ExecutionEngine:
    """Order state machine. Never assumes success without broker confirmation."""

    def __init__(self, paper_trading: bool = True) -> None:
        self.paper_trading = paper_trading
        self._order_counter = 0

    def submit_order(self, symbol: str, side: str, quantity: int, price: float) -> ExecutionResult:
        self._order_counter += 1
        order_id = f"ORD-{self._order_counter:06d}"

        if self.paper_trading:
            # Simulate immediate fill at price (paper mode)
            return ExecutionResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                status=OrderStatus.FILLED,
                filled_quantity=quantity,
                average_fill_price=price,
                timestamp=datetime.now(timezone.utc),
            )

        # Live mode: real broker integration goes here.
        # Returns SUBMITTED and waits for fill confirmation.
        return ExecutionResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            status=OrderStatus.SUBMITTED,
            timestamp=datetime.now(timezone.utc),
        )

    def poll_status(self, order_id: str) -> ExecutionResult:
        """Poll broker for fill. In paper mode, returns cached result."""
        raise NotImplementedError("Implement broker-specific polling in Milestone 2")