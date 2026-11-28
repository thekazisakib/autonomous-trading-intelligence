from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from data.store import Store


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
    timestamp: datetime = dataclass_field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None
    estimated_cost: float = 0.0
    realized_pnl: float = 0.0


class ExecutionEngine:
    """Order state machine. Never assumes success without broker confirmation."""

    def __init__(
        self,
        paper_trading: bool = True,
        store: Store | None = None,
        initial_capital_inr: float = 500000.0,
        slippage_bps: float = 5.0,
        cost_bps: float = 20.0,
    ) -> None:
        self.paper_trading = paper_trading
        self.store = store
        self.slippage_bps = slippage_bps
        self.cost_bps = cost_bps
        self._order_counter = 0
        if not paper_trading:
            raise RuntimeError(
                "Live order submission is disabled; only the paper execution engine is implemented"
            )
        if store is not None:
            store.get_or_create_paper_account(initial_capital_inr)

    def submit_order(self, symbol: str, side: str, quantity: int, price: float) -> ExecutionResult:
        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:06d}-{uuid4().hex[:8]}"

        if side not in {"BUY", "SELL"} or quantity <= 0 or price <= 0:
            return ExecutionResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                status=OrderStatus.REJECTED,
                timestamp=datetime.now(timezone.utc),
                error="Invalid paper order side, quantity, or price",
            )

        direction = 1 if side == "BUY" else -1
        fill_price = price * (1 + direction * self.slippage_bps / 10_000)
        gross_value = fill_price * quantity
        estimated_cost = gross_value * self.cost_bps / 10_000
        realized_pnl = 0.0
        if self.store is not None:
            try:
                _, realized_pnl = self.store.apply_paper_fill(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    fill_price=fill_price,
                    estimated_cost=estimated_cost,
                    filled_at=datetime.now(timezone.utc),
                )
            except ValueError as exc:
                return ExecutionResult(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    status=OrderStatus.REJECTED,
                    timestamp=datetime.now(timezone.utc),
                    error=str(exc),
                )

        return ExecutionResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            average_fill_price=fill_price,
            timestamp=datetime.now(timezone.utc),
            estimated_cost=estimated_cost,
            realized_pnl=realized_pnl,
        )

    def poll_status(self, order_id: str) -> ExecutionResult:
        """Poll broker for fill. In paper mode, returns cached result."""
        raise NotImplementedError("Implement broker-specific polling in Milestone 2")