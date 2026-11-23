from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class RiskDecision(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"


@dataclass
class TradeProposal:
    symbol: str
    action: str  # "BUY" or "SELL"
    quantity: int
    price: float
    order_value_inr: float
    timestamp: datetime


@dataclass
class RiskResult:
    decision: RiskDecision
    reasons: list[str]
    modified_proposal: TradeProposal | None = None


class RiskEngine:
    """Deterministic veto. Can reject even if strategy says BUY."""

    def __init__(
        self,
        max_daily_loss_pct: float = 1.0,
        max_drawdown_pct: float = 5.0,
        max_position_per_stock_pct: float = 10.0,
        max_sector_exposure_pct: float = 30.0,
        max_order_value_inr: float = 50000.0,
        max_orders_per_day: int = 50,
    ) -> None:
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_per_stock_pct = max_position_per_stock_pct
        self.max_sector_exposure_pct = max_sector_exposure_pct
        self.max_order_value_inr = max_order_value_inr
        self.max_orders_per_day = max_orders_per_day

        self.daily_realized_loss_pct = 0.0
        self.drawdown_from_peak_pct = 0.0
        self.orders_today_count = 0
        self.current_position_pct = 0.0
        self.sector_exposure_pct = 0.0
        self.data_freshness_ok = True
        self.broker_connected = True

    def check(self, proposal: TradeProposal) -> RiskResult:
        reasons: list[str] = []

        # Fat-finger
        if proposal.order_value_inr > self.max_order_value_inr:
            reasons.append(
                f"order_value {proposal.order_value_inr:.0f} > limit {self.max_order_value_inr:.0f}"
            )

        # Daily order count
        if self.orders_today_count >= self.max_orders_per_day:
            reasons.append(f"daily order limit {self.max_orders_per_day} reached")

        # Daily loss circuit
        if self.daily_realized_loss_pct <= -self.max_daily_loss_pct:
            reasons.append(
                f"daily_loss {self.daily_realized_loss_pct:.2f}% breached limit {-self.max_daily_loss_pct:.2f}%"
            )

        # Drawdown circuit
        if self.drawdown_from_peak_pct <= -self.max_drawdown_pct:
            reasons.append(
                f"drawdown {self.drawdown_from_peak_pct:.2f}% breached limit {-self.max_drawdown_pct:.2f}%"
            )

        # Position concentration
        if self.current_position_pct >= self.max_position_per_stock_pct:
            reasons.append(
                f"position {self.current_position_pct:.1f}% >= limit {self.max_position_per_stock_pct:.1f}%"
            )

        # Data quality
        if not self.data_freshness_ok:
            reasons.append("market_data_stale")

        # Broker connectivity
        if not self.broker_connected:
            reasons.append("broker_disconnected")

        if reasons:
            return RiskResult(RiskDecision.REJECTED, reasons)

        return RiskResult(RiskDecision.APPROVED, [])