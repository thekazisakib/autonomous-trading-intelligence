from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from data.store import FeatureSnapshot, PositionState


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NO_TRADE = "NO_TRADE"


@dataclass
class StrategySignal:
    action: Action
    symbol: str
    reason: str
    confidence: float = 1.0
    timestamp: datetime = datetime.now(timezone.utc)


class DeterministicStrategy:
    """
    Rule-based entry/exit. Configurable thresholds. No ML, no LLM in hot path.
    """

    def __init__(
        self,
        entry_rsi: float = 30.0,
        exit_rsi: float = 70.0,
        min_volume_surge_pct: float = 50.0,
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 4.0,
        max_holding_days: int = 5,
        thesis_weight: float = 0.1,
    ) -> None:
        self.entry_rsi = entry_rsi
        self.exit_rsi = exit_rsi
        self.min_volume_surge_pct = min_volume_surge_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_days = max_holding_days
        self.thesis_weight = thesis_weight

    def evaluate_entry(
        self,
        symbol: str,
        feature: FeatureSnapshot,
        position: PositionState | None,
        now: datetime,
    ) -> StrategySignal:
        """Return BUY if all conditions met, else NO_TRADE."""

        # Data freshness check
        if feature.data_freshness_seconds > 30:
            return StrategySignal(
                Action.NO_TRADE, symbol, "stale_data", confidence=0.0, timestamp=now
            )

        if feature.rsi_14 is None:
            return StrategySignal(
                Action.NO_TRADE, symbol, "no_rsi_feature", confidence=0.0, timestamp=now
            )

        # Position already open?
        if position and position.quantity > 0:
            return StrategySignal(
                Action.NO_TRADE, symbol, "position_already_open", confidence=0.0, timestamp=now
            )

        # RSI oversold + volume surge
        volume_ok = (
            feature.volume_surge_pct is not None
            and feature.volume_surge_pct > self.min_volume_surge_pct
        )
        if feature.rsi_14 < self.entry_rsi and volume_ok:
            return StrategySignal(
                Action.BUY,
                symbol,
                f"rsi={feature.rsi_14:.1f}<{self.entry_rsi}, volume_surge={feature.volume_surge_pct:.1f}%",
                timestamp=now,
            )

        return StrategySignal(Action.NO_TRADE, symbol, "no_entry_signal", timestamp=now)

    def evaluate_exit(
        self,
        symbol: str,
        feature: FeatureSnapshot,
        position: PositionState,
        entry_price: float,
        now: datetime,
    ) -> StrategySignal:
        """Return SELL if any exit condition met, else HOLD."""

        if not position or position.quantity <= 0:
            return StrategySignal(Action.NO_TRADE, symbol, "no_position", timestamp=now)

        ltp = feature.ltp if feature.ltp else 0.0

        # Stop loss
        stop_price = entry_price * (1 - self.stop_loss_pct / 100.0)
        if ltp <= stop_price:
            return StrategySignal(
                Action.SELL,
                symbol,
                f"stop_loss_hit ltp={ltp:.2f} <= {stop_price:.2f}",
                timestamp=now,
            )

        # Take profit
        target_price = entry_price * (1 + self.take_profit_pct / 100.0)
        if ltp >= target_price:
            return StrategySignal(
                Action.SELL,
                symbol,
                f"take_profit_hit ltp={ltp:.2f} >= {target_price:.2f}",
                timestamp=now,
            )

        # Max holding days
        if position.entry_timestamp:
            days_held = (now - position.entry_timestamp).days
            if days_held >= self.max_holding_days:
                return StrategySignal(
                    Action.SELL,
                    symbol,
                    f"max_holding_days {days_held} >= {self.max_holding_days}",
                    timestamp=now,
                )

        # RSI overbought exit
        if feature.rsi_14 is not None and feature.rsi_14 > self.exit_rsi:
            return StrategySignal(
                Action.SELL,
                symbol,
                f"rsi={feature.rsi_14:.1f} > {self.exit_rsi}",
                timestamp=now,
            )

        return StrategySignal(Action.HOLD, symbol, "holding", timestamp=now)