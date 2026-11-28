from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class QualityDecision(str, Enum):
    PASS = "PASS"
    REJECT = "REJECT"
    HALT = "HALT"


@dataclass
class QualityResult:
    decision: QualityDecision
    symbol: str
    reasons: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CircuitLimits:
    """Upper and lower circuit prices for a symbol (NSE circuit filters)."""

    symbol: str
    upper_circuit: float | None = None
    lower_circuit: float | None = None
    previous_close: float | None = None
    updated_at: datetime | None = None


@dataclass
class CorporateAction:
    """Corporate action for a symbol (split, bonus, dividend)."""

    symbol: str
    action_type: str  # "SPLIT", "BONUS", "DIVIDEND"
    ratio: float | None = None
    dividend_per_share: float | None = None
    ex_date: datetime | None = None
    applied: bool = False


class QualityGate:
    """Data quality gate between broker WebSocket and strategy engine."""

    def __init__(self, config) -> None:
        self.enabled = config.enabled
        self.max_staleness_seconds = config.max_staleness_seconds
        self.check_circuit_filters = config.check_circuit_filters
        self.check_price_sanity = config.check_price_sanity
        self.check_volume_surge = config.check_volume_surge
        self.circuit_buffer_pct = config.circuit_buffer_pct

        self.circuit_limits: dict[str, CircuitLimits] = {}
        self.corporate_actions: dict[str, list[CorporateAction]] = {}
        self.halted_symbols: set[str] = set()
        self.previous_prices: dict[str, float] = {}

    def validate(self, symbol: str, ltp: float, bid: float | None,
                 ask: float | None, volume: int,
                 data_freshness_seconds: float,
                 timestamp: datetime) -> QualityResult:
        """Run all quality checks on a tick. Returns QualityResult."""

        reasons: list[str] = []

        if not self.enabled:
            return QualityResult(QualityDecision.PASS, symbol, reasons, timestamp)

        # --- 1. Data Staleness ---
        if data_freshness_seconds > self.max_staleness_seconds:
            reasons.append(f"stale_data {data_freshness_seconds:.1f}s > {self.max_staleness_seconds}s")
            self.halted_symbols.add(symbol)
            return QualityResult(QualityDecision.HALT, symbol, reasons, timestamp)

        self.halted_symbols.discard(symbol)

        # --- 2. Price Sanity ---
        if self.check_price_sanity:
            if ltp is None or ltp <= 0:
                reasons.append(f"invalid_price ltp={ltp}")
                return QualityResult(QualityDecision.REJECT, symbol, reasons, timestamp)

            if bid is not None and bid <= 0:
                reasons.append(f"invalid_bid bid={bid}")
                return QualityResult(QualityDecision.REJECT, symbol, reasons, timestamp)

            if ask is not None and ask <= 0:
                reasons.append(f"invalid_ask ask={ask}")
                return QualityResult(QualityDecision.REJECT, symbol, reasons, timestamp)

            if bid is not None and ask is not None and bid > ask:
                reasons.append(f"bid_ask_inverted bid={bid} > ask={ask}")
                return QualityResult(QualityDecision.REJECT, symbol, reasons, timestamp)

        # --- 3. Circuit Filter ---
        if self.check_circuit_filters:
            circuit_result = self._check_circuit(symbol, ltp, bid, ask)
            if circuit_result == "HALT_UPPER":
                reasons.append(f"upper_circuit_hit ltp={ltp}")
                self.halted_symbols.add(symbol)
                return QualityResult(QualityDecision.HALT, symbol, reasons, timestamp)
            elif circuit_result == "HALT_LOWER":
                reasons.append(f"lower_circuit_hit ltp={ltp}")
                self.halted_symbols.add(symbol)
                return QualityResult(QualityDecision.HALT, symbol, reasons, timestamp)
            elif circuit_result == "NEAR_UPPER":
                reasons.append(f"near_upper_circuit ltp={ltp}")
            elif circuit_result == "NEAR_LOWER":
                reasons.append(f"near_lower_circuit ltp={ltp}")

        self.previous_prices[symbol] = ltp
        return QualityResult(QualityDecision.PASS, symbol, reasons, timestamp)


    def _check_circuit(self, symbol: str, ltp: float, bid: float | None,
                       ask: float | None) -> str:
        circuit = self.circuit_limits.get(symbol)
        if not circuit or not circuit.upper_circuit or not circuit.lower_circuit:
            prev = self.previous_prices.get(symbol)
            if prev and prev > 0:
                upper = prev * 1.10
                lower = prev * 0.90
                buffer = self.circuit_buffer_pct / 100.0
                near_upper = prev * (1.10 - buffer)
                near_lower = prev * (0.90 + buffer)

                if ltp >= upper:
                    return "HALT_UPPER"
                elif ltp >= near_upper:
                    return "NEAR_UPPER"
                elif ltp <= lower:
                    return "HALT_LOWER"
                elif ltp <= near_lower:
                    return "NEAR_LOWER"
            return "OK"

        upper = circuit.upper_circuit
        lower = circuit.lower_circuit
        buffer = self.circuit_buffer_pct / 100.0

        near_upper = upper * (1 - buffer)
        near_lower = lower * (1 + buffer)

        if ltp >= upper:
            return "HALT_UPPER"
        elif ltp >= near_upper:
            return "NEAR_UPPER"
        elif ltp <= lower:
            return "HALT_LOWER"
        elif ltp <= near_lower:
            return "NEAR_LOWER"

        return "OK"

    def update_circuit_limits(self, symbol: str, upper: float | None,
                              lower: float | None, previous_close: float | None) -> None:
        self.circuit_limits[symbol] = CircuitLimits(
            symbol=symbol,
            upper_circuit=upper,
            lower_circuit=lower,
            previous_close=previous_close,
            updated_at=datetime.now(timezone.utc),
        )
        if previous_close:
            self.previous_prices[symbol] = previous_close

    def update_corporate_action(self, symbol: str, action_type: str,
                                ratio: float | None = None,
                                dividend_per_share: float | None = None,
                                ex_date: datetime | None = None) -> None:
        action = CorporateAction(
            symbol=symbol,
            action_type=action_type,
            ratio=ratio,
            dividend_per_share=dividend_per_share,
            ex_date=ex_date,
        )
        if symbol not in self.corporate_actions:
            self.corporate_actions[symbol] = []
        self.corporate_actions[symbol].append(action)

    def is_halted(self, symbol: str) -> bool:
        return symbol in self.halted_symbols

    def resume_symbol(self, symbol: str) -> None:
        self.halted_symbols.discard(symbol)