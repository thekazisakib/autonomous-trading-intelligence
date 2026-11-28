from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta, timezone
from pathlib import Path

INDIA_TIMEZONE = timezone(timedelta(hours=5, minutes=30))


@dataclass
class MarketTick:
    symbol: str
    timestamp: datetime
    ltp: float
    bid: float | None = None
    ask: float | None = None
    bid_qty: int | None = None
    ask_qty: int | None = None
    volume: int = 0
    data_freshness_seconds: float = 0.0


@dataclass
class FeatureSnapshot:
    symbol: str
    timestamp: datetime
    rsi_14: float | None = None
    volume_surge_pct: float | None = None
    thesis_score: float | None = None
    ltp: float | None = None
    bid: float | None = None
    ask: float | None = None
    data_freshness_seconds: float = 0.0


@dataclass
class MarketCandle:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class PositionState:
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    entry_timestamp: datetime | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Store:
    def __init__(self, db_path: str = "data/trading.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def close(self) -> None:
        self._conn.close()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                rsi_14 REAL,
                volume_surge_pct REAL,
                thesis_score REAL,
                ltp REAL,
                bid REAL,
                ask REAL,
                data_freshness_seconds REAL,
                PRIMARY KEY (symbol, timestamp)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity INTEGER NOT NULL DEFAULT 0,
                avg_price REAL NOT NULL DEFAULT 0,
                entry_timestamp TEXT,
                updated_at TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS live_ticks (
                symbol TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                ltp REAL NOT NULL,
                bid REAL,
                ask REAL,
                volume INTEGER NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_account (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                initial_capital REAL NOT NULL,
                cash REAL NOT NULL,
                realized_pnl REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_fills (
                order_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                fill_price REAL NOT NULL,
                gross_value REAL NOT NULL,
                estimated_cost REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                filled_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_daily_equity (
                trading_date TEXT PRIMARY KEY,
                peak_equity REAL NOT NULL,
                latest_equity REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shadow_forecasts (
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                horizon_bars INTEGER NOT NULL,
                direction TEXT NOT NULL,
                expected_return_pct REAL NOT NULL,
                confidence REAL NOT NULL,
                model_version TEXT NOT NULL,
                PRIMARY KEY (symbol, timestamp, horizon_bars)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shadow_gemini_analyses (
                symbol TEXT NOT NULL,
                timestamp REAL NOT NULL,
                thesis_score REAL NOT NULL,
                confidence REAL NOT NULL,
                reasoning TEXT NOT NULL,
                model TEXT NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                symbol TEXT,
                payload TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS config_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                changed_at TEXT NOT NULL,
                changed_by TEXT DEFAULT 'system',
                config_key TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                reason TEXT,
                requires_restart BOOLEAN DEFAULT 1
            )
            """
        )
        self._conn.commit()

    def record_shadow_gemini_analysis(
        self,
        symbol: str,
        timestamp: float,
        thesis_score: float,
        confidence: float,
        reasoning: str,
        model: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO shadow_gemini_analyses
            (symbol, timestamp, thesis_score, confidence, reasoning, model)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (symbol, timestamp, thesis_score, confidence, reasoning, model),
        )
        self._conn.commit()

    def get_latest_shadow_gemini_analysis(self, symbol: str) -> dict | None:
        self._conn.row_factory = sqlite3.Row
        row = self._conn.execute(
            """
            SELECT * FROM shadow_gemini_analyses
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
        return dict(row) if row else None

    def record_shadow_forecast(
        self,
        symbol: str,
        timestamp: datetime,
        horizon_bars: int,
        direction: str,
        expected_return_pct: float,
        confidence: float,
        model_version: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO shadow_forecasts
            (symbol, timestamp, horizon_bars, direction, expected_return_pct,
             confidence, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                timestamp.isoformat(),
                horizon_bars,
                direction,
                expected_return_pct,
                confidence,
                model_version,
            ),
        )
        self._conn.commit()

    def get_shadow_forecasts(self, symbol: str, limit: int = 100) -> list[dict]:
        self._conn.row_factory = sqlite3.Row
        rows = self._conn.execute(
            """
            SELECT * FROM shadow_forecasts
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (symbol, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_or_create_paper_account(self, initial_capital: float) -> dict[str, float]:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO paper_account
            (id, initial_capital, cash, realized_pnl, updated_at)
            VALUES (1, ?, ?, 0, ?)
            """,
            (initial_capital, initial_capital, now),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT initial_capital, cash, realized_pnl FROM paper_account WHERE id = 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("Paper account could not be initialized")
        return {
            "initial_capital": float(row[0]),
            "cash": float(row[1]),
            "realized_pnl": float(row[2]),
        }

    def get_paper_session_stats(self, trading_date: date) -> dict[str, float | int]:
        day_start = datetime.combine(trading_date, time.min, tzinfo=INDIA_TIMEZONE)
        next_day = day_start + timedelta(days=1)
        rows = self._conn.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(realized_pnl), 0)
            FROM paper_fills
            WHERE filled_at >= ? AND filled_at < ?
            """,
            (
                day_start.astimezone(timezone.utc).isoformat(),
                next_day.astimezone(timezone.utc).isoformat(),
            ),
        ).fetchone()
        return {"orders": int(rows[0]), "realized_pnl": float(rows[1])}

    def get_all_positions(self) -> list[PositionState]:
        self._conn.row_factory = sqlite3.Row
        rows = self._conn.execute(
            "SELECT * FROM positions WHERE quantity > 0 ORDER BY symbol"
        ).fetchall()
        return [
            PositionState(
                symbol=row["symbol"],
                quantity=row["quantity"],
                avg_price=row["avg_price"],
                entry_timestamp=(
                    datetime.fromisoformat(row["entry_timestamp"])
                    if row["entry_timestamp"] else None
                ),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    def update_paper_equity_peak(
        self, trading_date: date, equity: float, timestamp: datetime
    ) -> tuple[float, float]:
        if equity < 0:
            raise ValueError("Paper equity cannot be negative")
        key = trading_date.isoformat()
        row = self._conn.execute(
            "SELECT peak_equity FROM paper_daily_equity WHERE trading_date = ?",
            (key,),
        ).fetchone()
        if row is None:
            self._conn.execute(
                """
                INSERT INTO paper_daily_equity
                (trading_date, peak_equity, latest_equity, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (key, equity, equity, timestamp.isoformat()),
            )
            self._conn.commit()
            peak = equity
        else:
            peak = float(row[0])
            if equity > peak:
                self._conn.execute(
                    """
                    UPDATE paper_daily_equity
                    SET peak_equity = ?, latest_equity = ?, updated_at = ?
                    WHERE trading_date = ?
                    """,
                    (equity, equity, timestamp.isoformat(), key),
                )
                self._conn.commit()
                peak = equity
        drawdown_pct = (equity - peak) / peak * 100 if peak > 0 else 0.0
        return peak, drawdown_pct

    def apply_paper_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        fill_price: float,
        estimated_cost: float,
        filled_at: datetime,
    ) -> tuple[float, float]:
        if side not in {"BUY", "SELL"} or quantity <= 0 or fill_price <= 0:
            raise ValueError("Paper fill must have a valid side, positive quantity, and price")

        gross_value = quantity * fill_price
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            account = self._conn.execute(
                "SELECT cash, realized_pnl FROM paper_account WHERE id = 1"
            ).fetchone()
            if account is None:
                raise RuntimeError("Paper account has not been initialized")
            position_row = self._conn.execute(
                "SELECT quantity, avg_price, entry_timestamp FROM positions WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            old_quantity = int(position_row[0]) if position_row else 0
            old_average = float(position_row[1]) if position_row else 0.0
            old_entry = position_row[2] if position_row else None
            cash = float(account[0])
            realized_pnl = 0.0

            if side == "BUY":
                debit = gross_value + estimated_cost
                if debit > cash:
                    raise ValueError("Insufficient paper cash for simulated buy")
                new_quantity = old_quantity + quantity
                new_average = (
                    old_average * old_quantity + debit
                ) / new_quantity
                new_entry = old_entry or filled_at.isoformat()
                cash -= debit
            else:
                if quantity > old_quantity:
                    raise ValueError("Cannot sell more shares than the paper position holds")
                proceeds = gross_value - estimated_cost
                realized_pnl = proceeds - old_average * quantity
                new_quantity = old_quantity - quantity
                new_average = old_average if new_quantity else 0.0
                new_entry = old_entry if new_quantity else None
                cash += proceeds

            new_realized_pnl = float(account[1]) + realized_pnl
            self._conn.execute(
                """
                INSERT OR REPLACE INTO positions
                (symbol, quantity, avg_price, entry_timestamp, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    new_quantity,
                    new_average,
                    new_entry,
                    filled_at.isoformat(),
                ),
            )
            self._conn.execute(
                """
                UPDATE paper_account SET cash = ?, realized_pnl = ?, updated_at = ?
                WHERE id = 1
                """,
                (cash, new_realized_pnl, filled_at.isoformat()),
            )
            self._conn.execute(
                """
                INSERT INTO paper_fills
                (order_id, symbol, side, quantity, fill_price, gross_value,
                 estimated_cost, realized_pnl, filled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    symbol,
                    side,
                    quantity,
                    fill_price,
                    gross_value,
                    estimated_cost,
                    realized_pnl,
                    filled_at.isoformat(),
                ),
            )
            self._conn.commit()
            return cash, realized_pnl
        except Exception:
            self._conn.rollback()
            raise

    def upsert_candle(self, candle: MarketCandle) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO candles
            (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candle.symbol,
                candle.timestamp.isoformat(),
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            ),
        )
        self._conn.commit()

    def upsert_live_tick(self, tick: MarketTick) -> None:
        timestamp = _as_utc(tick.timestamp).isoformat()
        self._conn.execute(
            """
            INSERT INTO live_ticks (symbol, timestamp, ltp, bid, ask, volume)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                timestamp = excluded.timestamp,
                ltp = excluded.ltp,
                bid = excluded.bid,
                ask = excluded.ask,
                volume = excluded.volume
            WHERE excluded.timestamp >= live_ticks.timestamp
            """,
            (tick.symbol, timestamp, tick.ltp, tick.bid, tick.ask, tick.volume),
        )
        self._conn.commit()

    def get_latest_live_tick(self, symbol: str) -> MarketTick | None:
        self._conn.row_factory = sqlite3.Row
        row = self._conn.execute(
            "SELECT * FROM live_ticks WHERE symbol = ?",
            (symbol,),
        ).fetchone()
        if row is None:
            return None
        timestamp = datetime.fromisoformat(row["timestamp"])
        freshness = max(
            (datetime.now(UTC) - _as_utc(timestamp)).total_seconds(),
            0.0,
        )
        return MarketTick(
            symbol=row["symbol"],
            timestamp=timestamp,
            ltp=row["ltp"],
            bid=row["bid"],
            ask=row["ask"],
            volume=row["volume"],
            data_freshness_seconds=freshness,
        )

    def get_latest_live_ticks(self) -> dict[str, MarketTick]:
        self._conn.row_factory = sqlite3.Row
        rows = self._conn.execute("SELECT * FROM live_ticks").fetchall()
        now = datetime.now(UTC)
        ticks: dict[str, MarketTick] = {}
        for row in rows:
            timestamp = datetime.fromisoformat(row["timestamp"])
            ticks[row["symbol"]] = MarketTick(
                symbol=row["symbol"],
                timestamp=timestamp,
                ltp=row["ltp"],
                bid=row["bid"],
                ask=row["ask"],
                volume=row["volume"],
                data_freshness_seconds=max(
                    (now - _as_utc(timestamp)).total_seconds(),
                    0.0,
                ),
            )
        return ticks

    def upsert_candles(self, candles: Sequence[MarketCandle]) -> None:
        if not candles:
            return
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO candles
            (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    candle.symbol,
                    candle.timestamp.isoformat(),
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                )
                for candle in candles
            ],
        )
        self._conn.commit()

    def get_recent_candles(self, symbol: str, limit: int = 100) -> list[MarketCandle]:
        self._conn.row_factory = sqlite3.Row
        rows = self._conn.execute(
            """
            SELECT * FROM candles
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (symbol, limit),
        ).fetchall()
        return [
            MarketCandle(
                symbol=row["symbol"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )
            for row in reversed(rows)
        ]

    def get_candles_between(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int = 100_000,
    ) -> list[MarketCandle]:
        if start >= end:
            raise ValueError("Candle range start must be before end")
        start_utc = _as_utc(start).isoformat()
        end_utc = _as_utc(end).isoformat()
        self._conn.row_factory = sqlite3.Row
        rows = self._conn.execute(
            """
            SELECT * FROM candles
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
            LIMIT ?
            """,
            (symbol, start_utc, end_utc, limit),
        ).fetchall()
        return [
            MarketCandle(
                symbol=row["symbol"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )
            for row in rows
        ]

    def get_shadow_forecasts_between(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int = 100_000,
    ) -> list[dict]:
        if start >= end:
            raise ValueError("Forecast range start must be before end")
        start_utc = _as_utc(start).isoformat()
        end_utc = _as_utc(end).isoformat()
        self._conn.row_factory = sqlite3.Row
        rows = self._conn.execute(
            """
            SELECT * FROM shadow_forecasts
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
            LIMIT ?
            """,
            (symbol, start_utc, end_utc, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def upsert_feature(self, feature: FeatureSnapshot) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO features
            (symbol, timestamp, rsi_14, volume_surge_pct, thesis_score, ltp, bid, ask, data_freshness_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feature.symbol,
                feature.timestamp.isoformat(),
                feature.rsi_14,
                feature.volume_surge_pct,
                feature.thesis_score,
                feature.ltp,
                feature.bid,
                feature.ask,
                feature.data_freshness_seconds,
            ),
        )
        self._conn.commit()

    def get_latest_feature(self, symbol: str) -> FeatureSnapshot | None:
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            """
            SELECT * FROM features
            WHERE symbol = ? AND rsi_14 IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (symbol,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return FeatureSnapshot(
            symbol=row["symbol"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            rsi_14=row["rsi_14"],
            volume_surge_pct=row["volume_surge_pct"],
            thesis_score=row["thesis_score"],
            ltp=row["ltp"],
            bid=row["bid"],
            ask=row["ask"],
            data_freshness_seconds=row["data_freshness_seconds"] or 0.0,
        )

    def upsert_position(self, position: PositionState) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO positions
            (symbol, quantity, avg_price, entry_timestamp, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                position.symbol,
                position.quantity,
                position.avg_price,
                position.entry_timestamp.isoformat() if position.entry_timestamp else None,
                position.updated_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_position(self, symbol: str) -> PositionState | None:
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT * FROM positions WHERE symbol = ?", (symbol,)
        )
        row = cur.fetchone()
        if not row:
            return None
        return PositionState(
            symbol=row["symbol"],
            quantity=row["quantity"],
            avg_price=row["avg_price"],
            entry_timestamp=datetime.fromisoformat(row["entry_timestamp"]) if row["entry_timestamp"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def write_thesis_score(self, symbol: str, score: float, timestamp: datetime) -> None:
        """Cold path (LangGraph/MCP) writes research signal here."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO features
            (symbol, timestamp, thesis_score, rsi_14, volume_surge_pct, ltp, bid, ask, data_freshness_seconds)
            VALUES (?, ?, ?, NULL, NULL, NULL, NULL, NULL, 0)
            """,
            (symbol, timestamp.isoformat(), score),
        )
        self._conn.commit()

    def audit(self, event_type: str, symbol: str | None, payload: object) -> None:
        self._conn.execute(
            "INSERT INTO audit_log (timestamp, event_type, symbol, payload) VALUES (?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                event_type,
                symbol,
                json.dumps(payload, default=str),
            ),
        )
        self._conn.commit()

    def log_config_change(
        self,
        config_key: str,
        old_value: object | None,
        new_value: object | None,
        reason: str = "",
        changed_by: str = "system",
        requires_restart: bool = True,
    ) -> None:
        """Log a YAML configuration change to config_audit table.

        This is an append-only audit trail. The actual config values
        still come from config/config.yaml; this table only records
        that a change was attempted or applied.
        """
        self._conn.execute(
            """
            INSERT INTO config_audit
            (changed_at, changed_by, config_key, old_value, new_value, reason, requires_restart)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                changed_by,
                config_key,
                json.dumps(old_value, default=str) if old_value is not None else None,
                json.dumps(new_value, default=str) if new_value is not None else None,
                reason,
                1 if requires_restart else 0,
            ),
        )
        self._conn.commit()

    def get_config_audit_history(self, limit: int = 50) -> list[dict]:
        """Retrieve recent config audit entries (newest first)."""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            """
            SELECT * FROM config_audit
            ORDER BY changed_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)