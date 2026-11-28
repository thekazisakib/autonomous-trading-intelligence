from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from data.store import FeatureSnapshot, MarketCandle, MarketTick, Store

INDIA_TIMEZONE = timezone(timedelta(hours=5, minutes=30))


@dataclass
class _BuildingCandle:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0

    def update(self, price: float, volume_delta: int) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume_delta

    def as_candle(self) -> MarketCandle:
        return MarketCandle(
            symbol=self.symbol,
            timestamp=self.timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


class MarketFeatureEngine:
    """Build fixed-interval OHLCV candles and indicators from normalized ticks."""

    def __init__(
        self,
        store: Store,
        interval_minutes: int = 5,
        rsi_period: int = 14,
        volume_lookback: int = 20,
    ) -> None:
        if interval_minutes <= 0 or rsi_period <= 0 or volume_lookback <= 0:
            raise ValueError("Feature periods and candle interval must be positive")
        self.store = store
        self.interval_seconds = interval_minutes * 60
        self.rsi_period = rsi_period
        self.volume_lookback = volume_lookback
        self._building: dict[str, _BuildingCandle] = {}
        self._closes: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.rsi_period + 1)
        )
        self._volumes: dict[str, deque[int]] = defaultdict(
            lambda: deque(maxlen=self.volume_lookback)
        )
        self._cumulative_volume: dict[str, tuple[date, int]] = {}
        self._latest: dict[str, FeatureSnapshot] = {}
        self._loaded: set[str] = set()

    def update(self, tick: MarketTick) -> tuple[FeatureSnapshot | None, bool]:
        """Return the latest feature with current tick price and whether a bar closed."""
        self._load_history(tick.symbol)
        bucket = self._bucket_start(tick.timestamp)
        volume_delta = self._volume_delta(tick)
        current = self._building.get(tick.symbol)
        bar_closed = False

        if current is None:
            self._building[tick.symbol] = _BuildingCandle(
                symbol=tick.symbol,
                timestamp=bucket,
                open=tick.ltp,
                high=tick.ltp,
                low=tick.ltp,
                close=tick.ltp,
                volume=volume_delta,
            )
        elif bucket > current.timestamp:
            self._close_candle(current)
            bar_closed = True
            self._building[tick.symbol] = _BuildingCandle(
                symbol=tick.symbol,
                timestamp=bucket,
                open=tick.ltp,
                high=tick.ltp,
                low=tick.ltp,
                close=tick.ltp,
                volume=volume_delta,
            )
        elif bucket == current.timestamp:
            current.update(tick.ltp, volume_delta)

        feature = self._latest.get(tick.symbol)
        if feature is None:
            return None, bar_closed

        now = datetime.now(timezone.utc)
        freshness = max((now - tick.timestamp.astimezone(timezone.utc)).total_seconds(), 0.0)
        current_feature = FeatureSnapshot(
            symbol=feature.symbol,
            timestamp=feature.timestamp,
            rsi_14=feature.rsi_14,
            volume_surge_pct=feature.volume_surge_pct,
            thesis_score=feature.thesis_score,
            ltp=tick.ltp,
            bid=tick.bid,
            ask=tick.ask,
            data_freshness_seconds=freshness,
        )
        return current_feature, bar_closed

    def _load_history(self, symbol: str) -> None:
        if symbol in self._loaded:
            return
        candles = self.store.get_recent_candles(
            symbol, max(self.rsi_period + 1, self.volume_lookback + 1)
        )
        for candle in candles:
            self._record_closed_candle(candle, persist=False)
        latest = self.store.get_latest_feature(symbol)
        if latest is not None:
            self._latest[symbol] = latest
        self._loaded.add(symbol)

    def _bucket_start(self, timestamp: datetime) -> datetime:
        utc_timestamp = timestamp.astimezone(timezone.utc)
        bucket_epoch = int(utc_timestamp.timestamp()) // self.interval_seconds
        return datetime.fromtimestamp(
            bucket_epoch * self.interval_seconds, tz=timezone.utc
        )

    def _volume_delta(self, tick: MarketTick) -> int:
        local_date = tick.timestamp.astimezone(INDIA_TIMEZONE).date()
        previous = self._cumulative_volume.get(tick.symbol)
        if previous is None:
            delta = 0
        elif previous[0] != local_date:
            delta = max(tick.volume, 0)
        elif tick.volume >= previous[1]:
            delta = tick.volume - previous[1]
        else:
            delta = max(tick.volume, 0)
        self._cumulative_volume[tick.symbol] = (local_date, tick.volume)
        return delta

    def _close_candle(self, building: _BuildingCandle) -> None:
        candle = building.as_candle()
        self._record_closed_candle(candle, persist=True)

    def _record_closed_candle(self, candle: MarketCandle, persist: bool) -> None:
        closes = self._closes[candle.symbol]
        volumes = self._volumes[candle.symbol]
        previous_volumes = list(volumes)
        closes.append(candle.close)

        rsi = self._calculate_rsi(list(closes), self.rsi_period)
        volume_surge = None
        if previous_volumes:
            average_volume = sum(previous_volumes) / len(previous_volumes)
            if average_volume > 0:
                volume_surge = (candle.volume / average_volume - 1.0) * 100.0
        volumes.append(candle.volume)

        if persist:
            self.store.upsert_candle(candle)
        feature = FeatureSnapshot(
            symbol=candle.symbol,
            timestamp=candle.timestamp,
            rsi_14=rsi,
            volume_surge_pct=volume_surge,
            ltp=candle.close,
        )
        self._latest[candle.symbol] = feature
        if persist:
            self.store.upsert_feature(feature)

    @staticmethod
    def _calculate_rsi(closes: list[float], period: int) -> float | None:
        if len(closes) <= period:
            return None
        changes = [right - left for left, right in zip(closes[-period - 1:-1], closes[-period:])]
        gains = [change for change in changes if change > 0]
        losses = [-change for change in changes if change < 0]
        average_gain = sum(gains) / period
        average_loss = sum(losses) / period
        if average_loss == 0:
            return 100.0 if average_gain > 0 else 50.0
        relative_strength = average_gain / average_loss
        return 100.0 - 100.0 / (1.0 + relative_strength)
