from __future__ import annotations

from datetime import UTC, datetime, timedelta

from data.store import MarketTick, Store


def test_store_exposes_latest_live_quote_and_ignores_out_of_order_ticks() -> None:
    store = Store(":memory:")
    now = datetime.now(UTC)
    store.upsert_live_tick(MarketTick("RELIANCE", now, 101.0, volume=10))
    store.upsert_live_tick(
        MarketTick("RELIANCE", now - timedelta(seconds=2), 99.0, volume=8)
    )

    latest = store.get_latest_live_tick("RELIANCE")
    all_latest = store.get_latest_live_ticks()

    assert latest is not None
    assert latest.ltp == 101.0
    assert all_latest["RELIANCE"].ltp == 101.0
    store.close()
