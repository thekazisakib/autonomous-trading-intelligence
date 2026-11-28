from __future__ import annotations

import csv
import io
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import URLError

import pytest

from config.config_schema import AppConfig
from config.loader import load_config, load_runtime_config
from data.universe import UniverseSnapshot, load_nifty500_universe


def _nifty500_csv(count: int = 500) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(("Company Name", "Industry", "Symbol"))
    for index in range(count):
        writer.writerow((f"Company {index}", "Industry", f"STOCK{index:03d}"))
    return output.getvalue().encode("utf-8")


def test_nifty500_download_is_validated_and_cached(tmp_path, monkeypatch) -> None:
    content = _nifty500_csv(count=501)
    monkeypatch.setattr("data.universe._fetch_nifty500_csv", lambda: content)
    cache_path = tmp_path / "nifty500.csv"

    snapshot = load_nifty500_universe(cache_path)

    assert len(snapshot.symbols) == 501
    assert snapshot.symbols[0] == "STOCK000"
    assert snapshot.source == "download"
    assert cache_path.read_bytes() == content

    def unexpected_fetch() -> bytes:
        raise AssertionError("fresh cache should avoid a network request")

    monkeypatch.setattr("data.universe._fetch_nifty500_csv", unexpected_fetch)
    cached = load_nifty500_universe(cache_path)
    assert cached.source == "cache"
    assert cached.symbols == snapshot.symbols


def test_nifty500_uses_recent_stale_cache_when_source_is_unavailable(
    tmp_path,
    monkeypatch,
) -> None:
    cache_path = tmp_path / "nifty500.csv"
    cache_path.write_bytes(_nifty500_csv())
    old_mtime = time.time() - timedelta(days=2).total_seconds()
    os.utime(cache_path, (old_mtime, old_mtime))
    monkeypatch.setattr(
        "data.universe._fetch_nifty500_csv",
        lambda: (_ for _ in ()).throw(URLError("offline")),
    )

    snapshot = load_nifty500_universe(cache_path, now=datetime.now(UTC))

    assert snapshot.source == "stale_cache"
    assert len(snapshot.symbols) == 500


def test_nifty500_fails_closed_without_valid_source_or_cache(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "data.universe._fetch_nifty500_csv",
        lambda: b"Symbol\nONLY_ONE\n",
    )

    with pytest.raises(RuntimeError, match="Could not load the official NIFTY 500"):
        load_nifty500_universe(tmp_path / "missing.csv")


def test_runtime_config_replaces_configured_symbols_with_index_members(
    tmp_path,
    monkeypatch,
) -> None:
    config = load_config()
    config_path = Path(tmp_path) / "config" / "config.yaml"
    snapshot = UniverseSnapshot(
        symbols=tuple(f"STOCK{index:03d}" for index in range(500)),
        source="download",
        fetched_at=datetime.now(UTC),
    )
    resolved_paths: list[Path] = []

    def fake_load_config(path: Path) -> AppConfig:
        assert path == config_path
        return config

    def fake_load_universe(cache_path: Path) -> UniverseSnapshot:
        resolved_paths.append(cache_path)
        return snapshot

    monkeypatch.setattr("config.loader.load_config", fake_load_config)
    monkeypatch.setattr("data.universe.load_nifty500_universe", fake_load_universe)

    runtime_config, resolved = load_runtime_config(config_path)

    assert runtime_config.trading.universe == list(snapshot.symbols)
    assert resolved == snapshot
    assert resolved_paths == [Path(tmp_path) / "data" / "nifty500_constituents.csv"]
