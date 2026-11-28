from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

NIFTY500_INDEX_CSV_URL = (
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
)
MIN_NIFTY500_SYMBOL_COUNT = 450
MAX_NIFTY500_SYMBOL_COUNT = 550
FRESH_CACHE_AGE = timedelta(hours=24)
MAX_STALE_CACHE_AGE = timedelta(days=7)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniverseSnapshot:
    symbols: tuple[str, ...]
    source: str
    fetched_at: datetime


def _parse_nifty500_csv(content: bytes) -> tuple[str, ...]:
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    symbol_column = next(
        (name for name in reader.fieldnames or () if name.strip().lower() == "symbol"),
        None,
    )
    if symbol_column is None:
        raise ValueError("NIFTY 500 constituent CSV has no Symbol column")

    symbols = tuple(
        str(row.get(symbol_column, "")).strip().upper()
        for row in reader
        if str(row.get(symbol_column, "")).strip()
    )
    unique_count = len(set(symbols))
    if (
        not MIN_NIFTY500_SYMBOL_COUNT <= len(symbols) <= MAX_NIFTY500_SYMBOL_COUNT
        or unique_count != len(symbols)
    ):
        raise ValueError(
            "NIFTY 500 constituent CSV must contain roughly 500 unique symbols "
            f"(received {len(symbols)} rows and {unique_count} unique symbols)"
        )
    return symbols


def _fetch_nifty500_csv() -> bytes:
    request = Request(
        NIFTY500_INDEX_CSV_URL,
        headers={"User-Agent": "autonomous-trading-intelligence/1.0"},
    )
    with urlopen(request, timeout=20) as response:
        return response.read()


def load_nifty500_universe(
    cache_path: Path,
    *,
    now: datetime | None = None,
) -> UniverseSnapshot:
    current_time = now or datetime.now(UTC)
    if current_time.tzinfo is None:
        raise ValueError("Universe cache time must be timezone-aware")
    current_time = current_time.astimezone(UTC)

    cached_content: bytes | None = None
    cache_age: timedelta | None = None
    if cache_path.exists():
        cache_age = current_time - datetime.fromtimestamp(
            cache_path.stat().st_mtime,
            tz=UTC,
        )
        try:
            cached_content = cache_path.read_bytes()
            cached_symbols = _parse_nifty500_csv(cached_content)
        except (OSError, UnicodeDecodeError, csv.Error, ValueError):
            cached_content = None
        else:
            if cache_age <= FRESH_CACHE_AGE:
                return UniverseSnapshot(cached_symbols, "cache", current_time - cache_age)

    try:
        downloaded_content = _fetch_nifty500_csv()
        downloaded_symbols = _parse_nifty500_csv(downloaded_content)
    except (OSError, URLError, TimeoutError, UnicodeDecodeError, csv.Error, ValueError) as exc:
        if cached_content is not None and cache_age is not None and cache_age <= MAX_STALE_CACHE_AGE:
            logger.warning(
                "using_stale_nifty500_constituents",
                extra={"cache_age_hours": cache_age.total_seconds() / 3600},
            )
            return UniverseSnapshot(
                _parse_nifty500_csv(cached_content),
                "stale_cache",
                current_time - cache_age,
            )
        raise RuntimeError(
            "Could not load the official NIFTY 500 constituents and no valid cache "
            f"younger than {MAX_STALE_CACHE_AGE.days} days is available: {exc}"
        ) from exc

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary_path.write_bytes(downloaded_content)
        temporary_path.replace(cache_path)
    except OSError as exc:
        logger.warning(
            "nifty500_cache_write_failed",
            extra={"path": str(cache_path), "error": str(exc)},
        )
    return UniverseSnapshot(downloaded_symbols, "download", current_time)
