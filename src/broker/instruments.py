"""Kite instrument master: maps internal tickers to exchange instruments.

The analysis layer speaks yfinance symbols (``RELIANCE.NS``, ``TATASTEEL.BO``)
while the broker speaks ``(exchange, tradingsymbol)`` pairs plus an
``instrument_token`` used by every quote and historical-data call. This module
owns that translation.

The instrument dump is a public endpoint, so resolution works even without a
valid access token.
"""

import logging
import os
import pickle
import re
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

from src.broker.models import SUFFIX_EXCHANGE, InstrumentRef

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"
CACHE_TTL_SECONDS = 24 * 3600
DEFAULT_EXCHANGES = ("NSE", "BSE")
CASH_SEGMENTS = {"NSE", "BSE"}

# Fields kept from the raw dump; the rest (strike, expiry, ...) is F&O noise here.
_KEPT_FIELDS = ("tradingsymbol", "instrument_token", "instrument_type", "segment", "tick_size", "lot_size")


class InstrumentNotFound(LookupError):
    """Raised when a ticker cannot be mapped to an exchange instrument."""


def _normalize(symbol: str) -> str:
    """Drop separators so ``M&M``/``M-M``/``M.M`` all collapse to one key."""
    return re.sub(r"[^A-Z0-9]", "", symbol.upper())


def split_ticker(ticker: str, exchange: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Split ``RELIANCE.NS`` into ``("RELIANCE", "NSE")``.

    An explicit exchange wins over the suffix. Returns ``(symbol, None)`` when
    no exchange can be inferred, in which case NSE is tried before BSE.
    """
    if exchange:
        return ticker.strip().upper(), exchange.upper()
    upper = ticker.strip().upper()
    for suffix, suffix_exchange in SUFFIX_EXCHANGE.items():
        if upper.endswith(suffix):
            return upper[: -len(suffix)], suffix_exchange
    return upper, None


def _default_loader(exchange: str) -> Iterable[dict]:
    from kiteconnect import KiteConnect

    # The instrument list is unauthenticated; the API key is only needed to
    # build a client object.
    client = KiteConnect(api_key=os.environ.get("KITE_API_KEY") or "public")
    return client.instruments(exchange)


class InstrumentRegistry:
    """Cached, lazily built symbol registry for NSE/BSE cash instruments."""

    def __init__(
        self,
        client=None,
        cache_dir: Optional[Path] = None,
        ttl_seconds: int = CACHE_TTL_SECONDS,
        loader: Optional[Callable[[str], Iterable[dict]]] = None,
    ):
        self._client = client
        self._cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self._ttl_seconds = ttl_seconds
        self._loader = loader or self._load_from_client
        self._rows: dict[str, list[dict]] = {}
        self._index: dict[str, dict[str, InstrumentRef]] = {}
        self._normalized: dict[str, dict[str, InstrumentRef]] = {}

    def _load_from_client(self, exchange: str) -> Iterable[dict]:
        if self._client is None:
            return _default_loader(exchange)
        return self._client.instruments(exchange)

    # ----- cache -----
    def _cache_path(self, exchange: str) -> Path:
        return self._cache_dir / f"kite_instruments_{exchange}.pkl"

    def _read_cache(self, exchange: str) -> Optional[list[dict]]:
        path = self._cache_path(exchange)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            if time.time() - payload.get("cached_at", 0) < self._ttl_seconds:
                return payload["rows"]
        except Exception as exc:  # corrupt/partial cache is not fatal
            logger.debug("Ignoring unreadable instrument cache %s: %s", path, exc)
        return None

    def _write_cache(self, exchange: str, rows: list[dict]) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path(exchange), "wb") as f:
                pickle.dump({"cached_at": time.time(), "rows": rows}, f)
        except OSError as exc:
            logger.debug("Could not write instrument cache for %s: %s", exchange, exc)

    def _rows_for(self, exchange: str) -> list[dict]:
        exchange = exchange.upper()
        if exchange in self._rows:
            return self._rows[exchange]
        rows = self._read_cache(exchange)
        if rows is None:
            try:
                raw = list(self._loader(exchange))
            except Exception as exc:
                raise InstrumentNotFound(
                    f"Could not download the {exchange} instrument list: {exc}"
                ) from exc
            rows = [
                {field: row.get(field) for field in _KEPT_FIELDS} for row in raw if row.get("tradingsymbol")
            ]
            self._write_cache(exchange, rows)
        self._rows[exchange] = rows
        return rows

    def _indexes(self, exchange: str) -> tuple[dict[str, InstrumentRef], dict[str, InstrumentRef]]:
        if exchange not in self._index:
            exact: dict[str, InstrumentRef] = {}
            normalized: dict[str, InstrumentRef] = {}
            for row in self._rows_for(exchange):
                if row.get("segment") not in CASH_SEGMENTS:
                    continue
                symbol = row["tradingsymbol"]
                ref = InstrumentRef(
                    exchange=exchange,
                    tradingsymbol=symbol,
                    instrument_token=int(row.get("instrument_token") or 0),
                    tick_size=float(row.get("tick_size") or 0.05),
                    lot_size=int(row.get("lot_size") or 1),
                )
                exact.setdefault(symbol.upper(), ref)
                normalized.setdefault(_normalize(symbol), ref)
            self._index[exchange] = exact
            self._normalized[exchange] = normalized
        return self._index[exchange], self._normalized[exchange]

    # ----- public API -----
    def resolve(self, ticker: str, exchange: Optional[str] = None) -> InstrumentRef:
        """Resolve a ticker (bare or suffixed) to an exchange instrument."""
        symbol, hinted = split_ticker(ticker, exchange)
        candidates = [hinted] if hinted else list(DEFAULT_EXCHANGES)

        for candidate in candidates:
            exact, normalized = self._indexes(candidate)
            if (ref := exact.get(symbol)) is not None:
                return ref
            if (ref := normalized.get(_normalize(symbol))) is not None:
                return ref

        tried = ", ".join(candidates)
        raise InstrumentNotFound(
            f"No {tried} cash-equity instrument found for '{ticker}'. "
            "Check the NSE/BSE symbol (e.g. RELIANCE, TATASTEEL.NS) or pass an explicit exchange."
        )

    def prime(self, exchanges: Iterable[str] = DEFAULT_EXCHANGES) -> None:
        """Warm the registry so the first ``resolve`` call is offline-cheap."""
        for exchange in exchanges:
            self._indexes(exchange.upper())
