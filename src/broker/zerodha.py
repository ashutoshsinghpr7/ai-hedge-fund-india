"""Zerodha Kite Connect broker adapter."""

import logging
from datetime import datetime, timedelta
from typing import Optional

from kiteconnect import KiteConnect

from src.broker import auth
from src.broker.base import Broker
from src.broker.instruments import InstrumentNotFound, InstrumentRegistry
from src.broker.models import BrokerPosition, Funds, InstrumentRef

logger = logging.getLogger(__name__)

# Kite rejects day-candle requests spanning more than ~2000 days.
_MAX_DAYS_PER_REQUEST = 1900


class ZerodhaBroker(Broker):
    def __init__(self, client: KiteConnect, registry: Optional[InstrumentRegistry] = None):
        self._client = client
        self._registry = registry or InstrumentRegistry(client=client)
        self._profile: Optional[dict] = None

    @classmethod
    def from_env(cls, **kwargs) -> "ZerodhaBroker":
        """Build from KITE_API_KEY / KITE_ACCESS_TOKEN / .kite_session.json."""
        return cls(auth.create_client(**kwargs))

    @property
    def registry(self) -> InstrumentRegistry:
        return self._registry

    def get_profile(self) -> dict:
        if self._profile is None:
            self._profile = self._client.profile()
        return self._profile

    @staticmethod
    def _position_from_row(row: dict, source: str) -> BrokerPosition:
        instrument = InstrumentRef(
            exchange=row.get("exchange") or "NSE",
            tradingsymbol=row["tradingsymbol"],
            instrument_token=int(row.get("instrument_token") or 0),
        )
        return BrokerPosition(
            instrument=instrument,
            quantity=int(row.get("quantity") or 0),
            average_price=float(row.get("average_price") or 0.0),
            last_price=float(row.get("last_price") or 0.0),
            product=row.get("product") or "",
            source=source,
        )

    def get_holdings(self) -> list[BrokerPosition]:
        return [
            self._position_from_row(row, "holding")
            for row in self._client.holdings()
            if int(row.get("quantity") or 0) != 0
        ]

    def get_positions(self) -> list[BrokerPosition]:
        book = self._client.positions() or {}
        return [
            self._position_from_row(row, "position")
            for row in book.get("net", [])
            if int(row.get("quantity") or 0) != 0
        ]

    def get_funds(self) -> Funds:
        margins = self._client.margins() or {}
        equity = margins.get("equity") or {}
        available = (equity.get("available") or {}).get("cash", 0.0)
        utilised = (equity.get("utilised") or {}).get("total", 0.0)
        collateral = (equity.get("available") or {}).get("collateral", 0.0)
        return Funds(
            cash=float(available or 0.0),
            margin_used=float(utilised or 0.0),
            collateral=float(collateral or 0.0),
        )

    def get_ltp(self, tickers: list[str]) -> dict[str, float]:
        refs: dict[str, InstrumentRef] = {}
        for ticker in tickers:
            try:
                refs[ticker] = self.resolve(ticker)
            except InstrumentNotFound as exc:
                logger.warning("Skipping quote for %s: %s", ticker, exc)
        if not refs:
            return {}

        quotes = self._client.ltp(*[ref.key for ref in refs.values()])
        prices: dict[str, float] = {}
        for ticker, ref in refs.items():
            quote = quotes.get(ref.key) or {}
            if quote.get("last_price") is not None:
                prices[ticker] = float(quote["last_price"])
        return prices

    def resolve(self, ticker: str, exchange: Optional[str] = None) -> InstrumentRef:
        return self._registry.resolve(ticker, exchange=exchange)

    def historical_candles(
        self,
        instrument: InstrumentRef,
        start_date: str,
        end_date: str,
        interval: str = "day",
    ) -> list[dict]:
        """Raw OHLCV candles for one instrument, chunked to Kite's per-request limit."""
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        candles: list[dict] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=_MAX_DAYS_PER_REQUEST - 1), end)
            candles.extend(
                self._client.historical_data(
                    instrument.instrument_token,
                    cursor.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                    interval,
                )
            )
            cursor = chunk_end + timedelta(days=1)
        return candles

    def get_historical(self, ticker: str, start_date: str, end_date: str, interval: str = "day") -> list[dict]:
        """Candles for a ticker, or an empty list when the symbol is unmapped."""
        try:
            instrument = self.resolve(ticker)
        except InstrumentNotFound as exc:
            logger.warning("No historical data for %s: %s", ticker, exc)
            return []
        return self.historical_candles(instrument, start_date, end_date, interval)
