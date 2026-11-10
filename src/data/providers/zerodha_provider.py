"""Zerodha Kite Connect data provider.

Replaces yfinance for *price history* — minute-accurate, exchange-sourced OHLCV
for NSE/BSE instruments. Kite has no fundamentals, news, or insider data, so
those calls fall through to a fallback provider (yfinance by default).
"""

import logging
from datetime import date, datetime
from typing import Optional

from src.broker.zerodha import ZerodhaBroker
from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)
from src.data.providers.base import DataProvider
from src.data.providers.yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)


class ZerodhaProvider(DataProvider):
    """Price data from the broker, everything else from the fallback provider."""

    def __init__(self, broker: ZerodhaBroker, fallback: Optional[DataProvider] = None):
        self._broker = broker
        self._fallback = fallback

    @property
    def fallback(self) -> DataProvider:
        if self._fallback is None:
            self._fallback = YFinanceProvider()
        return self._fallback

    def get_prices(self, ticker: str, start_date: str, end_date: str) -> list[Price]:
        prices: list[Price] = []
        for candle in self._broker.get_historical(ticker, start_date, end_date, interval="day"):
            close = float(candle.get("close") or 0.0)
            if close <= 0:
                continue
            prices.append(
                Price(
                    open=float(candle.get("open") or 0.0),
                    close=close,
                    high=float(candle.get("high") or 0.0),
                    low=float(candle.get("low") or 0.0),
                    volume=int(candle.get("volume") or 0),
                    time=_iso_timestamp(candle.get("date")),
                )
            )
        return prices

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[FinancialMetrics]:
        return self.fallback.get_financial_metrics(ticker, end_date, period, limit)

    def search_line_items(
        self,
        ticker: str,
        line_items: list[str],
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[LineItem]:
        return self.fallback.search_line_items(ticker, line_items, end_date, period, limit)

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: Optional[str] = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        return self.fallback.get_insider_trades(ticker, end_date, start_date, limit)

    def get_company_news(
        self,
        ticker: str,
        end_date: str,
        start_date: Optional[str] = None,
        limit: int = 1000,
    ) -> list[CompanyNews]:
        return self.fallback.get_company_news(ticker, end_date, start_date, limit)

    def get_market_cap(self, ticker: str, end_date: str) -> Optional[float]:
        return self.fallback.get_market_cap(ticker, end_date)


def _iso_timestamp(value) -> str:
    """Normalise Kite's candle timestamp (datetime or string) to ISO-8601."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day).isoformat()
    return str(value)
