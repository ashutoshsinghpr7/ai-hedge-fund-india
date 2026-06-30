"""Public API layer — delegates all data operations to the configured DataProvider.

This is the single interface used by all agents. Switching data sources
(e.g., yfinance → Zerodha Kite → Financial Datasets) only requires
changing one assignment in config or main.py.
"""

import logging
from typing import Optional

import pandas as pd

from src.data.cache import get_cache
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

_cache = get_cache()
_provider: Optional[DataProvider] = None


def get_provider() -> DataProvider:
    """Get the configured data provider (lazy init with default yfinance)."""
    global _provider
    if _provider is None:
        _provider = YFinanceProvider()
    return _provider


def set_provider(provider: DataProvider):
    """Set a custom data provider."""
    global _provider
    _provider = provider


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch price data with caching."""
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    provider = get_provider()
    prices = provider.get_prices(ticker, start_date, end_date)

    if prices:
        _cache.set_prices(cache_key, [p.model_dump() for p in prices])

    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics with caching."""
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    provider = get_provider()
    metrics = provider.get_financial_metrics(ticker, end_date, period, limit)

    if metrics:
        _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics])

    return metrics


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch financial statement line items."""
    provider = get_provider()
    return provider.search_line_items(ticker, line_items, end_date, period, limit)


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades with caching."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    provider = get_provider()
    trades = provider.get_insider_trades(ticker, end_date, start_date, limit)

    if trades:
        _cache.set_insider_trades(cache_key, [t.model_dump() for t in trades])

    return trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news with caching."""
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    provider = get_provider()
    news = provider.get_company_news(ticker, end_date, start_date, limit)

    if news:
        _cache.set_company_news(cache_key, [n.model_dump() for n in news])

    return news


def get_market_cap(ticker: str, end_date: str, api_key: str = None) -> Optional[float]:
    """Fetch market capitalization."""
    provider = get_provider()
    return provider.get_market_cap(ticker, end_date)


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    provider = get_provider()
    return provider.prices_to_df(prices)


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    """Fetch price data and return as a DataFrame."""
    provider = get_provider()
    return provider.get_price_data(ticker, start_date, end_date)
