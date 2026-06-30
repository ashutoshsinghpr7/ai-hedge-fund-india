"""Abstract base class for data providers."""

from abc import ABC, abstractmethod

import pandas as pd

from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)


class DataProvider(ABC):
    """Abstract data provider that all market-specific implementations must fulfill.

    Each method maps to the data layer consumed by the agent system.
    Implementations exist for yfinance (NSE/BSE), Financial Datasets (global),
    and any other data source.
    """

    @abstractmethod
    def get_prices(self, ticker: str, start_date: str, end_date: str) -> list[Price]:
        """Fetch OHLCV price history for a ticker over a date range."""

    @abstractmethod
    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[FinancialMetrics]:
        """Fetch financial metrics (PE, ROE, margins, etc.) for a ticker."""

    @abstractmethod
    def search_line_items(
        self,
        ticker: str,
        line_items: list[str],
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[LineItem]:
        """Fetch financial statement line items for a ticker."""

    @abstractmethod
    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        """Fetch insider trading activity for a ticker."""

    @abstractmethod
    def get_company_news(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[CompanyNews]:
        """Fetch company-specific news for a ticker."""

    @abstractmethod
    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        """Fetch current or historical market capitalization for a ticker."""

    def prices_to_df(self, prices: list[Price]) -> pd.DataFrame:
        """Convert a list of Price models to a pandas DataFrame."""
        df = pd.DataFrame([p.model_dump() for p in prices])
        df["Date"] = pd.to_datetime(df["time"])
        df.set_index("Date", inplace=True)
        numeric_cols = ["open", "close", "high", "low", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.sort_index(inplace=True)
        return df

    def get_price_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch price data and return as a DataFrame."""
        prices = self.get_prices(ticker, start_date, end_date)
        return self.prices_to_df(prices)
