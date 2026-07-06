"""Web scraper base interface."""

from abc import ABC, abstractmethod


class Scraper(ABC):
    """Abstract web scraper for financial data."""

    @abstractmethod
    def fetch_financials(self, ticker: str) -> dict | None:
        """Fetch fundamental financial ratios. Returns dict or None on failure."""

    @abstractmethod
    def fetch_prices(self, ticker: str) -> list[dict] | None:
        """Fetch price history. Returns list of {date, open, high, low, close, volume} or None."""

    @abstractmethod
    def fetch_news(self, ticker: str) -> list[dict] | None:
        """Fetch recent news headlines. Returns list of {title, date, url, source} or None."""
