"""Broker abstraction.

Only the read side is implemented so far: account, book, funds, quotes, and
historical candles. Order placement will extend this interface (see the roadmap
in the README) and must go through a guardrail layer, not straight from agents.
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.broker.models import BrokerPosition, Funds, InstrumentRef


class Broker(ABC):
    """Read-only view of a brokerage account."""

    @abstractmethod
    def get_profile(self) -> dict:
        """Authenticated account profile (user id, broker, enabled exchanges)."""

    @abstractmethod
    def get_holdings(self) -> list[BrokerPosition]:
        """Settled delivery holdings (T+1 book)."""

    @abstractmethod
    def get_positions(self) -> list[BrokerPosition]:
        """Net positions for the current day, including intraday."""

    @abstractmethod
    def get_funds(self) -> Funds:
        """Available cash and margin utilisation."""

    @abstractmethod
    def get_ltp(self, tickers: list[str]) -> dict[str, float]:
        """Last traded price per ticker; unresolved tickers are omitted."""

    @abstractmethod
    def resolve(self, ticker: str, exchange: Optional[str] = None) -> InstrumentRef:
        """Map an internal ticker to its exchange instrument."""
