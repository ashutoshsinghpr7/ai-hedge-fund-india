"""Broker-facing data models.

These are the primitives a broker adapter produces. They deliberately stay free
of agent/analysis concerns: the bridge into the analysis layer lives in
`src.broker.portfolio`.
"""

from typing import Optional

from pydantic import BaseModel

# Internal (yfinance-style) suffix <-> Kite exchange mapping.
EXCHANGE_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}
SUFFIX_EXCHANGE = {suffix: exchange for exchange, suffix in EXCHANGE_SUFFIX.items()}


class InstrumentRef(BaseModel):
    """Exchange-level identity of a tradeable instrument."""

    exchange: str
    tradingsymbol: str
    instrument_token: int
    tick_size: float = 0.05
    lot_size: int = 1

    @property
    def key(self) -> str:
        """Kite quote key, e.g. ``NSE:RELIANCE``."""
        return f"{self.exchange}:{self.tradingsymbol}"

    @property
    def yf_ticker(self) -> str:
        """Ticker in the form the rest of this codebase uses, e.g. ``RELIANCE.NS``."""
        suffix = EXCHANGE_SUFFIX.get(self.exchange, "")
        return f"{self.tradingsymbol}{suffix}"


class BrokerPosition(BaseModel):
    """A single open position as reported by the broker.

    `quantity` is signed: positive is long, negative is short.
    """

    instrument: InstrumentRef
    quantity: int
    average_price: float
    last_price: Optional[float] = None
    product: str = ""
    source: str = "position"

    @property
    def ticker(self) -> str:
        return self.instrument.yf_ticker


class Funds(BaseModel):
    """Broker account funds summary."""

    cash: float
    margin_used: float = 0.0
    collateral: float = 0.0
