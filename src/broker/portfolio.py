"""Bridge from a broker's book into the portfolio dict the agents expect.

`AgentState["data"]["portfolio"]` has this shape (see `src/main.py`):

    {"cash": float, "margin_requirement": float, "margin_used": float,
     "positions": {ticker: {"long", "short", "long_cost_basis",
                            "short_cost_basis", "short_margin_used"}},
     "realized_gains": {ticker: {"long": float, "short": float}}}

Hydrating it from the broker is what makes the risk manager size against the
*real* book instead of a fabricated flat portfolio.
"""

from typing import Iterable, Sequence

from src.broker.base import Broker
from src.broker.models import BrokerPosition


def _empty_position() -> dict:
    return {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "short_margin_used": 0.0}


def _position_entry(position: BrokerPosition) -> dict:
    quantity = int(position.quantity)
    long_quantity = max(quantity, 0)
    short_quantity = max(-quantity, 0)
    return {
        "long": long_quantity,
        "short": short_quantity,
        "long_cost_basis": float(position.average_price) if long_quantity else 0.0,
        "short_cost_basis": float(position.average_price) if short_quantity else 0.0,
        "short_margin_used": 0.0,
    }


def book_by_ticker(
    positions: Iterable[BrokerPosition],
    holdings: Iterable[BrokerPosition],
) -> dict[str, BrokerPosition]:
    """Merge the two books, preferring net positions over settled holdings.

    Kite's net book already includes delivery holdings, so positions win to
    avoid double counting while holdings still cover anything the net book
    omitted.
    """
    book = {position.ticker: position for position in positions}
    for holding in holdings:
        book.setdefault(holding.ticker, holding)
    return book


def portfolio_from_broker(
    broker: Broker,
    tickers: Sequence[str] = (),
    margin_requirement: float = 0.0,
) -> dict:
    """Build the agent-facing portfolio dict from a live broker account."""
    book = book_by_ticker(broker.get_positions(), broker.get_holdings())
    funds = broker.get_funds()

    positions = {ticker: _position_entry(position) for ticker, position in book.items()}
    for ticker in tickers:
        positions.setdefault(ticker, _empty_position())

    return {
        "cash": float(funds.cash),
        "margin_requirement": float(margin_requirement),
        "margin_used": float(funds.margin_used),
        "positions": positions,
        "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in positions},
    }
