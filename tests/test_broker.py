"""Offline tests for the broker layer — no network, no Kite credentials."""

from datetime import date, datetime, timedelta

import pytest

from src.broker.base import Broker
from src.broker.instruments import InstrumentNotFound, InstrumentRegistry, split_ticker
from src.broker.models import BrokerPosition, Funds, InstrumentRef
from src.broker.portfolio import book_by_ticker, portfolio_from_broker

NSE_ROWS = [
    {
        "tradingsymbol": "RELIANCE",
        "instrument_token": 738561,
        "instrument_type": "EQ",
        "segment": "NSE",
        "tick_size": 0.05,
        "lot_size": 1,
    },
    {
        "tradingsymbol": "M&M",
        "instrument_token": 519937,
        "instrument_type": "EQ",
        "segment": "NSE",
        "tick_size": 0.05,
        "lot_size": 1,
    },
    {
        "tradingsymbol": "RELIANCE26JANFUT",
        "instrument_token": 999001,
        "instrument_type": "FUT",
        "segment": "NFO-FUT",
        "tick_size": 0.05,
        "lot_size": 250,
    },
]

BSE_ROWS = [
    {
        "tradingsymbol": "TATASTEEL",
        "instrument_token": 128668420,
        "instrument_type": "EQ",
        "segment": "BSE",
        "tick_size": 0.05,
        "lot_size": 1,
    },
]


def _registry(tmp_path) -> InstrumentRegistry:
    rows = {"NSE": NSE_ROWS, "BSE": BSE_ROWS}
    return InstrumentRegistry(cache_dir=tmp_path, loader=lambda exchange: rows[exchange])


def _ref(tradingsymbol: str, exchange: str = "NSE", token: int = 1) -> InstrumentRef:
    return InstrumentRef(exchange=exchange, tradingsymbol=tradingsymbol, instrument_token=token)


class FakeBroker(Broker):
    """Minimal in-memory broker used to exercise the adapter-agnostic helpers."""

    def __init__(self, positions=(), holdings=(), candles=None):
        self._positions = list(positions)
        self._holdings = list(holdings)
        self._candles = candles or []

    def get_profile(self):
        return {"user_id": "AB1234"}

    def get_holdings(self):
        return list(self._holdings)

    def get_positions(self):
        return list(self._positions)

    def get_funds(self):
        return Funds(cash=250000.0, margin_used=50000.0, collateral=10000.0)

    def get_ltp(self, tickers):
        return {}

    def resolve(self, ticker, exchange=None):
        return _ref(ticker.split(".")[0])

    def get_historical(self, ticker, start_date, end_date, interval="day"):
        return list(self._candles)


# ----- ticker resolution -----


def test_split_ticker():
    assert split_ticker("RELIANCE.NS") == ("RELIANCE", "NSE")
    assert split_ticker("tata steel.bo") == ("TATA STEEL", "BSE")
    assert split_ticker("RELIANCE") == ("RELIANCE", None)


def test_resolve_suffixed_and_bare(tmp_path):
    registry = _registry(tmp_path)
    assert registry.resolve("RELIANCE.NS").instrument_token == 738561
    assert registry.resolve("RELIANCE").instrument_token == 738561


def test_resolve_falls_back_to_bse(tmp_path):
    registry = _registry(tmp_path)
    assert registry.resolve("TATASTEEL.BO").exchange == "BSE"
    assert registry.resolve("TATASTEEL").exchange == "BSE"


def test_resolve_respects_suffix_exchange(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(InstrumentNotFound):
        registry.resolve("TATASTEEL.NS")


def test_resolve_normalizes_separators(tmp_path):
    registry = _registry(tmp_path)
    assert registry.resolve("M&M").tradingsymbol == "M&M"
    assert registry.resolve("MM").tradingsymbol == "M&M"


def test_registry_ignores_derivatives(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(InstrumentNotFound):
        registry.resolve("RELIANCE26JANFUT")


def test_registry_uses_cache(tmp_path):
    calls = []

    def loader(exchange):
        calls.append(exchange)
        return NSE_ROWS if exchange == "NSE" else BSE_ROWS

    InstrumentRegistry(cache_dir=tmp_path, loader=loader).resolve("RELIANCE.NS")
    InstrumentRegistry(cache_dir=tmp_path, loader=loader).resolve("RELIANCE.NS")
    assert calls == ["NSE"]


def test_unknown_ticker_raises(tmp_path):
    with pytest.raises(InstrumentNotFound):
        _registry(tmp_path).resolve("NOTAREALSYMBOL.NS")


def test_yf_ticker_mapping():
    assert _ref("RELIANCE", "NSE").yf_ticker == "RELIANCE.NS"
    assert _ref("TATASTEEL", "BSE").yf_ticker == "TATASTEEL.BO"


# ----- portfolio hydration -----


def test_book_prefers_net_positions_over_holdings():
    held = BrokerPosition(instrument=_ref("RELIANCE"), quantity=10, average_price=2500.0, source="holding")
    netted = BrokerPosition(instrument=_ref("RELIANCE"), quantity=4, average_price=2400.0, source="position")
    book = book_by_ticker([netted], [held])
    assert book["RELIANCE.NS"].source == "position"
    assert book["RELIANCE.NS"].quantity == 4


def test_book_keeps_holdings_absent_from_positions():
    held = BrokerPosition(instrument=_ref("TATASTEEL", "BSE"), quantity=7, average_price=150.0, source="holding")
    book = book_by_ticker([], [held])
    assert book["TATASTEEL.BO"].quantity == 7


def test_portfolio_from_broker_shape():
    broker = FakeBroker(
        positions=[BrokerPosition(instrument=_ref("RELIANCE"), quantity=10, average_price=2500.0)],
        holdings=[
            BrokerPosition(instrument=_ref("TATASTEEL", "BSE"), quantity=5, average_price=150.0),
            BrokerPosition(instrument=_ref("WIPRO"), quantity=-3, average_price=300.0),
        ],
    )

    portfolio = portfolio_from_broker(broker, tickers=["RELIANCE.NS", "INFY.NS"], margin_requirement=0.2)

    assert portfolio["cash"] == 250000.0
    assert portfolio["margin_used"] == 50000.0
    assert portfolio["margin_requirement"] == 0.2
    assert set(portfolio["positions"]) == {"RELIANCE.NS", "TATASTEEL.BO", "WIPRO.NS", "INFY.NS"}
    assert portfolio["realized_gains"]["INFY.NS"] == {"long": 0.0, "short": 0.0}

    reliance = portfolio["positions"]["RELIANCE.NS"]
    assert reliance == {
        "long": 10,
        "short": 0,
        "long_cost_basis": 2500.0,
        "short_cost_basis": 0.0,
        "short_margin_used": 0.0,
    }

    short = portfolio["positions"]["WIPRO.NS"]
    assert short["long"] == 0
    assert short["short"] == 3
    assert short["short_cost_basis"] == 300.0

    assert portfolio["positions"]["INFY.NS"]["long"] == 0


# ----- Zerodha price provider -----


def test_zerodha_provider_maps_candles():
    from src.data.providers.zerodha_provider import ZerodhaProvider

    broker = FakeBroker(
        candles=[
            {"date": datetime(2024, 3, 1), "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1000},
            {"date": date(2024, 3, 4), "open": 104.0, "high": 106.0, "low": 103.0, "close": 0.0, "volume": 0},
        ]
    )
    provider = ZerodhaProvider(broker)

    prices = provider.get_prices("RELIANCE.NS", "2024-03-01", "2024-03-05")

    assert len(prices) == 1
    assert prices[0].close == 104.0
    assert prices[0].volume == 1000
    assert prices[0].time == "2024-03-01T00:00:00"


def test_zerodha_provider_delegates_fundamentals():
    from src.data.providers.zerodha_provider import ZerodhaProvider

    class RecordingFallback:
        def __init__(self):
            self.calls = []

        def get_market_cap(self, ticker, end_date):
            self.calls.append((ticker, end_date))
            return 1234.5

        def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
            return []

    fallback = RecordingFallback()
    provider = ZerodhaProvider(FakeBroker(), fallback=fallback)

    assert provider.get_market_cap("RELIANCE.NS", "2024-03-05") == 1234.5
    assert fallback.calls == [("RELIANCE.NS", "2024-03-05")]
    assert provider.fallback is fallback


def test_zerodha_provider_empty_on_unmapped_symbol():
    from src.data.providers.zerodha_provider import ZerodhaProvider

    assert ZerodhaProvider(FakeBroker(candles=[])).get_prices("NOPE.NS", "2024-03-01", "2024-03-05") == []


def test_historical_chunks_long_ranges(tmp_path):
    """Kite rejects >2000-day requests, so the adapter splits them."""
    from src.broker.zerodha import ZerodhaBroker

    class RecordingClient:
        def __init__(self):
            self.requests = []

        def historical_data(self, token, start, end, interval):
            self.requests.append((token, start, end))
            return [{"date": start, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}]

    client = RecordingClient()
    broker = ZerodhaBroker(client, registry=_registry(tmp_path))

    end = (date(2017, 8, 1) + timedelta(days=2500)).isoformat()
    candles = broker.historical_candles(_ref("RELIANCE", token=738561), "2017-08-01", end)

    assert len(client.requests) == 2
    assert all(request[0] == 738561 for request in client.requests)
    assert len(candles) == 2


def test_price_cache_bypasses_same_day_ranges():
    """Live ranges must never be served from the pickle cache."""
    from src.tools import _cacheable

    assert _cacheable("2020-01-01") is True
    assert _cacheable(date.today().isoformat()) is False
    assert _cacheable("not-a-date") is True


# ----- auth helpers -----


def test_extract_request_token_accepts_url_or_token():
    from src.broker.auth import extract_request_token

    url = "https://127.0.0.1/?request_token=abcd1234&action=login&status=success"
    assert extract_request_token(url) == "abcd1234"
    assert extract_request_token("abcd1234") == "abcd1234"


def test_create_client_requires_api_key(monkeypatch, tmp_path):
    from src.broker.auth import KiteAuthRequired, create_client

    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
    with pytest.raises(KiteAuthRequired):
        create_client(session_path=tmp_path / "missing.json")


def test_create_client_requires_access_token(monkeypatch, tmp_path):
    from src.broker.auth import KiteAuthRequired, create_client

    monkeypatch.setenv("KITE_API_KEY", "key")
    monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
    with pytest.raises(KiteAuthRequired):
        create_client(session_path=tmp_path / "missing.json")


def test_create_client_reads_session_file(monkeypatch, tmp_path):
    from src.broker import auth

    monkeypatch.setenv("KITE_API_KEY", "key")
    monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
    session_path = tmp_path / "session.json"
    auth.save_session({"api_key": "key", "access_token": "token-123"}, session_path)

    client = auth.create_client(session_path=session_path)

    assert client.api_key == "key"
    assert client.access_token == "token-123"
