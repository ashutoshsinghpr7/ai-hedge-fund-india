def test_imports():
    from src.main import _ensure_suffix, _display_name, create_workflow  # noqa: F401
    assert _ensure_suffix("RELIANCE.NS") == "RELIANCE.NS"
    assert _ensure_suffix("TATASTEEL.BO") == "TATASTEEL.BO"
    assert _display_name("RELIANCE.NS") == "RELIANCE"
    assert _display_name("TATASTEEL.BO") == "TATASTEEL"


def test_ticker_stripping():
    from src.utils.reporter import ticker_ns
    assert ticker_ns("RELIANCE") == "RELIANCE.NS"
    assert ticker_ns("TATASTEEL.BO") == "TATASTEEL.BO"


def test_universe():
    from src.data.universe import get_available_sectors, get_universe
    sectors = get_available_sectors()
    assert "nifty50" in sectors
    assert "pharma" in sectors
    nifty = get_universe(universe_type="nifty50")
    assert len(nifty) >= 10
    assert any(t.endswith(".NS") for t in nifty)
