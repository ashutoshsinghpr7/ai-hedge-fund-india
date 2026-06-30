"""NSE/BSE stock universe definitions for ai-hedge-fund-india."""

import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
SECTOR_CACHE_FILE = CACHE_DIR / "sector_cache.pkl"

NIFTY50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", "HINDUNILVR.NS", "KOTAKBANK.NS",
    "LT.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS", "HCLTECH.NS",
    "NTPC.NS", "POWERGRID.NS", "ADANIPORTS.NS", "TECHM.NS", "M&M.NS",
    "NESTLEIND.NS", "ONGC.NS", "JSWSTEEL.NS", "ADANIENT.NS", "COALINDIA.NS",
    "BAJAJFINSV.NS", "GRASIM.NS", "DRREDDY.NS", "HDFCLIFE.NS",
    "BPCL.NS", "TATASTEEL.NS", "BRITANNIA.NS", "SBILIFE.NS", "EICHERMOT.NS",
    "CIPLA.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS", "HINDALCO.NS", "UPL.NS",
    "DIVISLAB.NS", "BAJAJ-AUTO.NS", "BEL.NS", "TRENT.NS", "APOLLOHOSP.NS",
]

NIFTY_NEXT50_SYMBOLS = [
    "VEDL.NS", "DMART.NS", "HAL.NS", "INDIGO.NS", "TATAPOWER.NS",
    "ZOMATO.NS", "DLF.NS", "SIEMENS.NS", "PIDILITIND.NS", "VBL.NS",
    "ADANIENERGY.NS", "TVSMOTOR.NS", "JIOFIN.NS", "INDHOTEL.NS", "MOTHERSON.NS",
    "ICICIPRULI.NS", "CANBK.NS", "COLPAL.NS", "PAGEIND.NS", "AMBUJACEM.NS",
    "LTIM.NS", "HAVELLS.NS", "MARICO.NS", "UNITDSPR.NS", "DABUR.NS",
    "LICHSGFIN.NS", "ABB.NS", "BIOCON.NS", "NAUKRI.NS", "POLICYBZR.NS",
    "IRFC.NS", "MUTHOOTFIN.NS", "SRTRANSFIN.NS", "GODREJCP.NS", "IRCTC.NS",
    "TORNTPHARM.NS", "JINDALSTEL.NS", "CHOLAFIN.NS", "BERGEPAINT.NS", "BANKBARODA.NS",
    "GAIL.NS", "PFC.NS", "RECLTD.NS", "TATACONSUM.NS", "BAJAJHLDNG.NS",
    "ADANIGREEN.NS", "YESBANK.NS", "PERSISTENT.NS", "TIINDIA.NS", "CUMMINSIND.NS",
]

# Static sector map for known stocks (manually curated)
SECTOR_MAP = {
    "RELIANCE.NS": "Energy",
    "TCS.NS": "Technology",
    "HDFCBANK.NS": "Banking",
    "INFY.NS": "Technology",
    "ICICIBANK.NS": "Banking",
    "BHARTIARTL.NS": "Telecom",
    "SBIN.NS": "Banking",
    "ITC.NS": "Consumer",
    "HINDUNILVR.NS": "Consumer",
    "KOTAKBANK.NS": "Banking",
    "LT.NS": "Construction",
    "BAJFINANCE.NS": "Financial Services",
    "ASIANPAINT.NS": "Consumer",
    "AXISBANK.NS": "Banking",
    "MARUTI.NS": "Auto",
    "SUNPHARMA.NS": "Pharma",
    "TITAN.NS": "Consumer",
    "WIPRO.NS": "Technology",
    "ULTRACEMCO.NS": "Cement",
    "HCLTECH.NS": "Technology",
    "NTPC.NS": "Energy",
    "POWERGRID.NS": "Energy",
    "ADANIPORTS.NS": "Infrastructure",
    "TECHM.NS": "Technology",
    "M&M.NS": "Auto",
    "NESTLEIND.NS": "Consumer",
    "ONGC.NS": "Energy",
    "JSWSTEEL.NS": "Metals",
    "ADANIENT.NS": "Energy",
    "COALINDIA.NS": "Energy",
    "BAJAJFINSV.NS": "Financial Services",
    "GRASIM.NS": "Cement",
    "DRREDDY.NS": "Pharma",
    "HDFCLIFE.NS": "Insurance",
    "BPCL.NS": "Energy",
    "TATASTEEL.NS": "Metals",
    "BRITANNIA.NS": "Consumer",
    "SBILIFE.NS": "Insurance",
    "EICHERMOT.NS": "Auto",
    "CIPLA.NS": "Pharma",
    "HEROMOTOCO.NS": "Auto",
    "INDUSINDBK.NS": "Banking",
    "HINDALCO.NS": "Metals",
    "UPL.NS": "Agriculture",
    "DIVISLAB.NS": "Pharma",
    "BAJAJ-AUTO.NS": "Auto",
    "BEL.NS": "Defence",
    "TRENT.NS": "Consumer",
    "APOLLOHOSP.NS": "Pharma",
    "VEDL.NS": "Metals",
    "DMART.NS": "Consumer",
    "HAL.NS": "Defence",
    "INDIGO.NS": "Aviation",
    "TATAPOWER.NS": "Energy",
    "ZOMATO.NS": "Technology",
    "DLF.NS": "Real Estate",
    "SIEMENS.NS": "Industrial",
    "PIDILITIND.NS": "Chemicals",
    "VBL.NS": "Consumer",
    "ADANIENERGY.NS": "Energy",
    "TVSMOTOR.NS": "Auto",
    "JIOFIN.NS": "Financial Services",
    "INDHOTEL.NS": "Hospitality",
    "MOTHERSON.NS": "Auto",
    "ICICIPRULI.NS": "Insurance",
    "CANBK.NS": "Banking",
    "COLPAL.NS": "Consumer",
    "PAGEIND.NS": "Consumer",
    "AMBUJACEM.NS": "Cement",
    "LTIM.NS": "Technology",
    "HAVELLS.NS": "Consumer",
    "MARICO.NS": "Consumer",
    "UNITDSPR.NS": "Consumer",
    "DABUR.NS": "Consumer",
    "LICHSGFIN.NS": "Financial Services",
    "ABB.NS": "Industrial",
    "BIOCON.NS": "Pharma",
    "NAUKRI.NS": "Technology",
    "POLICYBZR.NS": "Financial Services",
    "IRFC.NS": "Financial Services",
    "MUTHOOTFIN.NS": "Financial Services",
    "SRTRANSFIN.NS": "Financial Services",
    "GODREJCP.NS": "Consumer",
    "IRCTC.NS": "Tourism",
    "TORNTPHARM.NS": "Pharma",
    "JINDALSTEL.NS": "Metals",
    "CHOLAFIN.NS": "Financial Services",
    "BERGEPAINT.NS": "Consumer",
    "BANKBARODA.NS": "Banking",
    "GAIL.NS": "Energy",
    "PFC.NS": "Financial Services",
    "RECLTD.NS": "Financial Services",
    "TATACONSUM.NS": "Consumer",
    "BAJAJHLDNG.NS": "Financial Services",
    "ADANIGREEN.NS": "Energy",
    "YESBANK.NS": "Banking",
    "PERSISTENT.NS": "Technology",
    "TIINDIA.NS": "Industrial",
    "CUMMINSIND.NS": "Industrial",
}


def _load_sector_cache():
    """Load persisted sector cache from disk."""
    if SECTOR_CACHE_FILE.exists():
        try:
            with open(SECTOR_CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def _save_sector_cache(cache):
    """Persist sector cache to disk."""
    try:
        with open(SECTOR_CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
    except Exception as e:
        logger.warning("Failed to save sector cache: %s", e)


_sector_cache = _load_sector_cache()


def _fetch_sector_from_yfinance(symbol: str):
    """Fetch sector and industry from yfinance for a single symbol."""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        return {
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
    except Exception as e:
        logger.debug("yfinance sector lookup failed for %s: %s", symbol, e)
        return {"sector": None, "industry": None}


def discover_sectors(tickers, force_refresh=False):
    """Discover sectors for a list of tickers using yfinance, with caching."""
    global _sector_cache
    results = {}

    for sym in tickers:
        if sym in SECTOR_MAP:
            results[sym] = SECTOR_MAP[sym]
            continue

        if not force_refresh and sym in _sector_cache:
            cached = _sector_cache[sym]
            if cached.get("sector"):
                results[sym] = cached["sector"]
                continue

        info = _fetch_sector_from_yfinance(sym)
        _sector_cache[sym] = info
        if info.get("sector"):
            results[sym] = info["sector"]
        elif sym in SECTOR_MAP:
            results[sym] = SECTOR_MAP[sym]

    _save_sector_cache(_sector_cache)
    return results


def discover_stocks_by_sector(sector_name, candidate_tickers=None):
    """Find all stocks that belong to a given sector.

    Uses static SECTOR_MAP first, then falls back to yfinance discovery.
    If candidate_tickers is provided, only searches within that set.
    """
    if candidate_tickers is None:
        candidate_tickers = get_nifty100_symbols()

    normalized = sector_name.lower().replace("_", " ")

    # First: static SECTOR_MAP
    results = []
    for sym in candidate_tickers:
        if sym in SECTOR_MAP and SECTOR_MAP[sym].lower() == normalized:
            results.append(sym)

    # Then: yfinance discovery for remaining unknowns
    unknown = [s for s in candidate_tickers if s not in SECTOR_MAP and s not in results]
    if unknown:
        sectors = discover_sectors(unknown)
        for sym in candidate_tickers:
            if sym in sectors and sectors[sym].lower() == normalized and sym not in results:
                results.append(sym)

    return sorted(set(results))


def get_nifty50_symbols():
    return NIFTY50_SYMBOLS.copy()


def get_nifty100_symbols():
    return NIFTY50_SYMBOLS + NIFTY_NEXT50_SYMBOLS


def get_nifty500_symbols():
    return get_nifty100_symbols()


def get_sector(symbol):
    cached = _sector_cache.get(symbol, {})
    if cached.get("sector"):
        return cached["sector"]
    return SECTOR_MAP.get(symbol, "Others")


def get_all_sectors():
    return sorted(set(SECTOR_MAP.values()))


def get_available_sectors():
    """Returns all valid --sector values: indices + sector names."""
    indices = ["nifty50", "nifty100", "nifty500"]
    sectors = [s.lower().replace(" ", "_") for s in get_all_sectors()]
    return sorted(indices + sectors)


def get_sector_stocks(sector_name):
    """Return all stocks in a given sector (case-insensitive, underscore-friendly)."""
    return discover_stocks_by_sector(sector_name, get_nifty100_symbols())


def filter_universe(symbols, exclude_sectors=None, exclude_symbols=None):
    exclude_sectors = set(exclude_sectors or [])
    exclude_symbols = set(exclude_symbols or [])

    result = []
    for sym in symbols:
        if sym in exclude_symbols:
            continue
        if get_sector(sym) in exclude_sectors:
            continue
        result.append(sym)

    return result


def get_universe(universe_type="nifty50", custom_symbols=None, exclude_sectors=None, exclude_symbols=None, sector=None):
    symbols = []

    if sector:
        normalized = sector.lower().replace("_", " ")
        if normalized in ("nifty50", "nifty100", "nifty500"):
            symbols = get_universe(normalized, custom_symbols, exclude_sectors, exclude_symbols)
        else:
            symbols = get_sector_stocks(sector)
        return sorted(set(symbols))

    if universe_type == "nifty50":
        symbols = get_nifty50_symbols()
    elif universe_type == "nifty100":
        symbols = get_nifty100_symbols()
    elif universe_type == "nifty500":
        symbols = get_nifty500_symbols()
    else:
        symbols = []

    if custom_symbols:
        for sym in custom_symbols:
            if not sym.endswith(".NS") and not sym.endswith(".BO"):
                sym = f"{sym}.NS"
            if sym not in symbols:
                symbols.append(sym)

    if exclude_sectors or exclude_symbols:
        symbols = filter_universe(symbols, exclude_sectors, exclude_symbols)

    return sorted(symbols)
