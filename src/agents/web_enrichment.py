"""Web enrichment agent — fetches supplementary data when yfinance data is insufficient.

Runs as a LangGraph node BEFORE other analyst agents. For each ticker, checks if
yfinance has adequate price data; if not, scrapes Screener.in for fundamentals.
Results are stored in analyst_signals so other agents can use them as fallback.
"""

import json
import logging

import yfinance as yf
from langchain_core.messages import HumanMessage

from src.data.scrapers.screener import ScreenerScraper

logger = logging.getLogger(__name__)


def web_enrichment_agent(state):
    """Fetch supplementary fundamental data from web for tickers with sparse yfinance data."""
    agent_id = "web_enrichment_agent"
    data = state["data"]
    tickers = data["tickers"]

    if state.get("metadata", {}).get("no_web_fallback"):
        data["analyst_signals"][agent_id] = {t: {"status": "skipped", "reason": "no_web_fallback"} for t in tickers}
        return {"messages": state["messages"], "data": data}

    scraper = ScreenerScraper()
    enrichment: dict[str, dict] = {}

    for ticker in tickers:
        display = ticker.replace(".NS", "").replace(".BO", "")

        if _has_yfinance_data(ticker):
            enrichment[ticker] = {"status": "skipped", "reason": "yfinance_data_adequate"}
            continue

        logger.info("Enriching %s from web sources", display)
        try:
            result = scraper.fetch_financials(ticker)
            if result and result.get("ratios"):
                enrichment[ticker] = {"status": "success", **result}
                logger.info("Enriched %s: %d ratios from Screener.in", display, len(result.get("ratios", {})))
            else:
                enrichment[ticker] = {"status": "unavailable", "reason": "no_data_from_scraper"}
        except Exception as e:
            logger.warning("Web enrichment failed for %s: %s", ticker, e)
            enrichment[ticker] = {"status": "unavailable", "reason": str(e)[:200]}

    data["analyst_signals"][agent_id] = enrichment

    enriched = sum(1 for v in enrichment.values() if v.get("status") == "success")
    skipped = sum(1 for v in enrichment.values() if v.get("status") == "skipped")
    output = {"enriched": enriched, "skipped": skipped, "unavailable": len(tickers) - enriched - skipped}

    message = HumanMessage(content=json.dumps(output), name=agent_id)
    return {"messages": state["messages"] + [message], "data": data}


def _has_yfinance_data(ticker: str) -> bool:
    """Check if yfinance returns price data for this ticker."""
    try:
        df = yf.Ticker(ticker).history(period="1mo")
        if df.empty or df["Close"].dropna().empty:
            return False
        return True
    except Exception:
        return False
