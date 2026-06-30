"""YFinance data provider for Indian NSE/BSE stocks.

Fetches price history, financial statements, and market data from Yahoo Finance.
Maps yfinance output to the project's Pydantic data models.
"""

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)
from src.data.providers.base import DataProvider

logger = logging.getLogger(__name__)

_FINANCIAL_FIELD_MAP = {
    "net_income": [
        "Net Income Common Stockholders",
        "Net Income",
        "Net Income From Continuing Ops",
    ],
    "earnings_per_share": [
        "Diluted EPS",
        "Basic EPS",
        "Diluted Earnings Per Share",
    ],
    "ebit": ["EBIT"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "operating_income": ["Operating Income", "Operating Revenue"],
    "revenue": ["Total Revenue", "Revenue"],
    "gross_profit": ["Gross Profit"],
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "shareholders_equity": ["Stockholders Equity", "Shareholders Equity", "Total Equity Gross Minority Interest"],
    "outstanding_shares": ["Ordinary Shares Number", "Share Issued"],
    "free_cash_flow": ["Free Cash Flow"],
    "capital_expenditure": ["Capital Expenditure", "Capital Expenditures"],
    "depreciation_and_amortization": ["Depreciation And Amortization", "Depreciation Amortization Depletion"],
    "dividends_and_other_cash_distributions": [
        "Cash Dividends Paid",
        "Common Stock Dividend",
    ],
    "issuance_or_purchase_of_equity_shares": [
        "Common Stock Issuance",
        "Repurchase Of Capital Stock",
        "Issuance Of Capital Stock",
    ],
    "operating_cash_flow": ["Operating Cash Flow"],
    "investing_cash_flow": ["Investing Cash Flow"],
    "financing_cash_flow": ["Financing Cash Flow"],
}


def _extract_value(financial_df: pd.DataFrame, field_name: str, col_idx: int = 0) -> Optional[float]:
    """Extract a value from a yfinance financial DataFrame using field name mapping."""
    if financial_df is None or financial_df.empty:
        return None

    candidates = _FINANCIAL_FIELD_MAP.get(field_name, [field_name])
    for candidate in candidates:
        if candidate in financial_df.index:
            val = financial_df.loc[candidate]
            if isinstance(val, pd.Series):
                val = val.iloc[col_idx] if col_idx < len(val) else None
            if val is not None and pd.notna(val):
                return float(val)
    return None


def _get_financial_periods(ticker: str) -> list[dict]:
    """Get available financial periods for a ticker.

    Returns list of dicts with keys: report_period, period (annual/quarterly)
    """
    periods = []
    try:
        stock = yf.Ticker(ticker)

        annual_financials = stock.financials
        if annual_financials is not None and not annual_financials.empty:
            for col in annual_financials.columns:
                date_str = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)[:10]
                periods.append({"report_period": date_str, "period": "annual"})

        quarterly_financials = stock.quarterly_financials
        if quarterly_financials is not None and not quarterly_financials.empty:
            for col in quarterly_financials.columns:
                date_str = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)[:10]
                if not any(p["report_period"] == date_str for p in periods):
                    periods.append({"report_period": date_str, "period": "quarterly"})

    except Exception as e:
        logger.warning("Failed to get financial periods for %s: %s", ticker, e)

    periods.sort(key=lambda x: x["report_period"], reverse=True)
    return periods


class YFinanceProvider(DataProvider):
    """Yahoo Finance data provider for NSE/BSE stocks.

    Uses yfinance library to fetch price and fundamental data.
    Tickers should use Yahoo Finance format (e.g., RELIANCE.NS, TCS.NS).
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir

    def _ensure_nse_suffix(self, ticker: str) -> str:
        """Ensure ticker has .NS or .BO suffix for Indian markets."""
        if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
            return f"{ticker}.NS"
        return ticker

    def get_prices(self, ticker: str, start_date: str, end_date: str) -> list[Price]:
        """Fetch OHLCV price history from yfinance."""
        symbol = self._ensure_nse_suffix(ticker)
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date, interval="1d", auto_adjust=False)

            if df.empty:
                logger.warning("No price data for %s (%s → %s)", symbol, start_date, end_date)
                return []

            prices = []
            for idx, row in df.iterrows():
                prices.append(Price(
                    open=float(row["Open"]),
                    close=float(row["Close"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    volume=int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
                    time=idx.strftime("%Y-%m-%d"),
                ))
            return prices

        except Exception as e:
            logger.error("Failed to fetch prices for %s: %s", symbol, e)
            return []

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[FinancialMetrics]:
        """Fetch financial metrics computed from yfinance data."""
        symbol = self._ensure_nse_suffix(ticker)
        metrics_list = []
        try:
            stock = yf.Ticker(symbol)

            annual_financials = stock.financials
            annual_balance = stock.balance_sheet
            annual_cashflow = stock.cashflow

            if annual_financials is None or annual_financials.empty:
                return []

            available_periods = annual_financials.columns

            for col_idx, col in enumerate(available_periods):
                report_date = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)[:10]

                if report_date > end_date:
                    continue

                net_income = _extract_value(annual_financials, "net_income", col_idx)
                revenue = _extract_value(annual_financials, "revenue", col_idx)
                operating_income = _extract_value(annual_financials, "operating_income", col_idx)
                gross_profit = _extract_value(annual_financials, "gross_profit", col_idx)

                total_assets = _extract_value(annual_balance, "total_assets", col_idx)
                total_liabilities = _extract_value(annual_balance, "total_liabilities", col_idx)
                shareholders_equity = _extract_value(annual_balance, "shareholders_equity", col_idx)
                current_assets = _extract_value(annual_balance, "current_assets", col_idx)
                current_liabilities = _extract_value(annual_balance, "current_liabilities", col_idx)

                free_cash_flow = _extract_value(annual_cashflow, "free_cash_flow", col_idx)

                market_cap = self.get_market_cap(ticker, report_date)

                gross_margin = (gross_profit / revenue) if (gross_profit and revenue and revenue != 0) else None
                operating_margin = (operating_income / revenue) if (operating_income and revenue and revenue != 0) else None
                net_margin = (net_income / revenue) if (net_income and revenue and revenue != 0) else None
                roe = (net_income / shareholders_equity) if (net_income and shareholders_equity and shareholders_equity != 0) else None
                roa = (net_income / total_assets) if (net_income and total_assets and total_assets != 0) else None
                debt_to_equity = (total_liabilities / shareholders_equity) if (total_liabilities and shareholders_equity and shareholders_equity != 0) else None
                current_ratio = (current_assets / current_liabilities) if (current_assets and current_liabilities and current_liabilities != 0) else None
                pe_ratio = (market_cap / net_income) if (market_cap and net_income and net_income != 0) else None
                pb_ratio = (market_cap / shareholders_equity) if (market_cap and shareholders_equity and shareholders_equity != 0) else None
                eps = _extract_value(annual_financials, "earnings_per_share", col_idx)

                metrics_list.append(FinancialMetrics(
                    ticker=ticker,
                    report_period=report_date,
                    period="annual",
                    currency="INR",
                    market_cap=market_cap,
                    price_to_earnings_ratio=pe_ratio,
                    price_to_book_ratio=pb_ratio,
                    gross_margin=gross_margin,
                    operating_margin=operating_margin,
                    net_margin=net_margin,
                    return_on_equity=roe,
                    return_on_assets=roa,
                    debt_to_equity=debt_to_equity,
                    current_ratio=current_ratio,
                    free_cash_flow_yield=(free_cash_flow / market_cap) if (free_cash_flow and market_cap and market_cap != 0) else None,
                    earnings_per_share=eps,
                ))

                if len(metrics_list) >= limit:
                    break

            return metrics_list

        except Exception as e:
            logger.error("Failed to fetch financial metrics for %s: %s", symbol, e)
            return []

    def search_line_items(
        self,
        ticker: str,
        line_items: list[str],
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[LineItem]:
        """Fetch financial statement line items from yfinance."""
        symbol = self._ensure_nse_suffix(ticker)
        results = []
        try:
            stock = yf.Ticker(symbol)

            annual_financials = stock.financials
            annual_balance = stock.balance_sheet
            annual_cashflow = stock.cashflow

            if annual_financials is None or annual_financials.empty:
                return []

            available_periods = annual_financials.columns

            for col_idx, col in enumerate(available_periods):
                report_date = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)[:10]

                if report_date > end_date:
                    continue

                line_item = LineItem(
                    ticker=ticker,
                    report_period=report_date,
                    period="annual",
                    currency="INR",
                )

                for field_name in line_items:
                    value = None
                    if field_name in ("total_assets", "current_assets", "current_liabilities",
                                      "total_liabilities", "shareholders_equity"):
                        value = _extract_value(annual_balance, field_name, col_idx)
                    elif field_name in ("free_cash_flow", "capital_expenditure",
                                        "depreciation_and_amortization", "dividends_and_other_cash_distributions",
                                        "issuance_or_purchase_of_equity_shares", "operating_cash_flow",
                                        "investing_cash_flow", "financing_cash_flow"):
                        value = _extract_value(annual_cashflow, field_name, col_idx)
                    else:
                        value = _extract_value(annual_financials, field_name, col_idx)

                    if hasattr(line_item, field_name) and value is not None:
                        setattr(line_item, field_name, value)

                results.append(line_item)

                if len(results) >= limit:
                    break

            return results

        except Exception as e:
            logger.error("Failed to fetch line items for %s: %s", symbol, e)
            return []

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        """Insider trade data not available from yfinance for Indian stocks."""
        logger.debug("Insider trades not available via yfinance for %s", ticker)
        return []

    def get_company_news(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[CompanyNews]:
        """Fetch company news from yfinance news endpoint."""
        symbol = self._ensure_nse_suffix(ticker)
        news_list = []
        try:
            stock = yf.Ticker(symbol)
            news_data = stock.news

            for item in news_data[:limit]:
                content = item.get("content", {})
                pub_date = content.get("pubDate", "")
                if start_date and pub_date and pub_date[:10] < start_date[:10]:
                    continue
                if end_date and pub_date and pub_date[:10] > end_date[:10]:
                    continue

                news_list.append(CompanyNews(
                    ticker=ticker,
                    title=content.get("title", ""),
                    author=content.get("provider", {}).get("displayName"),
                    source=content.get("publisher", ""),
                    date=pub_date or end_date,
                    url=content.get("canonicalUrl", {}).get("url"),
                    sentiment=None,
                ))

                if len(news_list) >= limit:
                    break

            return news_list

        except Exception as e:
            logger.warning("Failed to fetch news for %s: %s", symbol, e)
            return []

    def get_market_cap(self, ticker: str, end_date: str) -> Optional[float]:
        """Fetch market cap from yfinance info."""
        symbol = self._ensure_nse_suffix(ticker)
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            if info and "marketCap" in info:
                return float(info["marketCap"])
            return None
        except Exception as e:
            logger.warning("Failed to fetch market cap for %s: %s", symbol, e)
            return None
