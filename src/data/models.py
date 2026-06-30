"""Data models for the ai-hedge-fund-india system."""

from typing import Optional

from pydantic import BaseModel


class Price(BaseModel):
    open: float
    close: float
    high: float
    low: float
    volume: int
    time: str


class PriceResponse(BaseModel):
    prices: list[Price]


class FinancialMetrics(BaseModel):
    ticker: str
    report_period: str
    period: str
    currency: str
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    price_to_earnings_ratio: Optional[float] = None
    price_to_book_ratio: Optional[float] = None
    price_to_sales_ratio: Optional[float] = None
    enterprise_value_to_ebitda_ratio: Optional[float] = None
    enterprise_value_to_revenue_ratio: Optional[float] = None
    free_cash_flow_yield: Optional[float] = None
    peg_ratio: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    return_on_invested_capital: Optional[float] = None
    asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None
    receivables_turnover: Optional[float] = None
    days_sales_outstanding: Optional[float] = None
    operating_cycle: Optional[float] = None
    working_capital_turnover: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    cash_ratio: Optional[float] = None
    operating_cash_flow_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    interest_coverage: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    book_value_growth: Optional[float] = None
    earnings_per_share_growth: Optional[float] = None
    free_cash_flow_growth: Optional[float] = None
    operating_income_growth: Optional[float] = None
    ebitda_growth: Optional[float] = None
    payout_ratio: Optional[float] = None
    earnings_per_share: Optional[float] = None
    book_value_per_share: Optional[float] = None
    free_cash_flow_per_share: Optional[float] = None


class FinancialMetricsResponse(BaseModel):
    financial_metrics: list[FinancialMetrics]


class LineItem(BaseModel):
    ticker: str
    report_period: str
    period: str
    currency: str

    # Allow additional fields dynamically (matches original repo behavior)
    model_config = {"extra": "allow"}

    def __getattr__(self, name: str):
        if name in self.model_extra:
            return self.model_extra[name]
        try:
            return super().__getattr__(name)
        except AttributeError:
            return None


class LineItemResponse(BaseModel):
    search_results: list[LineItem]


class InsiderTrade(BaseModel):
    ticker: str
    issuer: Optional[str] = None
    name: Optional[str] = None
    title: Optional[str] = None
    is_board_director: Optional[bool] = None
    transaction_date: Optional[str] = None
    transaction_shares: Optional[float] = None
    transaction_price_per_share: Optional[float] = None
    transaction_value: Optional[float] = None
    shares_owned_before_transaction: Optional[float] = None
    shares_owned_after_transaction: Optional[float] = None
    security_title: Optional[str] = None
    filing_date: Optional[str] = None


class InsiderTradeResponse(BaseModel):
    insider_trades: list[InsiderTrade]


class CompanyNews(BaseModel):
    ticker: str
    title: str
    author: Optional[str] = None
    source: Optional[str] = None
    date: str
    url: Optional[str] = None
    sentiment: Optional[str] = None


class CompanyNewsResponse(BaseModel):
    news: list[CompanyNews]


class CompanyFacts(BaseModel):
    ticker: str
    name: Optional[str] = None
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None


class CompanyFactsResponse(BaseModel):
    company_facts: CompanyFacts
