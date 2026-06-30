from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.tools import get_financial_metrics, get_market_cap, search_line_items
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class RakeshJhunjhunwalaSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def rakesh_jhunjhunwala_agent(state: AgentState, agent_id: str = "rakesh_jhunjhunwala_agent"):
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analysis_data = {}
    jhunjhunwala_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        get_financial_metrics(ticker, end_date, period="ttm", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching financial line items")
        financial_line_items = search_line_items(
            ticker,
            ["net_income", "earnings_per_share", "ebit", "operating_income", "revenue",
             "operating_margin", "total_assets", "total_liabilities", "current_assets",
             "current_liabilities", "free_cash_flow", "dividends_and_other_cash_distributions",
             "issuance_or_purchase_of_equity_shares"],
            end_date, api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing growth")
        growth_analysis = analyze_growth(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing profitability")
        profitability_analysis = analyze_profitability(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing balance sheet")
        balancesheet_analysis = analyze_balance_sheet(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing cash flow")
        cashflow_analysis = analyze_cash_flow(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing management actions")
        management_analysis = analyze_management_actions(financial_line_items)

        progress.update_status(agent_id, ticker, "Calculating intrinsic value")
        intrinsic_value = calculate_intrinsic_value(financial_line_items, market_cap)

        total_score = (growth_analysis["score"] + profitability_analysis["score"] +
                       balancesheet_analysis["score"] + cashflow_analysis["score"] + management_analysis["score"])
        max_score = 24

        margin_of_safety = ((intrinsic_value - market_cap) / market_cap) if (intrinsic_value and market_cap) else None

        if margin_of_safety is not None and margin_of_safety >= 0.30:
            signal = "bullish"
        elif margin_of_safety is not None and margin_of_safety <= -0.30:
            signal = "bearish"
        else:
            quality_score = assess_quality_metrics(financial_line_items)
            if quality_score >= 0.7 and total_score >= max_score * 0.6:
                signal = "bullish"
            elif quality_score <= 0.4 or total_score <= max_score * 0.3:
                signal = "bearish"
            else:
                signal = "neutral"

        if margin_of_safety is not None:
            min(max(abs(margin_of_safety) * 150, 20), 95)
        else:
            min(max((total_score / max_score) * 100, 10), 80)

        analysis_data[ticker] = {
            "signal": signal, "score": total_score, "max_score": max_score,
            "margin_of_safety": margin_of_safety, "growth_analysis": growth_analysis,
            "profitability_analysis": profitability_analysis, "balancesheet_analysis": balancesheet_analysis,
            "cashflow_analysis": cashflow_analysis, "management_analysis": management_analysis,
            "intrinsic_value": intrinsic_value, "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating Jhunjhunwala analysis")
        jhunjhunwala_output = generate_jhunjhunwala_output(
            ticker=ticker, analysis_data=analysis_data[ticker], state=state, agent_id=agent_id)
        jhunjhunwala_analysis[ticker] = jhunjhunwala_output.model_dump()
        progress.update_status(agent_id, ticker, "Done", analysis=jhunjhunwala_output.reasoning)

    message = HumanMessage(content=json.dumps(jhunjhunwala_analysis), name=agent_id)
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(jhunjhunwala_analysis, "Rakesh Jhunjhunwala Agent")
    state["data"]["analyst_signals"][agent_id] = jhunjhunwala_analysis
    progress.update_status(agent_id, None, "Done")
    return {"messages": [message], "data": state["data"]}


def analyze_profitability(financial_line_items):
    if not financial_line_items:
        return {"score": 0, "details": "No profitability data"}
    latest = financial_line_items[0]
    score = 0
    reasoning = []
    if (getattr(latest, 'net_income', None) and latest.net_income > 0 and
        getattr(latest, 'total_assets', None) and getattr(latest, 'total_liabilities', None)
        and latest.total_assets and latest.total_liabilities):
        equity = latest.total_assets - latest.total_liabilities
        if equity > 0:
            roe = (latest.net_income / equity) * 100
            if roe > 20:
                score += 3
                reasoning.append(f"Excellent ROE: {roe:.1f}%")
            elif roe > 15:
                score += 2
                reasoning.append(f"Good ROE: {roe:.1f}%")
            elif roe > 10:
                score += 1
                reasoning.append(f"Decent ROE: {roe:.1f}%")
    if (getattr(latest, "operating_income", None) and latest.operating_income and
        getattr(latest, "revenue", None) and latest.revenue and latest.revenue > 0):
        op_margin = (latest.operating_income / latest.revenue) * 100
        if op_margin > 20:
            score += 2
            reasoning.append(f"Strong op margin: {op_margin:.1f}%")
        elif op_margin > 15:
            score += 1
            reasoning.append(f"Good op margin: {op_margin:.1f}%")
    eps_values = [getattr(item, "earnings_per_share", None) for item in financial_line_items
                  if getattr(item, "earnings_per_share", None) is not None and getattr(item, "earnings_per_share", None) > 0]
    if len(eps_values) >= 3:
        years = len(eps_values) - 1
        if eps_values[-1] > 0:
            eps_cagr = ((eps_values[0] / eps_values[-1]) ** (1/years) - 1) * 100
            if eps_cagr > 20:
                score += 3
                reasoning.append(f"High EPS CAGR: {eps_cagr:.1f}%")
            elif eps_cagr > 15:
                score += 2
                reasoning.append(f"Good EPS CAGR: {eps_cagr:.1f}%")
            elif eps_cagr > 10:
                score += 1
                reasoning.append(f"Moderate EPS CAGR: {eps_cagr:.1f}%")
    return {"score": score, "details": "; ".join(reasoning)}


def analyze_growth(financial_line_items):
    if len(financial_line_items) < 3:
        return {"score": 0, "details": "Insufficient data"}
    score = 0
    reasoning = []
    revenues = [getattr(item, "revenue", None) for item in financial_line_items
                if getattr(item, "revenue", None) is not None and getattr(item, "revenue", None) > 0]
    if len(revenues) >= 3:
        years = len(revenues) - 1
        if revenues[-1] > 0:
            revenue_cagr = ((revenues[0] / revenues[-1]) ** (1/years) - 1) * 100
            if revenue_cagr > 20:
                score += 3
                reasoning.append(f"Excellent revenue CAGR: {revenue_cagr:.1f}%")
            elif revenue_cagr > 15:
                score += 2
                reasoning.append(f"Good revenue CAGR: {revenue_cagr:.1f}%")
            elif revenue_cagr > 10:
                score += 1
                reasoning.append(f"Moderate revenue CAGR: {revenue_cagr:.1f}%")
    net_incomes = [getattr(item, "net_income", None) for item in financial_line_items
                   if getattr(item, "net_income", None) is not None and getattr(item, "net_income", None) > 0]
    if len(net_incomes) >= 3:
        years = len(net_incomes) - 1
        if net_incomes[-1] > 0:
            income_cagr = ((net_incomes[0] / net_incomes[-1]) ** (1/years) - 1) * 100
            if income_cagr > 25:
                score += 3
                reasoning.append(f"Excellent income CAGR: {income_cagr:.1f}%")
            elif income_cagr > 20:
                score += 2
                reasoning.append(f"High income CAGR: {income_cagr:.1f}%")
            elif income_cagr > 15:
                score += 1
                reasoning.append(f"Good income CAGR: {income_cagr:.1f}%")
    if len(revenues) >= 3:
        declining = sum(1 for i in range(1, len(revenues)) if revenues[i-1] > revenues[i])
        consistency = 1 - (declining / (len(revenues) - 1))
        if consistency >= 0.8:
            score += 1
            reasoning.append(f"Consistent growth ({consistency*100:.0f}%)")
    return {"score": score, "details": "; ".join(reasoning)}


def analyze_balance_sheet(financial_line_items):
    if not financial_line_items:
        return {"score": 0, "details": "No data"}
    latest = financial_line_items[0]
    score = 0
    reasoning = []
    if (getattr(latest, "total_assets", None) and getattr(latest, "total_liabilities", None)
        and latest.total_assets and latest.total_liabilities and latest.total_assets > 0):
        debt_ratio = latest.total_liabilities / latest.total_assets
        if debt_ratio < 0.5:
            score += 2
            reasoning.append(f"Low debt: {debt_ratio:.2f}")
        elif debt_ratio < 0.7:
            score += 1
            reasoning.append(f"Moderate debt: {debt_ratio:.2f}")
    if (getattr(latest, "current_assets", None) and getattr(latest, "current_liabilities", None)
        and latest.current_assets and latest.current_liabilities and latest.current_liabilities > 0):
        current_ratio = latest.current_assets / latest.current_liabilities
        if current_ratio > 2.0:
            score += 2
            reasoning.append(f"Strong liquidity: {current_ratio:.2f}")
        elif current_ratio > 1.5:
            score += 1
            reasoning.append(f"Good liquidity: {current_ratio:.2f}")
    return {"score": score, "details": "; ".join(reasoning)}


def analyze_cash_flow(financial_line_items):
    if not financial_line_items:
        return {"score": 0, "details": "No data"}
    latest = financial_line_items[0]
    score = 0
    reasoning = []
    if getattr(latest, "free_cash_flow", None) and latest.free_cash_flow:
        if latest.free_cash_flow > 0:
            score += 2
            reasoning.append("Positive FCF")
        else:
            reasoning.append("Negative FCF")
    if getattr(latest, "dividends_and_other_cash_distributions", None) and latest.dividends_and_other_cash_distributions:
        if latest.dividends_and_other_cash_distributions < 0:
            score += 1
            reasoning.append("Pays dividends")
    return {"score": score, "details": "; ".join(reasoning)}


def analyze_management_actions(financial_line_items):
    if not financial_line_items:
        return {"score": 0, "details": "No data"}
    latest = financial_line_items[0]
    score = 0
    reasoning = []
    issuance = getattr(latest, "issuance_or_purchase_of_equity_shares", None)
    if issuance is not None:
        if issuance < 0:
            score += 2
            reasoning.append(f"Buybacks: ₹{abs(issuance):,.0f}")
        elif issuance > 0:
            reasoning.append("Share dilution")
        else:
            score += 1
            reasoning.append("No dilution")
    return {"score": score, "details": "; ".join(reasoning)}


def assess_quality_metrics(financial_line_items):
    if not financial_line_items:
        return 0.5
    latest = financial_line_items[0]
    factors = []
    if (getattr(latest, 'net_income', None) and getattr(latest, 'total_assets', None) and
        getattr(latest, 'total_liabilities', None) and latest.total_assets and latest.total_liabilities):
        equity = latest.total_assets - latest.total_liabilities
        if equity > 0 and latest.net_income:
            roe = latest.net_income / equity
            if roe > 0.20:
                factors.append(1.0)
            elif roe > 0.15:
                factors.append(0.8)
            elif roe > 0.10:
                factors.append(0.6)
            else:
                factors.append(0.3)
        else:
            factors.append(0.0)
    else:
        factors.append(0.5)
    return sum(factors) / len(factors) if factors else 0.5


def calculate_intrinsic_value(financial_line_items, market_cap):
    if not financial_line_items or not market_cap:
        return None
    try:
        latest = financial_line_items[0]
        if not getattr(latest, 'net_income', None) or latest.net_income <= 0:
            return None
        net_incomes = [getattr(item, "net_income", None) for item in financial_line_items[:5]
                       if getattr(item, "net_income", None) is not None and getattr(item, "net_income", None) > 0]
        if len(net_incomes) < 2:
            return latest.net_income * 12
        years = len(net_incomes) - 1
        if net_incomes[-1] > 0:
            historical_growth = ((net_incomes[0] / net_incomes[-1]) ** (1/years) - 1)
        else:
            historical_growth = 0.05
        if historical_growth > 0.25:
            sustainable_growth = 0.20
        elif historical_growth > 0.15:
            sustainable_growth = historical_growth * 0.8
        elif historical_growth > 0.05:
            sustainable_growth = historical_growth * 0.9
        else:
            sustainable_growth = 0.05
        quality_score = assess_quality_metrics(financial_line_items)
        if quality_score >= 0.8:
            discount_rate, terminal_multiple = 0.12, 18
        elif quality_score >= 0.6:
            discount_rate, terminal_multiple = 0.15, 15
        else:
            discount_rate, terminal_multiple = 0.18, 12
        current_earnings = latest.net_income
        dcf_value = sum(current_earnings * ((1 + sustainable_growth) ** yr) / ((1 + discount_rate) ** yr) for yr in range(1, 6))
        year_5_earnings = current_earnings * ((1 + sustainable_growth) ** 5)
        terminal_value = (year_5_earnings * terminal_multiple) / ((1 + discount_rate) ** 5)
        return dcf_value + terminal_value
    except Exception:
        if getattr(latest, 'net_income', None) and latest.net_income > 0:
            return latest.net_income * 15
        return None


def generate_jhunjhunwala_output(ticker, analysis_data, state, agent_id):
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a Rakesh Jhunjhunwala AI agent. Decide on investment signals based on these principles:\n"
         "- Circle of Competence: Only invest in businesses you understand\n"
         "- Margin of Safety (> 30%): Buy at significant discount to intrinsic value\n"
         "- Economic Moat: Look for durable competitive advantages\n"
         "- Quality Management: Seek conservative, shareholder-oriented teams\n"
         "- Financial Strength: Favor low debt, strong returns on equity\n"
         "- Long-term Horizon: Invest in businesses, not just stocks\n"
         "- Growth Focus: Look for consistent earnings and revenue growth\n\n"
         "When providing your reasoning, be thorough and specific about the Indian market context. "
         "Use Rakesh Jhunjhunwala's voice and conversational style."),
        ("human", "Analysis Data for {ticker}:\n{analysis_data}\n\nReturn JSON: {{\"signal\":\"bullish\"|\"bearish\"|\"neutral\",\"confidence\":float 0-100,\"reasoning\":\"string\"}}"),
    ])
    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default():
        return RakeshJhunjhunwalaSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis")

    return call_llm(prompt=prompt, pydantic_model=RakeshJhunjhunwalaSignal, state=state, agent_name=agent_id, default_factory=create_default)
