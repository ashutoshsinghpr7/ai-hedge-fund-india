from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import json
from typing_extensions import Literal
from src.tools import get_financial_metrics, get_market_cap, search_line_items
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


def warren_buffett_agent(state: AgentState, agent_id: str = "warren_buffett_agent"):
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analysis_data = {}
    buffett_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            ["capital_expenditure", "depreciation_and_amortization", "net_income", "outstanding_shares",
             "total_assets", "total_liabilities", "shareholders_equity", "dividends_and_other_cash_distributions",
             "issuance_or_purchase_of_equity_shares", "gross_profit", "revenue", "free_cash_flow"],
            end_date, period="ttm", limit=10, api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing fundamentals")
        fundamental_analysis = analyze_fundamentals(metrics)

        progress.update_status(agent_id, ticker, "Analyzing consistency")
        consistency_analysis = analyze_consistency(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing competitive moat")
        moat_analysis = analyze_moat(metrics)

        progress.update_status(agent_id, ticker, "Analyzing pricing power")
        pricing_power_analysis = analyze_pricing_power(financial_line_items, metrics)

        progress.update_status(agent_id, ticker, "Analyzing book value growth")
        book_value_analysis = analyze_book_value_growth(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing management quality")
        mgmt_analysis = analyze_management_quality(financial_line_items)

        progress.update_status(agent_id, ticker, "Calculating intrinsic value")
        intrinsic_value_analysis = calculate_intrinsic_value(financial_line_items)

        total_score = (fundamental_analysis["score"] + consistency_analysis["score"] +
                       moat_analysis["score"] + mgmt_analysis["score"] +
                       pricing_power_analysis["score"] + book_value_analysis["score"])

        max_possible_score = 10 + moat_analysis["max_score"] + mgmt_analysis["max_score"] + 5 + 5

        margin_of_safety = None
        intrinsic_value = intrinsic_value_analysis["intrinsic_value"]
        if intrinsic_value and market_cap:
            margin_of_safety = (intrinsic_value - market_cap) / market_cap

        analysis_data[ticker] = {
            "ticker": ticker, "score": total_score, "max_score": max_possible_score,
            "fundamental_analysis": fundamental_analysis, "consistency_analysis": consistency_analysis,
            "moat_analysis": moat_analysis, "pricing_power_analysis": pricing_power_analysis,
            "book_value_analysis": book_value_analysis, "management_analysis": mgmt_analysis,
            "intrinsic_value_analysis": intrinsic_value_analysis, "market_cap": market_cap,
            "margin_of_safety": margin_of_safety,
        }

        progress.update_status(agent_id, ticker, "Generating Warren Buffett analysis")
        buffett_output = generate_buffett_output(ticker=ticker, analysis_data=analysis_data[ticker],
                                                  state=state, agent_id=agent_id)
        buffett_analysis[ticker] = {"signal": buffett_output.signal, "confidence": buffett_output.confidence,
                                     "reasoning": buffett_output.reasoning}
        progress.update_status(agent_id, ticker, "Done", analysis=buffett_output.reasoning)

    message = HumanMessage(content=json.dumps(buffett_analysis), name=agent_id)
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(buffett_analysis, agent_id)
    state["data"]["analyst_signals"][agent_id] = buffett_analysis
    progress.update_status(agent_id, None, "Done")
    return {"messages": [message], "data": state["data"]}


def analyze_fundamentals(metrics):
    if not metrics:
        return {"score": 0, "details": "Insufficient fundamental data"}
    latest_metrics = metrics[0]
    score = 0
    reasoning = []
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:
        score += 2
        reasoning.append(f"Strong ROE of {latest_metrics.return_on_equity:.1%}")
    elif latest_metrics.return_on_equity:
        reasoning.append(f"Weak ROE of {latest_metrics.return_on_equity:.1%}")
    if latest_metrics.debt_to_equity and latest_metrics.debt_to_equity < 0.5:
        score += 2
        reasoning.append("Conservative debt levels")
    elif latest_metrics.debt_to_equity:
        reasoning.append(f"High debt to equity of {latest_metrics.debt_to_equity:.1f}")
    if latest_metrics.operating_margin and latest_metrics.operating_margin > 0.15:
        score += 2
        reasoning.append("Strong operating margins")
    elif latest_metrics.operating_margin:
        reasoning.append(f"Weak operating margin of {latest_metrics.operating_margin:.1%}")
    if latest_metrics.current_ratio and latest_metrics.current_ratio > 1.5:
        score += 1
        reasoning.append("Good liquidity position")
    return {"score": score, "details": "; ".join(reasoning), "metrics": latest_metrics.model_dump()}


def analyze_consistency(financial_line_items):
    if len(financial_line_items) < 4:
        return {"score": 0, "details": "Insufficient historical data"}
    score = 0
    reasoning = []
    earnings_values = [item.net_income for item in financial_line_items if item.net_income]
    if len(earnings_values) >= 4:
        earnings_growth_found = all(earnings_values[i] > earnings_values[i + 1] for i in range(len(earnings_values) - 1))
        if earnings_growth_found:
            score += 3
            reasoning.append("Consistent earnings growth")
        else:
            reasoning.append("Inconsistent earnings pattern")
    return {"score": score, "details": "; ".join(reasoning)}


def analyze_moat(metrics, max_score=5):
    if not metrics or len(metrics) < 5:
        return {"score": 0, "max_score": max_score, "details": "Insufficient data"}
    reasoning = []
    moat_score = 0
    historical_roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    if len(historical_roes) >= 5:
        high_roe_periods = sum(1 for roe in historical_roes if roe > 0.15)
        roe_consistency = high_roe_periods / len(historical_roes)
        if roe_consistency >= 0.8:
            moat_score += 2
            reasoning.append(f"Excellent ROE consistency: {high_roe_periods}/{len(historical_roes)} periods")
        elif roe_consistency >= 0.6:
            moat_score += 1
            reasoning.append(f"Good ROE: {high_roe_periods}/{len(historical_roes)} periods")
    historical_margins = [m.operating_margin for m in metrics if m.operating_margin is not None]
    if len(historical_margins) >= 5:
        avg_margin = sum(historical_margins) / len(historical_margins)
        if avg_margin > 0.2:
            moat_score += 1
            reasoning.append(f"Strong margins ({avg_margin:.1%}) indicate pricing power")
    return {"score": min(moat_score, max_score), "max_score": max_score, "details": "; ".join(reasoning)}


def analyze_management_quality(financial_line_items):
    if not financial_line_items:
        return {"score": 0, "max_score": 2, "details": "Insufficient data"}
    reasoning = []
    mgmt_score = 0
    latest = financial_line_items[0]
    if hasattr(latest, "issuance_or_purchase_of_equity_shares") and latest.issuance_or_purchase_of_equity_shares and latest.issuance_or_purchase_of_equity_shares < 0:
        mgmt_score += 1
        reasoning.append("Share buybacks (shareholder-friendly)")
    if hasattr(latest, "dividends_and_other_cash_distributions") and latest.dividends_and_other_cash_distributions and latest.dividends_and_other_cash_distributions < 0:
        mgmt_score += 1
        reasoning.append("Pays dividends")
    return {"score": mgmt_score, "max_score": 2, "details": "; ".join(reasoning)}


def calculate_owner_earnings(financial_line_items):
    if not financial_line_items or len(financial_line_items) < 2:
        return {"owner_earnings": None, "details": ["Insufficient data"]}
    latest = financial_line_items[0]
    net_income = latest.net_income
    depreciation = latest.depreciation_and_amortization
    capex = latest.capital_expenditure
    if not all([net_income, depreciation, capex]):
        return {"owner_earnings": None, "details": ["Missing components"]}
    maintenance_capex = abs(capex) * 0.85
    owner_earnings = net_income + depreciation - maintenance_capex
    return {"owner_earnings": owner_earnings, "details": [f"Net income: ₹{net_income:,.0f}", f"Depreciation: ₹{depreciation:,.0f}", f"Maint. capex: ₹{maintenance_capex:,.0f}", f"Owner earnings: ₹{owner_earnings:,.0f}"]}


def calculate_intrinsic_value(financial_line_items):
    if not financial_line_items or len(financial_line_items) < 3:
        return {"intrinsic_value": None, "details": ["Insufficient data"]}
    earnings_data = calculate_owner_earnings(financial_line_items)
    if not earnings_data["owner_earnings"]:
        return {"intrinsic_value": None, "details": earnings_data["details"]}
    owner_earnings = earnings_data["owner_earnings"]
    latest = financial_line_items[0]
    shares_outstanding = latest.outstanding_shares
    if not shares_outstanding or shares_outstanding <= 0:
        return {"intrinsic_value": None, "details": ["Missing shares"]}
    growth_rate = 0.05
    discount_rate = 0.10
    pv = sum(owner_earnings * (1 + growth_rate) ** yr / (1 + discount_rate) ** yr for yr in range(1, 6))
    terminal = owner_earnings * (1 + growth_rate) ** 5 * 1.025 / (discount_rate - 0.025)
    pv_terminal = terminal / (1 + discount_rate) ** 5
    return {"intrinsic_value": (pv + pv_terminal) * 0.85, "owner_earnings": owner_earnings, "details": [f"PV: ₹{pv:,.0f}", f"Terminal: ₹{pv_terminal:,.0f}"]}


def analyze_book_value_growth(financial_line_items):
    if len(financial_line_items) < 3:
        return {"score": 0, "details": "Insufficient data"}
    book_values = [item.shareholders_equity / item.outstanding_shares for item in financial_line_items
                   if hasattr(item, 'shareholders_equity') and hasattr(item, 'outstanding_shares')
                   and item.shareholders_equity and item.outstanding_shares]
    if len(book_values) < 3:
        return {"score": 0, "details": "Insufficient book value data"}
    growth_periods = sum(1 for i in range(len(book_values) - 1) if book_values[i] > book_values[i + 1])
    growth_rate = growth_periods / (len(book_values) - 1)
    score = 0
    reasoning = []
    if growth_rate >= 0.8:
        score += 3
        reasoning.append("Consistent book value growth")
    elif growth_rate >= 0.6:
        score += 2
        reasoning.append("Good book value growth")
    elif growth_rate >= 0.4:
        score += 1
        reasoning.append("Moderate book value growth")
    return {"score": score, "details": "; ".join(reasoning)}


def analyze_pricing_power(financial_line_items, metrics):
    if not financial_line_items or not metrics:
        return {"score": 0, "details": "Insufficient data"}
    score = 0
    reasoning = []
    gross_margins = [item.gross_margin for item in financial_line_items if hasattr(item, 'gross_margin') and item.gross_margin is not None]
    if len(gross_margins) >= 3:
        recent_avg = sum(gross_margins[:2]) / 2 if len(gross_margins) >= 2 else gross_margins[0]
        older_avg = sum(gross_margins[-2:]) / 2 if len(gross_margins) >= 2 else gross_margins[-1]
        if recent_avg > older_avg + 0.02:
            score += 3
            reasoning.append("Expanding margins → strong pricing power")
        elif recent_avg > older_avg:
            score += 2
            reasoning.append("Improving margins")
    if gross_margins:
        avg_margin = sum(gross_margins) / len(gross_margins)
        if avg_margin > 0.5:
            score += 2
    return {"score": score, "details": "; ".join(reasoning)}


def generate_buffett_output(ticker, analysis_data, state, agent_id="warren_buffett_agent"):
    facts = {"score": analysis_data.get("score"), "max_score": analysis_data.get("max_score"),
             "fundamentals": analysis_data.get("fundamental_analysis", {}).get("details"),
             "consistency": analysis_data.get("consistency_analysis", {}).get("details"),
             "moat": analysis_data.get("moat_analysis", {}).get("details"),
             "pricing_power": analysis_data.get("pricing_power_analysis", {}).get("details"),
             "book_value": analysis_data.get("book_value_analysis", {}).get("details"),
             "management": analysis_data.get("management_analysis", {}).get("details"),
             "intrinsic_value": analysis_data.get("intrinsic_value_analysis", {}).get("intrinsic_value"),
             "market_cap": analysis_data.get("market_cap"),
             "margin_of_safety": analysis_data.get("margin_of_safety")}

    template = ChatPromptTemplate.from_messages([
        ("system", "You are Warren Buffett. Decide bullish, bearish, or neutral using only the provided facts.\n"
         "Checklist: circle of competence, competitive moat, management quality, financial strength, valuation.\n"
         "Signal: bullish if strong business AND margin_of_safety > 0. Bearish if poor business OR overvalued.\n"
         "Confidence: 90-100 exceptional, 70-89 good, 50-69 mixed, 30-49 concerning, 10-29 poor.\n"
         "Keep reasoning under 120 chars. Return JSON only."),
        ("human", "Ticker: {ticker}\nFacts:\n{facts}\n\nReturn exactly: {{\"signal\":\"bullish\"|\"bearish\"|\"neutral\",\"confidence\":int,\"reasoning\":\"short justification\"}}"),
    ])

    prompt = template.invoke({"facts": json.dumps(facts, separators=(",", ":"), ensure_ascii=False), "ticker": ticker})

    def create_default():
        return WarrenBuffettSignal(signal="neutral", confidence=50, reasoning="Insufficient data")

    return call_llm(prompt=prompt, pydantic_model=WarrenBuffettSignal, agent_name=agent_id, state=state, default_factory=create_default)
