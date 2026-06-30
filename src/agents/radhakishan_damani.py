"""Radhakishan Damani Agent — The Retail Superinvestor.

Radhakishan Damani is the founder of D-Mart (Avenue Supermarts) and one of India's
most successful value investors. His philosophy centers on:
- Invest in businesses you deeply understand (circle of competence in retail/consumer)
- Buy durable consumer franchises with predictable cash flows
- Focus on low-cost business models (not high margin, but high efficiency)
- Hold for the long term — true wealth is built through compounding
- Avoid excessive debt; financial prudence is paramount
- Look for businesses where you can predict earnings 10 years out
"""

import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.tools import get_financial_metrics, get_market_cap, search_line_items
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class RadhakishanDamaniSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Damani-style investment reasoning")


def radhakishan_damani_agent(state: AgentState, agent_id: str = "radhakishan_damani_agent"):
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    damani_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial data")
        financial_line_items = search_line_items(
            ticker,
            ["revenue", "gross_profit", "net_income", "operating_income", "total_assets",
             "total_liabilities", "shareholders_equity", "current_assets", "current_liabilities",
             "free_cash_flow", "capital_expenditure", "depreciation_and_amortization",
             "dividends_and_other_cash_distributions", "issuance_or_purchase_of_equity_shares"],
            end_date, period="ttm", limit=5, api_key=api_key,
        )

        market_cap = get_market_cap(ticker, end_date, api_key=api_key)
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing business durability")
        durability = analyze_business_durability(financial_line_items, metrics)

        progress.update_status(agent_id, ticker, "Analyzing financial prudence")
        prudence = analyze_financial_prudence(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing operational efficiency")
        efficiency = analyze_operational_efficiency(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing growth compounding")
        compounding = analyze_compounding_power(financial_line_items, market_cap)

        progress.update_status(agent_id, ticker, "Calculating Damani-style intrinsic value")
        intrinsic_value = calculate_damani_intrinsic_value(financial_line_items, market_cap)

        total_score = durability["score"] + prudence["score"] + efficiency["score"] + compounding["score"]
        max_score = 8 + 7 + 6 + 5

        margin_of_safety = ((intrinsic_value - market_cap) / market_cap) if (intrinsic_value and market_cap) else None

        signal = "neutral"
        if margin_of_safety is not None and margin_of_safety >= 0.25 and total_score >= max_score * 0.6:
            signal = "bullish"
        elif margin_of_safety is not None and margin_of_safety <= -0.20:
            signal = "bearish"
        elif total_score <= max_score * 0.35:
            signal = "bearish"

        analysis_data = {
            "ticker": ticker, "signal": signal, "total_score": total_score, "max_score": max_score,
            "margin_of_safety": margin_of_safety, "durability": durability, "prudence": prudence,
            "efficiency": efficiency, "compounding": compounding,
            "intrinsic_value": intrinsic_value, "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating Damani analysis")
        damani_output = generate_damani_output(ticker, analysis_data, state, agent_id)
        damani_analysis[ticker] = damani_output.model_dump()
        progress.update_status(agent_id, ticker, "Done", analysis=damani_output.reasoning)

    message = HumanMessage(content=json.dumps(damani_analysis), name=agent_id)
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(damani_analysis, "Radhakishan Damani Agent")
    state["data"]["analyst_signals"][agent_id] = damani_analysis
    return {"messages": [message], "data": state["data"]}


def analyze_business_durability(line_items, metrics):
    """Assess business durability — Damani's #1 criterion."""
    if not line_items or not metrics:
        return {"score": 0, "details": "Insufficient data", "max_score": 8}
    latest = line_items[0]
    score = 0
    reasoning = []

    if latest.revenue:
        years_with_data = sum(1 for li in line_items if li.revenue and li.revenue > 0)
        if years_with_data >= 4:
            score += 2
            reasoning.append("Long operating history (4+ years of revenue data)")

    revenues = [li.revenue for li in line_items if li.revenue and li.revenue > 0]
    if len(revenues) >= 3:
        declining = sum(1 for i in range(1, len(revenues)) if revenues[i - 1] < revenues[i])
        if declining == 0:
            score += 3
            reasoning.append("Consistent revenue growth — never declined")
        elif declining <= len(revenues) * 0.2:
            score += 1
            reasoning.append(f"Mostly consistent revenue ({declining}/{len(revenues)-1} dips)")

    if metrics and metrics[0].return_on_equity:
        roe_values = [m.return_on_equity for m in metrics if m.return_on_equity and m.return_on_equity > 0]
        if len(roe_values) >= 3:
            roe_stable = all(0.08 < roe < 0.30 for roe in roe_values)
            if roe_stable:
                score += 2
                reasoning.append("Stable and reasonable ROE (8-30%) suggests durable business")
            elif all(roe > 0 for roe in roe_values):
                score += 1
                reasoning.append("Positive ROE across periods")

    if latest.gross_profit and latest.revenue and latest.revenue > 0:
        gross_margin = latest.gross_profit / latest.revenue
        if 0.15 < gross_margin < 0.40:
            score += 1
            reasoning.append(f"Reasonable gross margin ({gross_margin:.1%}) — not too high to attract competition, not too low")

    return {"score": min(score, 8), "max_score": 8, "details": "; ".join(reasoning) if reasoning else "No durability signals"}


def analyze_financial_prudence(line_items):
    """Damani is famously debt-averse — financial prudence is critical."""
    if not line_items:
        return {"score": 0, "details": "Insufficient data", "max_score": 7}
    latest = line_items[0]
    score = 0
    reasoning = []

    if latest.total_assets and latest.total_liabilities and latest.total_assets > 0:
        debt_ratio = latest.total_liabilities / latest.total_assets
        if debt_ratio < 0.30:
            score += 2
            reasoning.append(f"Very low debt ({debt_ratio:.1%}) — Damani would approve")
        elif debt_ratio < 0.50:
            score += 1
            reasoning.append(f"Manageable debt ({debt_ratio:.1%})")

    if latest.shareholders_equity and latest.total_liabilities:
        d_to_e = latest.total_liabilities / latest.shareholders_equity if latest.shareholders_equity > 0 else None
        if d_to_e is not None:
            if d_to_e < 0.3:
                score += 2
                reasoning.append(f"Excellent D/E ({d_to_e:.2f})")
            elif d_to_e < 0.7:
                score += 1
                reasoning.append(f"Reasonable D/E ({d_to_e:.2f})")

    if latest.current_assets and latest.current_liabilities and latest.current_liabilities > 0:
        curr = latest.current_assets / latest.current_liabilities
        if 1.5 < curr < 3.0:
            score += 1
            reasoning.append(f"Comfortable liquidity ({curr:.2f})")
        elif curr > 3.0:
            reasoning.append(f"Excess liquidity ({curr:.2f}) — might be inefficient")

    issuance = getattr(latest, "issuance_or_purchase_of_equity_shares", None)
    if issuance is not None:
        if issuance < 0:
            score += 1
            reasoning.append("Buying back shares — management is confident")
        elif issuance == 0:
            score += 1
            reasoning.append("No dilution — shareholder-friendly")

    if latest.free_cash_flow and latest.free_cash_flow > 0:
        score += 1
        reasoning.append("Generates positive free cash flow")

    return {"score": min(score, 7), "max_score": 7, "details": "; ".join(reasoning) if reasoning else "No prudence signals"}


def analyze_operational_efficiency(line_items):
    """Damani values low-cost operators with high inventory turns (retail DNA)."""
    if not line_items:
        return {"score": 0, "details": "Insufficient data", "max_score": 6}
    latest = line_items[0]
    score = 0
    reasoning = []

    if latest.revenue and latest.total_assets and latest.total_assets > 0:
        asset_turnover = latest.revenue / latest.total_assets
        if asset_turnover > 1.0:
            score += 2
            reasoning.append(f"High asset turnover ({asset_turnover:.2f}x) — efficient use of capital")
        elif asset_turnover > 0.5:
            score += 1
            reasoning.append(f"Decent asset efficiency ({asset_turnover:.2f}x)")

    if latest.gross_profit and latest.revenue and latest.revenue > 0:
        margin = latest.gross_profit / latest.revenue
        gross_margins = [li.gross_profit / li.revenue for li in line_items
                         if li.gross_profit and li.revenue and li.revenue > 0]
        if len(gross_margins) >= 2:
            margin_trend = gross_margins[0] - gross_margins[-1] if gross_margins[-1] != 0 else 0
            if -0.02 < margin_trend < 0.02:
                score += 1
                reasoning.append(f"Stable gross margin ({margin:.1%}) — consistent operations")
            elif margin_trend > 0.02:
                score += 1
                reasoning.append(f"Improving margins ({margin:.1%}) — scaling efficiency")

    if latest.capital_expenditure and latest.depreciation_and_amortization and latest.depreciation_and_amortization > 0:
        capex_ratio = abs(latest.capital_expenditure) / latest.depreciation_and_amortization
        if 0.5 < capex_ratio < 1.5:
            score += 1
            reasoning.append("Sustainable capex — maintaining assets without overspending")

    fcf_values = [li.free_cash_flow for li in line_items if li.free_cash_flow]
    if len(fcf_values) >= 2 and fcf_values[0] > 0:
        if all(f > 0 for f in fcf_values):
            score += 1
            reasoning.append("Consistently FCF-positive")
        else:
            score += 1
            reasoning.append("Currently FCF-positive")

    if latest.revenue:
        rev_values = [li.revenue for li in line_items if li.revenue and li.revenue > 0]
        if len(rev_values) >= 2:
            revenue_growth = (rev_values[0] / rev_values[-1]) ** (1 / (len(rev_values) - 1)) - 1
            if 0.08 < revenue_growth < 0.25:
                score += 1
                reasoning.append(f"Steady revenue growth ({revenue_growth:.1%} CAGR) — sustainable pace")

    return {"score": min(score, 6), "max_score": 6, "details": "; ".join(reasoning) if reasoning else "No efficiency signals"}


def analyze_compounding_power(line_items, market_cap):
    """Damani's wealth came from compounding — assess the compounding trajectory."""
    if not line_items:
        return {"score": 0, "details": "Insufficient data", "max_score": 5}
    latest = line_items[0]
    score = 0
    reasoning = []

    net_incomes = [li.net_income for li in line_items if li.net_income and li.net_income > 0]
    if len(net_incomes) >= 3:
        cagr = (net_incomes[0] / net_incomes[-1]) ** (1 / (len(net_incomes) - 1)) - 1
        if cagr > 0.15:
            score += 2
            reasoning.append(f"Strong earnings CAGR ({cagr:.1%}) — compounding machine")
        elif cagr > 0.08:
            score += 1
            reasoning.append(f"Decent earnings CAGR ({cagr:.1%})")

    if latest.shareholders_equity and latest.net_income and latest.shareholders_equity > 0:
        roe = latest.net_income / latest.shareholders_equity
        dividend = getattr(latest, "dividends_and_other_cash_distributions", None)
        if dividend and dividend < 0:
            payout = abs(dividend) / latest.net_income if latest.net_income > 0 else 0
            retained_roe = roe * (1 - min(payout, 0.9))
            if retained_roe > 0.12:
                score += 2
                reasoning.append(f"Great reinvestment rate — retaining {1-payout:.0%} at {roe:.1%} ROE")
            elif retained_roe > 0.06:
                score += 1
                reasoning.append(f"Good compounding engine — {retained_roe:.1%} retained ROE")

    if latest.revenue and market_cap and latest.revenue > 0:
        ps_ratio = market_cap / latest.revenue
        if ps_ratio < 2:
            score += 1
            reasoning.append(f"Reasonable P/S ({ps_ratio:.1f}) — room for compounding")
        elif ps_ratio < 5:
            score += 0
        else:
            reasoning.append(f"P/S ratio is high ({ps_ratio:.1f}x) — already pricing in growth")

    return {"score": min(score, 5), "max_score": 5, "details": "; ".join(reasoning) if reasoning else "No compounding signals"}


def calculate_damani_intrinsic_value(line_items, market_cap):
    """Damani-style intrinsic value: conservative earnings-based valuation."""
    if not line_items or not market_cap:
        return None
    latest = line_items[0]
    if not latest.net_income or latest.net_income <= 0:
        return None
    net_incomes = [li.net_income for li in line_items[:5] if li.net_income and li.net_income > 0]
    sustainable_earnings = latest.net_income
    if len(net_incomes) >= 3:
        sustainable_earnings = sum(net_incomes[:3]) / 3
    cagr = 0.05
    if len(net_incomes) >= 3 and net_incomes[-1] > 0:
        cagr = min((net_incomes[0] / net_incomes[-1]) ** (1 / (len(net_incomes) - 1)) - 1, 0.20)
    growth_phase = cagr * 0.6
    discount = 0.12
    pv = sum(sustainable_earnings * (1 + growth_phase) ** yr / (1 + discount) ** yr for yr in range(1, 6))
    terminal = sustainable_earnings * (1 + growth_phase) ** 5 / (discount - 0.03)
    pv_terminal = terminal / (1 + discount) ** 5
    return (pv + pv_terminal) * 0.80


def generate_damani_output(ticker, analysis_data, state, agent_id):
    template = ChatPromptTemplate.from_messages([
        ("system", """You are Radhakishan Damani, India's retail superinvestor and D-Mart founder.
Your investment philosophy:
- Circle of Competence: Only invest in businesses you thoroughly understand (consumer, retail)
- Durable Franchises: Look for businesses with lasting competitive advantages
- Financial Prudence: Avoid debt; let the business fund its own growth
- Operational Efficiency: Low-cost operators win in India's price-sensitive market
- Long-term Compounding: Buy quality and hold for decades
- Margin of Safety: Always buy at a discount to intrinsic value (at least 25%)

When providing your reasoning:
1. Speak in Damani's calm, measured voice — no hype, no panic
2. Focus on business quality over stock price fluctuations
3. Use Indian context: mention the Indian consumer, the retail opportunity
4. Be patient in your assessment — Damani waited years for the right entry

Keep reasoning to 200-300 chars. Return JSON only."""),
        ("human", """Ticker: {ticker}
Analysis: {analysis_data}

Return JSON:
{{"signal": "bullish" | "bearish" | "neutral", "confidence": float 0-100, "reasoning": "string"}}"""),
    ])
    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def default():
        return RadhakishanDamaniSignal(signal="neutral", confidence=0.0, reasoning="Insufficient data")

    return call_llm(prompt=prompt, pydantic_model=RadhakishanDamaniSignal, agent_name=agent_id,
                    state=state, default_factory=default)
