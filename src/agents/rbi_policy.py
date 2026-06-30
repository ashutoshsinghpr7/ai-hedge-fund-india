"""RBI Policy Agent — Monetary Policy Impact Analysis for Indian Markets.

Analyzes the impact of RBI (Reserve Bank of India) monetary policy on specific
stocks and sectors. Since we don't have a real-time RBI policy data feed, this
agent analyzes macro context, sector sensitivity to interest rates, and uses
LLM reasoning with Indian economic context.

Key factors:
- Repo rate direction → impacts banking, real estate, auto sectors most
- CRR/SLR changes → impacts bank liquidity and lending capacity
- INR/USD movement → impacts importers (negative) and exporters/IT (positive)
- Inflation trajectory → impacts consumer demand and RBI stance
- Bond yield movements → signal market's rate expectations
"""

import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.tools import get_financial_metrics, get_market_cap, get_company_news
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class RBIPolicySignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(description="Confidence 0-100")
    reasoning: str = Field(description="RBI policy impact reasoning")


def rbi_policy_agent(state: AgentState, agent_id: str = "rbi_policy_agent"):
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    policy_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Analyzing sector sensitivity to RBI policy")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5, api_key=api_key)
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Assessing interest rate sensitivity")
        rate_sensitivity = analyze_rate_sensitivity(metrics, market_cap)

        progress.update_status(agent_id, ticker, "Assessing currency exposure")
        currency_exposure = analyze_currency_exposure(ticker, metrics)

        progress.update_status(agent_id, ticker, "Assessing liquidity impact")
        liquidity_impact = analyze_liquidity_impact(metrics)

        progress.update_status(agent_id, ticker, "Fetching macro news context")
        news = get_company_news(ticker, end_date, limit=20, api_key=api_key)

        total_score = rate_sensitivity["score"] + currency_exposure["score"] + liquidity_impact["score"]
        max_score = rate_sensitivity.get("max_score", 6) + currency_exposure.get("max_score", 5) + liquidity_impact.get("max_score", 4)

        analysis_data = {
            "ticker": ticker, "rate_sensitivity": rate_sensitivity,
            "currency_exposure": currency_exposure, "liquidity_impact": liquidity_impact,
            "total_score": total_score, "max_score": max_score, "news_count": len(news),
        }

        progress.update_status(agent_id, ticker, "Generating RBI policy analysis")
        policy_output = generate_rbi_output(ticker, analysis_data, state, agent_id)
        policy_analysis[ticker] = policy_output.model_dump()
        progress.update_status(agent_id, ticker, "Done", analysis=policy_output.reasoning)

    message = HumanMessage(content=json.dumps(policy_analysis), name=agent_id)
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(policy_analysis, "RBI Policy Agent")
    state["data"]["analyst_signals"][agent_id] = policy_analysis
    return {"messages": [message], "data": state["data"]}


def analyze_rate_sensitivity(metrics: list, market_cap) -> dict:
    """Assess how sensitive the stock is to interest rate changes."""
    score = 0
    reasoning = []
    max_score = 6

    if not metrics:
        return {"score": 0, "max_score": max_score, "details": "No data"}

    latest = metrics[0]

    if latest.debt_to_equity is not None:
        d_to_e = latest.debt_to_equity
        if d_to_e < 0.3:
            score += 2
            reasoning.append(f"Low debt (D/E: {d_to_e:.2f}) — less sensitive to rate hikes, benefits from lower rates")
        elif d_to_e < 0.7:
            score += 1
            reasoning.append(f"Moderate debt (D/E: {d_to_e:.2f}) — some rate sensitivity")
        else:
            reasoning.append(f"High debt (D/E: {d_to_e:.2f}) — vulnerable to rate hikes")

    if latest.interest_coverage is not None:
        coverage = latest.interest_coverage
        if coverage > 5:
            score += 2
            reasoning.append(f"Strong interest coverage ({coverage:.1f}x) — can absorb rate increases")
        elif coverage > 2:
            score += 1
            reasoning.append(f"Adequate interest coverage ({coverage:.1f}x)")
        else:
            reasoning.append(f"Low interest coverage ({coverage:.1f}x) — rate-sensitive")

    if market_cap and market_cap > 1_000_000_000_000:
        score += 1
        reasoning.append("Large-cap — easier access to capital, less rate-sensitive")
    elif market_cap and market_cap < 100_000_000_000:
        reasoning.append("Small/mid-cap — more dependent on credit availability")

    if latest.free_cash_flow_yield is not None:
        fcf_yield = latest.free_cash_flow_yield
        if fcf_yield > 0.03:
            score += 1
            reasoning.append(f"Good FCF yield ({fcf_yield:.1%}) — internally funded, less rate-dependent")

    return {"score": min(score, max_score), "max_score": max_score, "details": "; ".join(reasoning) if reasoning else "No rate sensitivity data"}


def analyze_currency_exposure(ticker: str, metrics: list) -> dict:
    """Assess INR/USD exposure — key for RBI policy impact."""
    score = 0
    reasoning = []
    max_score = 5

    sector_map = {
        "Technology": {"export_heavy": True, "score": 3, "label": "IT exporter — benefits from INR weakness"},
        "Pharma": {"export_heavy": True, "score": 2, "label": "Pharma exporter — mild INR benefit"},
        "Auto": {"export_heavy": False, "score": 1, "label": "Auto — partially import-dependent, INR-sensitive"},
        "Energy": {"export_heavy": False, "score": 0, "label": "Energy — import-heavy, hurt by INR weakness"},
        "Metals": {"export_heavy": False, "score": 1, "label": "Metals — mixed exposure to global prices"},
        "Banking": {"export_heavy": False, "score": 2, "label": "Banking — domestic-focused, indirect FX impact"},
        "Financial Services": {"export_heavy": False, "score": 2, "label": "Financials — domestic, stable"},
        "Consumer": {"export_heavy": False, "score": 2, "label": "Consumer — domestic demand, some input imports"},
        "Cement": {"export_heavy": False, "score": 2, "label": "Cement — domestic, low FX exposure"},
        "Construction": {"export_heavy": False, "score": 2, "label": "Construction — domestic, infrastructure play"},
        "Telecom": {"export_heavy": False, "score": 2, "label": "Telecom — domestic, some equipment imports"},
        "Insurance": {"export_heavy": False, "score": 2, "label": "Insurance — domestic, stable"},
        "Defence": {"export_heavy": False, "score": 2, "label": "Defence — domestic, government contracts"},
        "Industrial": {"export_heavy": False, "score": 1, "label": "Industrial — mixed, some import dependence"},
        "Hospitality": {"export_heavy": False, "score": 2, "label": "Hospitality — domestic demand-driven"},
        "Real Estate": {"export_heavy": False, "score": 2, "label": "Real Estate — domestic, rate-sensitive"},
        "Aviation": {"export_heavy": False, "score": 0, "label": "Aviation — fuel imports, USD-denominated costs"},
        "Agriculture": {"export_heavy": True, "score": 2, "label": "Agriculture — depends on commodity type"},
        "Infrastructure": {"export_heavy": False, "score": 2, "label": "Infrastructure — domestic, capex-driven"},
        "Tourism": {"export_heavy": False, "score": 2, "label": "Tourism — domestic, INR-sensitive"},
        "Chemicals": {"export_heavy": True, "score": 1, "label": "Chemicals — mixed, some exports"},
    }

    from src.data.universe import get_sector
    sector = get_sector(ticker)
    sector_info = sector_map.get(sector, {"export_heavy": False, "score": 1, "label": "Unknown sector"})

    score += sector_info["score"]
    reasoning.append(sector_info["label"])

    if metrics and metrics[0].revenue_growth is not None and metrics[0].revenue_growth > 0.15:
        score += 1
        reasoning.append("Strong revenue growth — can absorb FX headwinds")

    if metrics and metrics[0].operating_margin is not None and metrics[0].operating_margin > 0.20:
        score += 1
        reasoning.append("Healthy margins — cushion against INR volatility")

    return {"score": min(score, max_score), "max_score": max_score, "details": "; ".join(reasoning)}


def analyze_liquidity_impact(metrics: list) -> dict:
    """Assess RBI liquidity measures impact (CRR, SLR, OMO, etc.)."""
    score = 0
    reasoning = []
    max_score = 4

    if not metrics:
        return {"score": 2, "max_score": max_score, "details": "Neutral — no data"}

    latest = metrics[0]

    if latest.current_ratio is not None:
        curr = latest.current_ratio
        if curr > 2.0:
            score += 2
            reasoning.append(f"Strong liquidity ({curr:.2f}) — not dependent on RBI liquidity measures")
        elif curr > 1.2:
            score += 1
            reasoning.append(f"Adequate liquidity ({curr:.2f})")
        else:
            reasoning.append(f"Tight liquidity ({curr:.2f}) — would benefit from RBI easing")

    if latest.free_cash_flow_yield is not None and latest.free_cash_flow_yield > 0.03:
        score += 1
        reasoning.append("Generates own cash — less dependent on systemic liquidity")

    if latest.operating_cash_flow_ratio is not None and latest.operating_cash_flow_ratio > 1.0:
        score += 1
        reasoning.append("Operating cash flow covers obligations")

    return {"score": min(score, max_score), "max_score": max_score, "details": "; ".join(reasoning) if reasoning else "Neutral liquidity position"}


def generate_rbi_output(ticker, analysis_data, state, agent_id):
    template = ChatPromptTemplate.from_messages([
        ("system", """You are an RBI Monetary Policy Analyst for Indian markets.
Your expertise: analyzing how RBI policy decisions impact individual stocks.

Key RBI policy levers and their market impact:
- Repo Rate changes: Higher = negative for banks (NIM pressure), auto, real estate, NBFCs. Lower = positive.
- CRR (Cash Reserve Ratio): Higher = tight liquidity (negative). Lower = more lending capacity (positive).
- INR/USD: Weak INR = positive for IT/Pharma exporters, negative for importers (oil, metals). Strong INR = opposite.
- CPI Inflation: Above RBI target (4%±2%) = hawkish stance expected, negative for rate-sensitives.
- Bond Yields: Rising = market expects rate hikes. Falling = dovish expectations.

Sector sensitivity to RBI policy (in descending order):
- Most sensitive: Banking, NBFCs, Real Estate, Auto, Infrastructure
- Moderate: Consumer, Industrial, Cement, Metals
- Least sensitive: IT, Pharma, FMCG

When providing your reasoning:
1. Assess whether current RBI stance helps or hurts this specific stock
2. Consider the sector context (banking vs IT have opposite rate reactions)
3. Use Indian economic context — mention repo rate levels, CPI inflation, INR level
4. Be specific about which policy factor matters most for this ticker
5. Keep reasoning to 200-300 chars

Return JSON only."""),
        ("human", """Ticker: {ticker}
RBI Policy Analysis: {analysis_data}

Return JSON:
{{"signal": "bullish" | "bearish" | "neutral", "confidence": float 0-100, "reasoning": "string"}}"""),
    ])
    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def default():
        return RBIPolicySignal(signal="neutral", confidence=50.0, reasoning="Could not assess RBI policy impact")

    return call_llm(prompt=prompt, pydantic_model=RBIPolicySignal, agent_name=agent_id,
                    state=state, default_factory=default)
