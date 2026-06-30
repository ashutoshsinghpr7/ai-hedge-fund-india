"""FII/DII Flow Agent — Foreign vs Domestic Institutional Investor Analysis.

Tracks institutional money flows in Indian markets. FII (Foreign Institutional
Investor) and DII (Domestic Institutional Investor) flows are key drivers of
Indian market direction. This agent analyzes price/volume patterns that reveal
institutional footprint and uses LLM reasoning for macro flow context.

Key signals:
- Rising prices on rising volume → institutional accumulation
- Falling prices on rising volume → institutional distribution
- FII buying supports large-caps; DII buying supports mid-caps
- FII outflows often coincide with INR weakness and global risk-off
"""

import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal
import pandas as pd
import numpy as np

from src.graph.state import AgentState, show_agent_reasoning
from src.tools import get_prices, prices_to_df, get_company_news
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class FIIDIISignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(description="Confidence 0-100")
    reasoning: str = Field(description="FII/DII flow analysis reasoning")


def fii_dii_flow_agent(state: AgentState, agent_id: str = "fii_dii_flow_agent"):
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    flow_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching price and volume data")
        prices = get_prices(ticker=ticker, start_date=start_date, end_date=end_date, api_key=api_key)
        if not prices:
            flow_analysis[ticker] = {"signal": "neutral", "confidence": 0, "reasoning": "No price data"}
            continue

        prices_df = prices_to_df(prices)

        progress.update_status(agent_id, ticker, "Analyzing institutional accumulation patterns")
        insti_score = analyze_institutional_footprint(prices_df)

        progress.update_status(agent_id, ticker, "Analyzing volume-price divergence")
        vp_divergence = analyze_volume_price_divergence(prices_df)

        progress.update_status(agent_id, ticker, "Analyzing sector flow context")
        sector_context = analyze_flow_context(prices_df)

        progress.update_status(agent_id, ticker, "Fetching flow-relevant news")
        news = get_company_news(ticker, end_date, limit=50, api_key=api_key)

        total_score = insti_score["score"] + vp_divergence["score"] + sector_context["score"]
        max_score = insti_score.get("max_score", 6) + vp_divergence.get("max_score", 5) + sector_context.get("max_score", 4)

        analysis_data = {
            "ticker": ticker, "total_score": total_score, "max_score": max_score,
            "institutional_footprint": insti_score, "vp_divergence": vp_divergence,
            "sector_context": sector_context, "news_count": len(news),
        }

        progress.update_status(agent_id, ticker, "Generating FII/DII analysis")
        flow_output = generate_fii_dii_output(ticker, analysis_data, state, agent_id)
        flow_analysis[ticker] = flow_output.model_dump()
        progress.update_status(agent_id, ticker, "Done", analysis=flow_output.reasoning)

    message = HumanMessage(content=json.dumps(flow_analysis), name=agent_id)
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(flow_analysis, "FII/DII Flow Agent")
    state["data"]["analyst_signals"][agent_id] = flow_analysis
    return {"messages": [message], "data": state["data"]}


def analyze_institutional_footprint(prices_df: pd.DataFrame) -> dict:
    """Detect institutional buying/selling via volume and price patterns."""
    if prices_df.empty or len(prices_df) < 20:
        return {"score": 0, "max_score": 6, "details": "Insufficient data"}

    score = 0
    reasoning = []

    close = prices_df["close"]
    volume = prices_df["volume"]

    volume_ma_20 = volume.rolling(20).mean()

    high_vol_days = volume.iloc[-20:] > volume_ma_20.iloc[-20:] * 1.3
    up_days = close.pct_change().iloc[-20:] > 0

    heavy_up = sum(high_vol_days & up_days)
    heavy_down = sum(high_vol_days & ~up_days)

    if heavy_up > heavy_down * 2:
        score += 3
        reasoning.append(f"Strong accumulation: {heavy_up} heavy-up days vs {heavy_down} heavy-down days")
    elif heavy_up > heavy_down:
        score += 2
        reasoning.append(f"Mild accumulation: {heavy_up} up vs {heavy_down} down on high volume")
    elif heavy_down > heavy_up * 2:
        score += 0
        reasoning.append(f"Distribution detected: {heavy_down} heavy-down days")
    else:
        score += 1
        reasoning.append("Mixed institutional activity — no clear direction")

    recent_vol = volume.iloc[-5:].mean()
    avg_vol = volume_ma_20.iloc[-5:].mean()
    vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1

    if vol_ratio > 1.5:
        score += 2
        reasoning.append(f"Volume surge ({vol_ratio:.1f}x avg) — institutions may be active")
    elif vol_ratio > 1.2:
        score += 1
        reasoning.append(f"Above-average volume ({vol_ratio:.1f}x)")
    elif vol_ratio < 0.6:
        score += 1
        reasoning.append(f"Low volume ({vol_ratio:.1f}x) — institutional disinterest")

    close_ma_50 = close.rolling(50).mean()
    if len(prices_df) >= 50:
        price_vs_ma50 = close.iloc[-1] / close_ma_50.iloc[-1]
        if price_vs_ma50 > 1.05:
            score += 1
            reasoning.append(f"Trading above 50-DMA ({price_vs_ma50:.1%}) — bullish institutional posture")
        elif price_vs_ma50 < 0.95:
            reasoning.append(f"Trading below 50-DMA ({price_vs_ma50:.1%}) — bearish posture")

    return {"score": min(score, 6), "max_score": 6, "details": "; ".join(reasoning)}


def analyze_volume_price_divergence(prices_df: pd.DataFrame) -> dict:
    """Detect volume-price divergences that signal institutional activity."""
    if prices_df.empty or len(prices_df) < 10:
        return {"score": 0, "max_score": 5, "details": "Insufficient data"}

    score = 0
    reasoning = []

    close = prices_df["close"]
    volume = prices_df["volume"]

    recent_close_change = (close.iloc[-1] / close.iloc[-10] - 1) if len(close) >= 10 else 0
    recent_vol_ratio = volume.iloc[-5:].mean() / volume.iloc[-10:-5].mean() if len(volume) >= 10 and volume.iloc[-10:-5].mean() > 0 else 1

    if recent_close_change < -0.05 and recent_vol_ratio < 0.7:
        score += 2
        reasoning.append("Price falling on low volume — selling exhaustion, possible accumulation zone")
    elif recent_close_change > 0.05 and recent_vol_ratio < 0.7:
        score += 1
        reasoning.append("Price rising on low volume — no strong institutional conviction")

    if recent_close_change < -0.03 and recent_vol_ratio > 1.3:
        score += 0
        reasoning.append("Selling on high volume — institutional distribution")
    elif recent_close_change > 0.03 and recent_vol_ratio > 1.3:
        score += 2
        reasoning.append("Buying on high volume — institutional accumulation")

    up_vol = volume[close.pct_change() > 0].iloc[-20:].mean() if len(volume[close.pct_change() > 0]) > 0 else 0
    down_vol = volume[close.pct_change() <= 0].iloc[-20:].mean() if len(volume[close.pct_change() <= 0]) > 0 else 0

    if up_vol > down_vol * 1.5:
        score += 1
        reasoning.append("Up-day volume dominates — buyers are more aggressive")
    elif down_vol > up_vol * 1.5:
        reasoning.append("Down-day volume dominates — sellers are more aggressive")

    return {"score": min(score, 5), "max_score": 5, "details": "; ".join(reasoning) if reasoning else "No clear divergence"}


def analyze_flow_context(prices_df: pd.DataFrame) -> dict:
    """Analyze broader market context affecting FII/DII flows."""
    score = 0
    reasoning = []

    close = prices_df["close"]
    if len(close) < 20:
        return {"score": 0, "max_score": 4, "details": "Insufficient data"}

    mom_1m = close.iloc[-1] / close.iloc[-21] - 1 if len(close) >= 21 else 0
    mom_3m = close.iloc[-1] / close.iloc[-63] - 1 if len(close) >= 63 else 0

    if mom_1m > 0.05 and mom_3m > 0.10:
        score += 2
        reasoning.append(f"Strong momentum (1M: {mom_1m:.1%}, 3M: {mom_3m:.1%}) — FIIs attracted to momentum")
    elif mom_1m > 0 and mom_3m > 0:
        score += 1
        reasoning.append("Positive momentum — institutional interest likely")

    returns = close.pct_change().dropna()
    recent_volatility = returns.iloc[-20:].std() * np.sqrt(252)
    hist_vol = returns.std() * np.sqrt(252)

    if recent_volatility < hist_vol * 0.8:
        score += 1
        reasoning.append(f"Volatility declining ({recent_volatility:.1%}) — FIIs prefer stable environments")
    elif recent_volatility > hist_vol * 1.3:
        reasoning.append(f"Elevated volatility ({recent_volatility:.1%}) — FIIs may reduce exposure")

    if close.iloc[-1] > 1000:
        score += 1
        reasoning.append("Large-cap stock — historically favored by FIIs")

    return {"score": min(score, 4), "max_score": 4, "details": "; ".join(reasoning) if reasoning else "No context signals"}


def generate_fii_dii_output(ticker, analysis_data, state, agent_id):
    template = ChatPromptTemplate.from_messages([
        ("system", """You are an FII/DII Flow Analyst for Indian markets.
Your expertise: analyzing institutional money flows in NSE/BSE stocks.

Key principles:
- FIIs (Foreign Institutional Investors) drive large-cap momentum; they buy on India's growth story
- DIIs (Domestic Institutions: mutual funds, insurance) provide steady counter-flow; they buy on dips
- High volume + rising prices = institutional accumulation (bullish)
- High volume + falling prices = institutional distribution (bearish)
- FII flows correlate with INR strength, global risk appetite, and emerging market allocations
- DII flows are driven by domestic SIP flows, insurance premiums — more stable

When providing your reasoning:
1. Be specific about volume patterns you're seeing
2. Mention likely FII vs DII behavior based on the stock profile
3. Contextualize with Indian market dynamics
4. Keep reasoning concise (200-300 chars)

Return JSON only."""),
        ("human", """Ticker: {ticker}
Flow Analysis Data: {analysis_data}

Return JSON:
{{"signal": "bullish" | "bearish" | "neutral", "confidence": float 0-100, "reasoning": "string"}}"""),
    ])
    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def default():
        return FIIDIISignal(signal="neutral", confidence=50.0, reasoning="Could not determine institutional flows")

    return call_llm(prompt=prompt, pydantic_model=FIIDIISignal, agent_name=agent_id,
                    state=state, default_factory=default)
