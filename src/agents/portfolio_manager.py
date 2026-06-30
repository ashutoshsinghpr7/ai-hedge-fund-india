import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.utils.llm import call_llm


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")


def portfolio_management_agent(state: AgentState, agent_id: str = "portfolio_manager"):
    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Processing analyst signals")
        risk_manager_id = "risk_management_agent"
        risk_data = analyst_signals.get(risk_manager_id, {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0.0)
        current_prices[ticker] = float(risk_data.get("current_price", 0.0))
        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] // current_prices[ticker])
        else:
            max_shares[ticker] = 0

        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            if not agent.startswith("risk_management_agent") and ticker in signals:
                sig = signals[ticker].get("signal")
                conf = signals[ticker].get("confidence")
                if sig is not None and conf is not None:
                    ticker_signals[agent] = {"sig": sig, "conf": conf}
        signals_by_ticker[ticker] = ticker_signals

    state["data"]["current_prices"] = current_prices
    progress.update_status(agent_id, None, "Generating trading decisions")

    result = generate_trading_decision(
        tickers=tickers, signals_by_ticker=signals_by_ticker, current_prices=current_prices,
        max_shares=max_shares, portfolio=portfolio, agent_id=agent_id, state=state,
    )
    message = HumanMessage(
        content=json.dumps({t: d.model_dump() for t, d in result.decisions.items()}), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(
            {t: d.model_dump() for t, d in result.decisions.items()}, "Portfolio Manager")

    return {"messages": state["messages"] + [message], "data": state["data"]}


def compute_allowed_actions(tickers, current_prices, max_shares, portfolio):
    allowed = {}
    cash = float(portfolio.get("cash", 0.0))
    positions = portfolio.get("positions", {}) or {}

    for ticker in tickers:
        price = float(current_prices.get(ticker, 0.0))
        pos = positions.get(ticker, {"long": 0, "long_cost_basis": 0.0, "short": 0, "short_cost_basis": 0.0})
        long_shares = int(pos.get("long", 0) or 0)
        short_shares = int(pos.get("short", 0) or 0)
        max_qty = int(max_shares.get(ticker, 0) or 0)

        actions = {"buy": 0, "sell": 0, "short": 0, "cover": 0, "hold": 0}
        if long_shares > 0:
            actions["sell"] = long_shares
        if cash > 0 and price > 0:
            max_buy = max(0, min(max_qty, int(cash // price)))
            if max_buy > 0:
                actions["buy"] = max_buy
        if short_shares > 0:
            actions["cover"] = short_shares

        pruned = {"hold": 0}
        for k, v in actions.items():
            if k != "hold" and v > 0:
                pruned[k] = v
        allowed[ticker] = pruned

    return allowed


def _compact_signals(signals_by_ticker):
    out = {}
    for t, agents in signals_by_ticker.items():
        if not agents:
            out[t] = {}
            continue
        compact = {}
        for agent, payload in agents.items():
            sig = payload.get("sig") or payload.get("signal")
            conf = payload.get("conf") if "conf" in payload else payload.get("confidence")
            if sig is not None and conf is not None:
                compact[agent] = {"sig": sig, "conf": conf}
        out[t] = compact
    return out


def generate_trading_decision(tickers, signals_by_ticker, current_prices, max_shares, portfolio, agent_id, state):
    allowed_actions_full = compute_allowed_actions(tickers, current_prices, max_shares, portfolio)

    prefilled_decisions: dict[str, PortfolioDecision] = {}
    tickers_for_llm: list[str] = []
    for t in tickers:
        aa = allowed_actions_full.get(t, {"hold": 0})
        if set(aa.keys()) == {"hold"}:
            prefilled_decisions[t] = PortfolioDecision(action="hold", quantity=0, confidence=100,
                                                        reasoning="No valid trade available")
        else:
            tickers_for_llm.append(t)

    if not tickers_for_llm:
        return PortfolioManagerOutput(decisions=prefilled_decisions)

    compact_signals = _compact_signals({t: signals_by_ticker.get(t, {}) for t in tickers_for_llm})
    compact_allowed = {t: allowed_actions_full[t] for t in tickers_for_llm}

    template = ChatPromptTemplate.from_messages([
        ("system", "You are a portfolio manager. Pick one allowed action per ticker. "
         "Keep reasoning very concise (max 100 chars). Return JSON only."),
        ("human", "Signals:\n{signals}\n\nAllowed actions:\n{allowed}\n\n"
         'Return: {{\"decisions\": {{\"TICKER\": {{\"action\":\"...\",\"quantity\":int,\"confidence\":int,\"reasoning\":\"...\"}}}}}}'),
    ])

    prompt = template.invoke({
        "signals": json.dumps(compact_signals, separators=(",", ":"), ensure_ascii=False),
        "allowed": json.dumps(compact_allowed, separators=(",", ":"), ensure_ascii=False),
    })

    def create_default():
        decisions = dict(prefilled_decisions)
        for t in tickers_for_llm:
            decisions[t] = PortfolioDecision(action="hold", quantity=0, confidence=0,
                                               reasoning="Default decision: hold")
        return PortfolioManagerOutput(decisions=decisions)

    llm_out = call_llm(prompt=prompt, pydantic_model=PortfolioManagerOutput,
                       agent_name=agent_id, state=state, default_factory=create_default)

    merged = dict(prefilled_decisions)
    merged.update(llm_out.decisions)
    return PortfolioManagerOutput(decisions=merged)
