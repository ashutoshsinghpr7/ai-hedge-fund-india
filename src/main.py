#!/usr/bin/env python3
"""ai-hedge-fund-india — AI-powered hedge fund for Indian NSE/BSE markets.

Usage:
    python src/main.py run --ticker RELIANCE,TCS       # Full AI agent analysis
    python src/main.py analyze --ticker RELIANCE        # Quick data fetch
    python src/main.py scan --sector pharma             # Market scan by sector

Inspired by virattt/ai-hedge-fund (MIT License).
"""

import argparse
import json
import os
import warnings
from datetime import datetime

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.progress import progress
from src.utils.analysts import get_analyst_nodes
from src.utils.reporter import save_report
from src.agents.risk_manager import risk_management_agent
from src.agents.portfolio_manager import portfolio_management_agent
from src.graph.state import AgentState
from src.data.universe import get_available_sectors

load_dotenv()

console = Console()
SECTORS = get_available_sectors()


def _ensure_nse(ticker: str) -> str:
    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        return f"{ticker}.NS"
    return ticker


def _display_name(ticker: str) -> str:
    return ticker.replace(".NS", "").replace(".BO", "")


def cmd_run(args):
    """Run the full AI agent pipeline on specified tickers or sector."""
    from src.data.universe import get_universe

    if args.sector:
        tickers = get_universe(sector=args.sector)
        if not tickers:
            console.print(f"[red]No stocks found for sector '{args.sector}'[/red]")
            return
        console.print(f"[dim]Sector '{args.sector}' → {len(tickers)} stocks[/dim]")
    elif args.ticker:
        tickers = [_ensure_nse(t.strip()) for t in args.ticker.split(",")]
    else:
        console.print("[red]Provide --ticker or --sector[/red]")
        return
    start = args.start or "2024-01-01"
    end = args.end or datetime.now().strftime("%Y-%m-%d")
    show_reasoning = args.show_reasoning

    progress.start(verbose=show_reasoning)

    console.rule(f"[bold blue]AI Hedge Fund Analysis: {', '.join(_display_name(t) for t in tickers)}[/bold blue]")
    console.print(f"Period: {start} → {end}")
    console.print(
        "Available agents: warren_buffett, rakesh_jhunjhunwala, radhakishan_damani, "
        "ben_graham, bill_ackman, cathie_wood, charlie_munger, michael_burry, "
        "peter_lynch, phil_fisher, stanley_druckenmiller, aswath_damodaran, "
        "mohnish_pabrai, nassim_taleb, fii_dii_flow, rbi_policy, "
        "technical_analyst, fundamentals_analyst, sentiment_analyst, valuation_analyst, "
        "growth_analyst, news_sentiment"
    )
    console.print()

    portfolio = {
        "cash": args.initial_cash or 10000000.0,
        "margin_requirement": args.margin_requirement or 0.0,
        "margin_used": 0.0,
        "positions": {
            ticker: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "short_margin_used": 0.0}
            for ticker in tickers
        },
        "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in tickers},
    }

    try:
        workflow = create_workflow()
        agent = workflow.compile()

        final_state = agent.invoke({
            "messages": [HumanMessage(content="Make trading decisions based on the provided data.")],
            "data": {"tickers": tickers, "portfolio": portfolio, "start_date": start, "end_date": end,
                     "analyst_signals": {}},
            "metadata": {"show_reasoning": show_reasoning, "model_name": args.model or "deepseek-chat",
                         "model_provider": args.provider or "DeepSeek"},
        })

        analyst_signals = final_state["data"]["analyst_signals"]
        decisions_raw = final_state["messages"][-1].content

        try:
            decisions = json.loads(decisions_raw) if isinstance(decisions_raw, str) else decisions_raw
        except json.JSONDecodeError:
            decisions = {"error": "Could not parse decisions", "raw": str(decisions_raw)}

        if not getattr(args, "no_save", False):
            risk_signals = analyst_signals.get("risk_management_agent", {})
            metadata = final_state.get("metadata", {})
            for ticker in tickers:
                try:
                    price = risk_signals.get(ticker, {}).get("current_price", 0.0)
                    decision = decisions.get(ticker, {})
                    console.print(f"[dim]Saving report for {ticker}...[/dim]")
                    save_report(ticker, analyst_signals, decision, price or 0.0, metadata)
                except Exception as e:
                    console.print(f"[red]Report save failed for {ticker}: {e}[/red]")

        _print_results(tickers, analyst_signals, decisions, show_reasoning)

    finally:
        progress.stop()


def _print_results(tickers, analyst_signals, decisions, show_reasoning):
    console.rule("[bold green]Analysis Results[/bold green]")

    for ticker in tickers:
        display = _display_name(ticker)
        console.print(f"\n[bold cyan]━━━ {display} ━━━[/bold cyan]")

        table = Table(title="Agent Signals")
        table.add_column("Agent", style="cyan")
        table.add_column("Signal", style="bold")
        table.add_column("Confidence", style="green")

        for agent_name, signals in analyst_signals.items():
            if agent_name.startswith("risk_management") or ticker not in signals:
                continue
            sig = signals[ticker]
            signal_str = sig.get("signal", "unknown")
            confidence = sig.get("confidence", 0)
            signal_color = {"bullish": "green", "bearish": "red", "neutral": "yellow"}.get(signal_str, "white")
            table.add_row(agent_name.replace("_agent", ""), f"[{signal_color}]{signal_str}[/{signal_color}]",
                         f"{confidence:.0f}%")

        console.print(table)

        decision = decisions.get(ticker, {})
        if decision:
            action = decision.get("action", "unknown")
            qty = decision.get("quantity", 0)
            conf = decision.get("confidence", 0)
            reasoning = decision.get("reasoning", "")
            action_color = {"buy": "green", "sell": "red", "hold": "yellow", "short": "red"}.get(action, "white")
            console.print(Panel(f"[{action_color}]Decision: {action.upper()} {qty} shares[/{action_color}] "
                               f"(confidence: {conf}%)\n{reasoning}", title="Portfolio Manager"))

    console.print()


def cmd_analyze(args):
    """Quick data fetch without running AI agents."""
    from src.tools import get_price_data, get_financial_metrics

    tickers = [_ensure_nse(t.strip()) for t in args.ticker.split(",")]
    for ticker in tickers:
        display = _display_name(ticker)
        console.rule(f"[bold blue]{display} Data[/bold blue]")
        start = args.start or "2024-01-01"
        end = args.end or datetime.now().strftime("%Y-%m-%d")

        prices_df = get_price_data(ticker, start, end)
        if not prices_df.empty:
            console.print(f"  [green]✓[/green] Price: {len(prices_df)} bars | "
                         f"Close: ₹{prices_df['close'].iloc[-1]:.2f}")
        else:
            console.print("  [red]✗[/red] No price data")
            continue

        metrics = get_financial_metrics(ticker, end, limit=1)
        if metrics:
            m = metrics[0]
            table = Table(title="Financial Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            pct_fields = {"return_on_equity", "operating_margin", "net_margin", "gross_margin",
                         "return_on_assets", "free_cash_flow_yield"}
            ratio_fields = {"price_to_earnings_ratio", "price_to_book_ratio", "debt_to_equity",
                           "current_ratio", "earnings_per_share"}
            for name in ["market_cap", "price_to_earnings_ratio", "return_on_equity", "operating_margin",
                         "net_margin", "debt_to_equity", "current_ratio", "earnings_per_share",
                         "free_cash_flow_yield"]:
                val = getattr(m, name, None)
                if val is not None:
                    label = name.replace("_", " ").title()
                    if name in pct_fields:
                        table.add_row(label, f"{val:.2%}")
                    elif name in ratio_fields:
                        table.add_row(label, f"{val:.2f}")
                    elif name == "market_cap":
                        table.add_row(label, f"₹{val/1e12:.2f}T" if val >= 1e12 else f"₹{val/1e9:.2f}B")
            console.print(table)
        console.print()


def cmd_scan(args):
    """Scan a universe of stocks."""
    from src.data.universe import get_universe
    from src.tools import get_price_data

    if args.sector:
        symbols = get_universe(sector=args.sector)
    else:
        symbols = get_universe(
            universe_type=args.universe or "nifty50",
            exclude_sectors=args.exclude_sectors.split(",") if args.exclude_sectors else None,
            exclude_symbols=args.exclude_symbols.split(",") if args.exclude_symbols else None,
        )

    label = args.sector or args.universe or "nifty50"
    console.rule(f"[bold blue]Scan: {label} ({len(symbols)} stocks)[/bold blue]")
    end = datetime.now().strftime("%Y-%m-%d")
    start = "2024-01-01"

    for ticker in symbols[:min(args.limit or 10, len(symbols))]:
        try:
            df = get_price_data(ticker, start, end)
            if not df.empty:
                valid = df["close"].dropna()
                if valid.empty:
                    console.print(f"  [yellow]? {ticker}: No recent price data[/yellow]")
                    continue
                close = valid.iloc[-1]
                chg = (valid.iloc[-1] / valid.iloc[-2] - 1) * 100 if len(valid) >= 2 else 0
                sym = "▲" if chg > 0 else "▼" if chg < 0 else "—"
                color = "green" if chg > 0 else "red" if chg < 0 else "white"
                console.print(f"  [{color}]{sym} {ticker.replace('.NS',''):<20} ₹{close:>8.2f}  ({chg:+.2f}%)[/{color}]")
            else:
                console.print(f"  [red]✗ {ticker}: No data[/red]")
        except Exception as e:
            console.print(f"  [yellow]? {ticker}: {e}[/yellow]")
    console.print()


def create_workflow(selected_analysts=None):
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", lambda s: s)

    analyst_nodes = get_analyst_nodes()
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())

    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)
    workflow.set_entry_point("start_node")
    return workflow


def main():
    parser = argparse.ArgumentParser(
        description="ai-hedge-fund-india — AI hedge fund for Indian NSE/BSE markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true", help="Show debug warnings (pandas deprecation, etc.)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    run = subparsers.add_parser("run", help="Run full AI agent pipeline")
    run.add_argument("--ticker", help="Comma-separated tickers (e.g. RELIANCE,TCS.NS,HDFCBANK)")
    run.add_argument("--sector", choices=SECTORS, help="Sector or index (e.g. pharma, banking, nifty50)")
    run.add_argument("--start", help="Start date (YYYY-MM-DD)")
    run.add_argument("--end", help="End date (YYYY-MM-DD)")
    run.add_argument("--show-reasoning", action="store_true", help="Show agent reasoning")
    run.add_argument("--model", default="deepseek-chat", help="LLM model name")
    run.add_argument("--provider", default="DeepSeek", help="LLM provider (DeepSeek, OpenAI, Anthropic, Groq)")
    run.add_argument("--initial-cash", type=float, default=10000000.0, help="Initial cash (default ₹1 Cr)")
    run.add_argument("--margin-requirement", type=float, default=0.0, help="Margin requirement")
    run.add_argument("--no-save", action="store_true", help="Skip saving reports to disk")

    analyze = subparsers.add_parser("analyze", help="Quick data fetch (no AI agents)")
    analyze.add_argument("--ticker", required=True, help="Comma-separated tickers")
    analyze.add_argument("--start", help="Start date")
    analyze.add_argument("--end", help="End date")

    scan = subparsers.add_parser("scan", help="Scan a universe of stocks")
    scan.add_argument("--universe", default="nifty50", choices=["nifty50", "nifty100", "nifty500"])
    scan.add_argument("--sector", choices=SECTORS, help="Sector or index (e.g. pharma, banking, nifty50)")
    scan.add_argument("--exclude-sectors", help="Comma-separated sectors")
    scan.add_argument("--exclude-symbols", help="Comma-separated symbols")
    scan.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    debug = args.debug or os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
    if not debug:
        warnings.filterwarnings("ignore", category=FutureWarning)

    {"run": cmd_run, "analyze": cmd_analyze, "scan": cmd_scan}.get(args.command, lambda a: parser.print_help())(args)


if __name__ == "__main__":
    main()
