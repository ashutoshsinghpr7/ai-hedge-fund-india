"""Report saving — Markdown + JSON per ticker. On re-run, old report moves to archive/."""

import json
from datetime import datetime
from pathlib import Path


def _str_val(v):
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False, default=str, indent=2)
    return str(v) if v is not None else ""


def _read_json_timestamp(path: Path) -> str | None:
    """Read timestamp from an existing JSON report."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("timestamp")
    except Exception:
        return None


def _archive_old(reports_dir: Path, archive_dir: Path, display: str) -> None:
    """If a previous report exists, move it to archive/ with its original timestamp."""
    md_path = reports_dir / f"{display}.md"
    json_path = reports_dir / f"{display}.json"

    ts = None
    if json_path.exists():
        ts = _read_json_timestamp(json_path)
    if not ts:
        ts = datetime.fromtimestamp(md_path.stat().st_mtime).isoformat() if md_path.exists() else datetime.now().isoformat()

    safe_ts = ts.replace(":", "").replace(" ", "T")
    archive_dir.mkdir(parents=True, exist_ok=True)

    for p in (md_path, json_path):
        if p.exists():
            p.rename(archive_dir / f"{p.stem}_{safe_ts}{p.suffix}")


def save_report(ticker: str, analyst_signals: dict, decision: dict,
                current_price: float, metadata: dict, base_dir: str = "reports") -> None:
    """Save Markdown + JSON report for one ticker. Previous report moves to archive/."""
    display = ticker.replace(".NS", "").replace(".BO", "")
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M IST")
    model = metadata.get("model_name", "unknown")
    provider = metadata.get("model_provider", "unknown")

    reports_dir = Path(base_dir)
    archive_dir = reports_dir / "archive"
    reports_dir.mkdir(parents=True, exist_ok=True)

    risk = analyst_signals.get("risk_management_agent", {})
    risk_data = risk.get(ticker, {})
    price = risk_data.get("current_price", current_price) or current_price

    _archive_old(reports_dir, archive_dir, display)

    _write_md(reports_dir / f"{display}.md", display, now, date_str, model, provider,
              price, analyst_signals, decision)
    _write_json(reports_dir / f"{display}.json", display, now, date_str, model, provider,
                price, analyst_signals, decision)


def _write_md(path: Path, display: str, now: datetime, date_str: str, model: str,
              provider: str, price: float, analyst_signals: dict, decision: dict) -> None:
    lines = [
        f"# {display} — AI Hedge Fund Analysis",
        "",
        f"**Date:** {date_str}",
        f"**Model:** {model} ({provider})",
        f"**Price:** ₹{price:,.2f}",
        "",
        "---",
        "",
        "## Portfolio Decision",
    ]

    action = decision.get("action", "unknown")
    qty = decision.get("quantity", 0)
    conf = decision.get("confidence", 0)
    reasoning = _str_val(decision.get("reasoning", ""))

    lines.append(f"- **Action:** {action.upper()} {qty} shares")
    lines.append(f"- **Confidence:** {conf}%")
    lines.append(f"- **Reasoning:** {reasoning}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Agent Signals")
    lines.append("")
    lines.append("| Agent | Signal | Confidence |")
    lines.append("|-------|--------|------------|")

    signal_bullets: list[tuple[str, str, float, str]] = []
    for agent_name, signals in analyst_signals.items():
        if agent_name.startswith("risk_management") or ticker_ns(display) not in signals:
            continue
        sig = signals[ticker_ns(display)]
        signal_str = sig.get("signal", "unknown")
        confidence_val = sig.get("confidence", 0)
        reasoning_text = _str_val(sig.get("reasoning", ""))
        agent_display = agent_name.replace("_agent", "").replace("_", " ").title()
        emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(signal_str, "⚪")
        lines.append(f"| {agent_display} | {emoji} {signal_str} | {confidence_val:.0f}% |")
        if reasoning_text:
            signal_bullets.append((agent_display, signal_str, confidence_val, reasoning_text))

    lines.append("")

    if signal_bullets:
        lines.append("---")
        lines.append("")
        lines.append("## Agent Reasonings")
        lines.append("")
        for agent_display, signal_str, confidence_val, reasoning_text in signal_bullets:
            emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(signal_str, "⚪")
            lines.append(f"### {agent_display} — {emoji} {signal_str} ({confidence_val:.0f}%)")
            lines.append("")
            lines.append(reasoning_text)
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, display: str, now: datetime, date_str: str, model: str,
                provider: str, price: float, analyst_signals: dict, decision: dict) -> None:
    agents_out = {}
    for agent_name, signals in analyst_signals.items():
        if agent_name.startswith("risk_management"):
            key = ticker_ns(display)
            entry = signals.get(key, {})
            agents_out[agent_name.replace("_agent", "")] = {
                "signal": entry.get("signal", "n/a"),
                "confidence": entry.get("confidence"),
                "reasoning": _str_val(entry.get("reasoning")),
                "remaining_position_limit": entry.get("remaining_position_limit"),
                "current_price": entry.get("current_price"),
                "daily_volatility": entry.get("daily_volatility"),
                "annualized_volatility": entry.get("annualized_volatility"),
                "volatility_percentile": entry.get("volatility_percentile"),
            }
        else:
            key = ticker_ns(display)
            if key not in signals:
                continue
            sig = signals[key]
            agents_out[agent_name.replace("_agent", "")] = {
                "signal": sig.get("signal", "unknown"),
                "confidence": sig.get("confidence", 0),
                "reasoning": _str_val(sig.get("reasoning", "")),
            }

    report = {
        "ticker": display,
        "timestamp": now.isoformat(),
        "date": date_str,
        "model": model,
        "provider": provider,
        "price": price,
        "decision": {
            "action": decision.get("action", "unknown"),
            "quantity": decision.get("quantity", 0),
            "confidence": decision.get("confidence", 0),
            "reasoning": _str_val(decision.get("reasoning", "")),
        },
        "agents": agents_out,
    }

    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def ticker_ns(display: str) -> str:
    return f"{display}.NS" if not display.endswith(".NS") and not display.endswith(".BO") else display
