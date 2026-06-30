# ai-hedge-fund-india

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Poetry](https://img.shields.io/badge/Poetry-✓-green)](https://python-poetry.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/Lint-Ruff-cyan)](https://github.com/astral-sh/ruff)

AI-powered hedge fund for Indian **NSE/BSE** markets — 22 AI agents analyze stocks with distinct investment philosophies, from Buffett to Jhunjhunwala. Adapted from [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) (MIT).

> **Educational purposes only.** Not intended for real trading or investment.

## Quick Start

```bash
git clone https://github.com/ashutoshsinghpr7/ai-hedge-fund-india.git
cd ai-hedge-fund-india
poetry install
cp .env.example .env     # add your LLM API key (DeepSeek recommended — ₹0.02/1M tokens)

# Run analysis
poetry run hedgefund run --ticker RELIANCE,TCS

# Analyze an entire sector
poetry run hedgefund run --sector pharma

# See agent reasoning
poetry run hedgefund run --ticker RELIANCE --show-reasoning

# Quick scan
poetry run hedgefund scan --sector banking
```

Output:

```
━━━ RELIANCE ━━━                         ━━━ TCS ━━━
 Agent          Signal     Confidence     Agent          Signal     Confidence
 warren_buffett  bullish      80%         warren_buffett  bullish      75%
 ben_graham      neutral      55%         ben_graham      neutral      50%
 rakesh_jhunjhunwala  bullish  85%        rakesh_jhunjhunwala  bullish  80%
 ...
 Portfolio Manager: HOLD 0 shares        Portfolio Manager: BUY 500 shares (75%)
```

Reports auto-save to `reports/RELIANCE.md` + `reports/RELIANCE.json`. Previous reports move to `reports/archive/`. [→ View sample report](samples/KSHINTL-report.md)

## Why This Repo?

| Feature | Original (virattt) | ai-hedge-fund-india |
|---------|-------------------|---------------------|
| Market | US (NYSE/NASDAQ) | Indian NSE/BSE |
| Data source | Financial Datasets API (paid) | yfinance (free, `.NS` suffix) |
| Agents | 11 agents (8 personality + 3 utility) | **22 agents** (13 personality + 6 utility + 3 India-specific) |
| Default LLM | OpenAI GPT-4o (expensive) | **DeepSeek** (~₹0.02/1M tokens) |
| Stock discovery | Manual ticker entry | Sector-based auto-discovery (pharma, banking, auto, etc.) |
| Report generation | None | Per-ticker Markdown + JSON with archive |
| India-specific | None | Rakesh Jhunjhunwala, Radhakishan Damani, FII/DII Flow, RBI Policy |
| Rolled-up costs | Basic | STT, Stamp Duty, GST, SEBI charges configured |

## Architecture

```
CLI (hedgefund run|scan|analyze)
    │
    ▼
LangGraph Workflow
    │
    ├─── 22 Agent Nodes (parallel) ─────┐
    │    ├─ Personality (13) ──→ LLM     │
    │    ├─ Utility (6) ──→ Pure calc    │
    │    └─ India-specific (3) ──→ LLM   │
    │                                    │
    ├─── Risk Manager ◄──────────────────┘
    │
    └─── Portfolio Manager → BUY/SELL/HOLD
         │
         ▼
    Data Layer: YFinanceProvider → Pickle Cache
```

## Agents (22)

### Personality Agents (13)

| Agent | Philosophy | LLM |
|-------|-----------|-----|
| Warren Buffett | Wonderful companies at fair prices | Yes |
| Ben Graham | Father of value investing, margin of safety | Yes |
| Bill Ackman | Activist value investor | Yes |
| Cathie Wood | Disruptive innovation, growth at any price | Yes |
| Charlie Munger | Mental models, wonderful businesses | Yes |
| Michael Burry | The Big Short contrarian, deep value | Yes |
| Mohnish Pabrai | Dhando investing, heads-I-win-tails-I-don't-lose-much | Yes |
| Nassim Taleb | Black Swan risk, antifragility, asymmetric payoffs | Yes |
| Peter Lynch | GARP, ten-baggers in everyday businesses | Yes |
| Phil Fisher | Quality growth, scuttlebutt research | Yes |
| Stanley Druckenmiller | Macro momentum, asymmetric opportunities | Yes |
| Aswath Damodaran | Dean of Valuation, story + numbers | Yes |
| Rakesh Jhunjhunwala | The Big Bull of India | Yes |

### India-Specific Agents (3)

| Agent | Focus | LLM |
|-------|-------|-----|
| Radhakishan Damani | Retail superinvestor, deep value, D-MART style | Yes |
| FII/DII Flow | Institutional money flows (FII buying/selling pressure) | Yes |
| RBI Policy | Monetary policy impact (repo rate, INR/USD, G-Sec yields) | Yes |

### Utility Agents (6)

| Agent | Focus | LLM |
|-------|-------|-----|
| Technical Analyst | RSI, MACD, Bollinger Bands, volume profile | No |
| Fundamentals Analyst | ROE, margins, D/E, profitability growth | No |
| Sentiment Analyst | Price-based sentiment indicators | No |
| Valuation Analyst | DCF with Indian WACC, owner earnings | No |
| Growth Analyst | Revenue/earnings growth trajectories | No |
| News Sentiment | Company news sentiment analysis | Yes |

## Commands

```bash
# Analyze specific stocks
poetry run hedgefund run --ticker RELIANCE,TCS,HDFCBANK

# Analyze an entire sector (auto-discovers stocks)
poetry run hedgefund run --sector pharma
poetry run hedgefund run --sector banking

# Scan a sector (quick price-only scan)
poetry run hedgefund scan --sector energy --limit 10

# Quick data fetch (no AI agents)
poetry run hedgefund analyze --ticker RELIANCE

# Show agent reasoning
poetry run hedgefund run --ticker RELIANCE --show-reasoning

# Custom dates and model
poetry run hedgefund run --ticker RELIANCE --start 2023-01-01 --model gpt-4o --provider OpenAI

# Skip report saving
poetry run hedgefund run --ticker RELIANCE --no-save
```

## Supported Sectors

`agriculture, auto, aviation, banking, cement, chemicals, construction, consumer, defence, energy, financial_services, hospitality, industrial, infrastructure, insurance, metals, pharma, real_estate, technology, telecom, tourism`

Sector discovery: static map (100 Nifty 100 stocks) + dynamic yfinance lookup with `.cache/sector_cache.pkl`.

## LLM Providers

| Provider | Model | Cost (per 1M tokens) |
|----------|-------|---------------------|
| DeepSeek (default) | `deepseek-chat` | ~₹0.02 |
| OpenAI | `gpt-4o`, `gpt-4o-mini` | ~₹1-5 |
| Anthropic | `claude-3-5-sonnet` | ~₹3 |
| Groq | `llama-3.3-70b` | Free tier |
| Google | `gemini-2.0-flash` | Free tier |
| Local | Ollama models | Free |

Set via `--provider DeepSeek --model deepseek-chat` or `<PROVIDER>_API_KEY` in `.env`.

## Project Structure

```
ai-hedge-fund-india/
├── src/
│   ├── main.py              # CLI entry point
│   ├── agents/              # 22 agent implementations
│   ├── data/                # Data provider + models + universe
│   │   ├── providers/       # YFinanceProvider (Zerodha planned)
│   │   ├── models.py        # Pydantic data models
│   │   ├── universe.py      # Stock universe + sector discovery
│   │   └── cache.py         # Pickle cache
│   ├── tools/               # Public API layer (caching proxy)
│   ├── graph/               # LangGraph workflow + state
│   ├── llm/                 # LLM provider configs
│   └── utils/               # Analysts registry, LLM helpers, reporter, progress
├── config/                  # YAML configuration
├── tests/                   # Test suite
├── pyproject.toml
└── README.md
```

## Indian Market Specifics

- **Data source**: yfinance (free, NSE `.NS` suffix)
- **Costs configured**: STT (0.1%), Stamp Duty (0.015%), GST (18%), SEBI charges
- **Universe**: Nifty 50 + Nifty Next 50 (100 stocks) with sector map
- **India-specific agents**: Rakesh Jhunjhunwala, Radhakishan Damani, FII/DII Flow, RBI Policy
- **Reports**: Per-ticker Markdown + JSON with auto-archive on re-run

## Roadmap

- [ ] Zerodha Kite Connect integration for live data
- [ ] Indian backtester with STT/circuit filters
- [ ] Walk-forward optimization
- [ ] Monte Carlo simulation
- [ ] Web UI (FastAPI + React)
- [ ] BSE `.BO` suffix support

## Attribution

Adapted from [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) (MIT License). The original project pioneered the multi-agent LLM-based trading system architecture. This adaptation adds Indian market data, India-specific agents, sector-based discovery, and Indian market costs.

## License

MIT — see [LICENSE](LICENSE) for details. Original work [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) also MIT.

---

**Keywords**: AI hedge fund, algorithmic trading India, NSE, BSE, LangGraph, multi-agent trading, stock market India, DeepSeek, value investing, Indian stock analysis, LLM trading, yfinance India, open source trading bot, Warren Buffett bot, Rakesh Jhunjhunwala, FII DII flow, RBI policy trading, sector discovery, quantitative finance, LangChain
