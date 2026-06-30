# Changelog

## [0.1.0] - 2026-06-30

### Added

- **22 AI agents**: 13 personality agents (Warren Buffett, Ben Graham, Bill Ackman, Cathie Wood, Charlie Munger, Michael Burry, Mohnish Pabrai, Nassim Taleb, Peter Lynch, Phil Fisher, Stanley Druckenmiller, Aswath Damodaran, Rakesh Jhunjhunwala) + 6 utility agents (Technical, Fundamentals, Sentiment, Valuation, Growth, News Sentiment) + 3 India-specific (Radhakishan Damani, FII/DII Flow, RBI Policy)
- **YFinanceProvider**: Free NSE/BSE data via yfinance (prices, financials, news, market cap)
- **Sector-based discovery**: `--sector pharma` discovers all stocks in a sector using static map + yfinance dynamic lookup
- **CLI**: `hedgefund run --ticker RELIANCE` / `hedgefund scan --sector banking` / `hedgefund analyze --ticker RELIANCE`
- **LLM providers**: DeepSeek (default), OpenAI, Anthropic, Groq, Google, Ollama
- **LangGraph workflow**: Parallel agent execution → Risk Manager → Portfolio Manager
- **Indian market costs**: STT, stamp duty, GST, SEBI charges in config
- **Nifty 50/100 universe** with sector classification
- **Pickle cache** for data provider responses

### Attribution

Adapted from [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) (MIT License).
