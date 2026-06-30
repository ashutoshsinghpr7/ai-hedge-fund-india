# Contributing

Thanks for your interest in contributing.

## Setup

```bash
git clone https://github.com/YOUR_USER/ai-hedge-fund-india.git
cd ai-hedge-fund-india
poetry install
cp .env.example .env
# Add your LLM API keys to .env
```

## Quality checks

```bash
poetry run ruff check .       # Lint
poetry run pytest tests/ -v   # Tests
```

## Adding a new agent

1. Create `src/agents/your_agent.py` following the existing pattern:
   - Define a Pydantic signal model for LLM output
   - Implement your `agent(state, agent_id)` function
   - Compute scores from data, then call `call_llm()` for the final decision
2. Register in `src/utils/analysts.py`:
   ```python
   from src.agents.your_agent import your_agent

   "your_agent": {
       "display_name": "Your Agent Name",
       "description": "Brief description",
       "agent_func": your_agent,
       "order": N,
   }
   ```
3. No other files need changes — the LangGraph workflow auto-discovers agents from the registry.

## Coding conventions

- Python 3.10+ with type hints
- PEP 8 via ruff
- New agents inherit the existing data fetching pattern (use `src.tools` API)

## Attribution

This project is adapted from [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) (MIT).
Please preserve the original attribution when contributing agents from that project.
