"""Helper functions for LLM calls with retry and JSON parsing."""

import json
import logging

from pydantic import BaseModel

from src.llm.models import get_llm, has_json_mode

logger = logging.getLogger(__name__)


def call_llm(
    prompt: any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    model_name: str = "deepseek-chat",
    model_provider: str = "DeepSeek",
    api_keys: dict = None,
    max_retries: int = 3,
    default_factory=None,
    state=None,
) -> BaseModel:
    """Call an LLM with retries and structured output parsing.

    Args:
        prompt: The prompt to send
        pydantic_model: Pydantic model for structured output
        agent_name: Optional agent name for logging
        model_name: Model name (default: deepseek-chat)
        model_provider: Provider name (default: DeepSeek)
        api_keys: Optional dict of API keys
        max_retries: Max retry attempts
        default_factory: Factory for default response on failure
        state: Optional AgentState dict for backward compat (extracts model config)

    Returns:
        Instance of pydantic_model
    """
    if state is not None:
        metadata = state.get("metadata", {})
        model_name = metadata.get("model_name", model_name)
        model_provider = metadata.get("model_provider", model_provider)
        request = metadata.get("request")
        if request and hasattr(request, "api_keys") and request.api_keys:
            api_keys = request.api_keys or api_keys

    llm = get_llm(model_name, model_provider, api_keys)

    if has_json_mode(model_provider):
        llm = llm.with_structured_output(pydantic_model, method="json_mode")

    for attempt in range(max_retries):
        try:
            result = llm.invoke(prompt)

            if not has_json_mode(model_provider):
                parsed = _extract_json(result.content)
                if parsed:
                    return pydantic_model(**parsed)
            else:
                return result

        except Exception as e:
            logger.warning("%s: LLM call attempt %d/%d failed: %s", agent_name or "agent", attempt + 1, max_retries, e)

            if attempt == max_retries - 1:
                logger.error("%s: All %d retries failed", agent_name or "agent", max_retries)
                if default_factory:
                    return default_factory()
                return _create_default(pydantic_model)

    return _create_default(pydantic_model)


def _extract_json(content) -> dict | None:
    """Extract JSON from LLM response, handling markdown code blocks."""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        content = "\n".join(parts)

    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7:]
            json_end = json_text.find("```")
            if json_end != -1:
                return json.loads(json_text[:json_end].strip())
    except Exception:
        pass

    try:
        json_start = content.find("```")
        if json_start != -1:
            json_text = content[json_start + 3:]
            json_end = json_text.find("```")
            if json_end != -1:
                return json.loads(json_text[:json_end].strip())
    except Exception:
        pass

    try:
        return json.loads(content.strip())
    except Exception:
        pass

    try:
        brace_start = content.find("{")
        if brace_start != -1:
            depth = 0
            for i, char in enumerate(content[brace_start:], brace_start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(content[brace_start:i + 1])
    except Exception:
        pass

    return None


def _create_default(model_class: type[BaseModel]) -> BaseModel:
    """Create a safe default instance of a Pydantic model."""
    defaults = {}
    for field_name, field in model_class.model_fields.items():
        ann = field.annotation
        if ann is str:
            defaults[field_name] = "Error in analysis, using default"
        elif ann is float:
            defaults[field_name] = 0.0
        elif ann is int:
            defaults[field_name] = 0
        elif hasattr(ann, "__origin__") and ann.__origin__ is dict:
            defaults[field_name] = {}
        elif hasattr(ann, "__args__"):
            defaults[field_name] = ann.__args__[0]
        else:
            defaults[field_name] = None
    return model_class(**defaults)
