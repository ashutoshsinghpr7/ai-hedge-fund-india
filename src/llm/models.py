"""LLM model definitions and factory for ai-hedge-fund-india.

Supports DeepSeek (default), OpenAI, Anthropic, Groq, Google, and Ollama.
"""

import os
from enum import Enum

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


class ModelProvider(str, Enum):
    DEEPSEEK = "DeepSeek"
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    GROQ = "Groq"
    GOOGLE = "Google"
    OLLAMA = "Ollama"


def _get_deepseek(model_name: str, api_keys: dict = None) -> ChatOpenAI:
    api_key = (api_keys or {}).get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set in .env")
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.3,
    )


def _get_openai(model_name: str, api_keys: dict = None):
    api_key = (api_keys or {}).get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env")
    base_url = os.getenv("OPENAI_API_BASE")
    return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0.3)


def _get_anthropic(model_name: str, api_keys: dict = None):
    api_key = (api_keys or {}).get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")
    return ChatAnthropic(model=model_name, api_key=api_key, temperature=0.3)


def _get_groq(model_name: str, api_keys: dict = None):
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError("langchain-groq not installed. Run: pip install langchain-groq")
    api_key = (api_keys or {}).get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env")
    return ChatGroq(model=model_name, api_key=api_key, temperature=0.3)


def _get_google(model_name: str, api_keys: dict = None):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError("langchain-google-genai not installed. Run: pip install langchain-google-genai")
    api_key = (api_keys or {}).get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in .env")
    return ChatGoogleGenerativeAI(model=model_name, api_key=api_key, temperature=0.3)


def _get_ollama(model_name: str, api_keys: dict = None):
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError("langchain-ollama not installed. Run: pip install langchain-ollama")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(model=model_name, base_url=base_url, temperature=0.3)


_PROVIDER_FACTORIES = {
    ModelProvider.DEEPSEEK: _get_deepseek,
    ModelProvider.OPENAI: _get_openai,
    ModelProvider.ANTHROPIC: _get_anthropic,
    ModelProvider.GROQ: _get_groq,
    ModelProvider.GOOGLE: _get_google,
    ModelProvider.OLLAMA: _get_ollama,
}


def get_llm(
    model_name: str = "deepseek-chat",
    model_provider: str = "DeepSeek",
    api_keys: dict = None,
):
    provider = ModelProvider(model_provider)
    factory = _PROVIDER_FACTORIES.get(provider)
    if factory is None:
        raise ValueError(f"Unsupported provider: {model_provider}. Supported: {[p.value for p in ModelProvider]}")
    return factory(model_name, api_keys)


def has_json_mode(model_provider: str) -> bool:
    """Check if the model supports native JSON structured output."""
    provider = ModelProvider(model_provider)
    return provider not in (ModelProvider.DEEPSEEK, ModelProvider.GOOGLE, ModelProvider.OLLAMA)
