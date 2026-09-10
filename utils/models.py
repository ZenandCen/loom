"""LLM Catalog — pre-configured model instances + usage tracking + fallback.

Usage:
    from utils.models import LLM, get_llm, get_llm_with_fallback
    from utils.llm_usage import usage_tracker

    llm = get_llm(LLM.OPENAI)
    llm = get_llm_with_fallback(LLM.OPENAI)  # auto-fallback if primary fails

    # Check usage at end of session:
    print(usage_tracker.report())
"""

import os
import logging
import warnings
from enum import Enum

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import BaseChatModel

# Suppress known non-actionable warnings from provider SDKs
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")
warnings.filterwarnings("ignore", message=".*Direct use of automatic function calling.*")
warnings.filterwarnings("ignore", message=".*AFC in Models.generate_content.*")

# Suppress google_genai AFC deprecation log
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

load_dotenv(override=True)


# ============================================================
# LLM Enum
# ============================================================

class LLM(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    FALLBACK = "fallback"  # llama3.1:latest (local, always available)


# ============================================================
# Model Instances
# ============================================================

_openai_model = ChatOpenAI(
    model=os.getenv("LOOM_MODEL_OPENAI", "qwen/qwen3.8-27b"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://code-agent.cads.live/v1",
    temperature=0,
)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    _gemini_model = ChatGoogleGenerativeAI(
        model=os.getenv("LOOM_MODEL_GEMINI", "gemini-3.5-flash-lite"),
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
except ImportError:
    _gemini_model = None

try:
    from langchain_ollama import ChatOllama

    _ollama_model = ChatOllama(
        model=os.getenv("LOOM_MODEL_OLLAMA", "gemma:latest"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )

    _fallback_model = ChatOllama(
        model=os.getenv("LOOM_MODEL_FALLBACK", "llama3.1:latest"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )
except ImportError:
    _ollama_model = None
    _fallback_model = None


# ============================================================
# Registry
# ============================================================

_REGISTRY: dict[LLM, BaseChatModel] = {
    LLM.OPENAI: _openai_model,
    LLM.GEMINI: _gemini_model,
    LLM.OLLAMA: _ollama_model,
    LLM.FALLBACK: _fallback_model,
}


def get_llm(name: LLM) -> BaseChatModel:
    """Get model instance by LLM name."""
    instance = _REGISTRY.get(name)
    if instance is None:
        raise ValueError(f"LLM '{name}' not available. Check API key / install package.")
    return instance


def get_llm_with_fallback(name: LLM, fallback: LLM = LLM.FALLBACK) -> BaseChatModel:
    """Get a model that auto-falls-back to a local LLM if the primary fails.

    Tries the primary model first. If it raises an error (quota exhausted,
    rate limit, network down), logs a warning and returns the fallback model.

    This is a one-shot check at creation time. For per-call fallback,
    use FallbackLLM wrapper.

    Args:
        name: Primary LLM to use.
        fallback: Fallback LLM (default: LLM.FALLBACK / llama3.1:latest).

    Returns:
        A BaseChatModel instance.
    """
    from utils.llm_usage import usage_tracker

    try:
        primary = get_llm(name)
        # Quick connectivity test
        primary.invoke("Say ok")
        usage_tracker.record_call(name.value, success=True)
        return primary
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"LLM '{name.value}' unavailable ({e}). Falling back to '{fallback.value}'.")
        usage_tracker.record_call(name.value, success=False, error=str(e))
        fallback_model = get_llm(fallback)
        usage_tracker.record_call(fallback.value, success=True, note=f"fallback for {name.value}")
        return fallback_model
