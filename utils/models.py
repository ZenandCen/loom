"""LLM Catalog — pre-configured model instances.

Usage:
    from utils.models import LLM, get_llm

    llm = get_llm(LLM.OPENAI)
    llm = get_llm(LLM.GEMINI)
"""

import os
from enum import Enum

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import BaseChatModel

load_dotenv(override=True)


# ============================================================
# LLM Enum
# ============================================================

class LLM(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


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
        model=os.getenv("LOOM_MODEL_GEMINI", "gemini-2.0-flash"),
        api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0,
    )
except ImportError:
    _gemini_model = None


# ============================================================
# Registry
# ============================================================

_REGISTRY: dict[LLM, BaseChatModel] = {
    LLM.OPENAI: _openai_model,
    LLM.GEMINI: _gemini_model,
}


def get_llm(name: LLM) -> BaseChatModel:
    """Get model instance by LLM name."""
    instance = _REGISTRY.get(name)
    if instance is None:
        raise ValueError(f"LLM '{name}' not available. Check API key / install package.")
    return instance
