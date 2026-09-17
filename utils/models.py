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
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

# Suppress known non-actionable warnings from provider SDKs
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")
warnings.filterwarnings("ignore", message=".*Direct use of automatic function calling.*")
warnings.filterwarnings("ignore", message=".*AFC in Models.generate_content.*")

# Suppress google_genai AFC deprecation log
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

load_dotenv(override=True)


# ============================================================
# LLM Enum
# ============================================================

class LLM(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    VISION = "vision"
    FALLBACK = "fallback"  # gemma:latest (local, last-resort, always available)


# ============================================================
# Model Instances
# ============================================================

_openai_model = ChatOpenAI(
    model=os.getenv("LOOM_MODEL_OPENAI", "qwen/qwen3.8-27b"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://code-agent.cads.live/v1",
    temperature=0,
    max_tokens=32768,
)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    _gemini_model = ChatGoogleGenerativeAI(
        model=os.getenv("LOOM_MODEL_GEMINI", "gemini-3.5-flash-lite"),
        api_key=os.environ.get("GEMINI_API_KEY"),
        max_output_tokens=8192,
    )
except ImportError:
    _gemini_model = None

try:
    from langchain_ollama import ChatOllama

    _ollama_model = ChatOllama(
        model=os.getenv("LOOM_MODEL_OLLAMA", "gemma:latest"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
        num_ctx=8192,
        num_predict=4096,
    )

    _fallback_model = ChatOllama(
        model=os.getenv("LOOM_MODEL_FALLBACK", "gemma:latest"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
        num_ctx=8192,
        num_predict=4096,
    )
except ImportError:
    _ollama_model = None
    _fallback_model = None

_vision_model: ChatOpenAI | None = None
if os.getenv("LOOM_MODEL_VISION"):
    _vision_model = ChatOpenAI(
    model=os.getenv("LOOM_MODEL_VISION", "qwen-vl-max"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://code-agent.cads.live/v1",
    temperature=0,
    max_tokens=32768,
)


# ============================================================
# Registry
# ============================================================

_REGISTRY: dict[LLM, BaseChatModel] = {
    LLM.OPENAI: _openai_model,
    LLM.GEMINI: _gemini_model,
    LLM.OLLAMA: _ollama_model,
    LLM.VISION: _vision_model,
    LLM.FALLBACK: _fallback_model,
}


def get_llm(name: LLM) -> BaseChatModel:
    """Get model instance by LLM name."""
    instance = _REGISTRY.get(name)
    if instance is None:
        raise ValueError(f"LLM '{name}' not available. Check API key / install package.")
    return instance


class FallbackChatModel(BaseChatModel):
    """Chat model that tries each sub-model in priority order and returns the first
    successful result.

    Supports both .invoke()/.ainvoke() and .bind_tools() (so create_agent / ReAct
    agents can use it directly). A single model outage (quota, rate limit, network)
    therefore never crashes a node — the call rolls to the next model in the chain.
    """

    _chain: list = PrivateAttr(default=[])

    def __init__(self, chain: list, **kwargs):
        # chain: list of (name, BaseChatModel) in priority order
        super().__init__(**kwargs)
        self._chain = chain

    @property
    def _llm_type(self) -> str:
        return "fallback_chat_model"

    @property
    def _identifying_params(self) -> dict:
        return {"chain": [name for name, _ in self._chain]}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Delegate via invoke() (not the private _generate or .generate): invoke is the
        # only path that parses provider tool-call payloads into message.tool_calls.
        # The nested run's callbacks are suppressed (callbacks=[]) so the tracer doesn't
        # double-log the sub-model.
        last_err: "Exception | None" = None
        for name, model in self._chain:
            try:
                msg = model.invoke(list(messages), config={"callbacks": []}, stop=stop)
                return ChatResult(generations=[ChatGeneration(message=msg)])
            except Exception as e:
                last_err = e
                logger.warning(f"[FALLBACK] '{name}' failed ({type(e).__name__}: {e}); trying next model")
        raise last_err

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        last_err: "Exception | None" = None
        for name, model in self._chain:
            try:
                msg = await model.ainvoke(list(messages), config={"callbacks": []}, stop=stop)
                return ChatResult(generations=[ChatGeneration(message=msg)])
            except Exception as e:
                last_err = e
                logger.warning(f"[FALLBACK] (async) '{name}' failed ({type(e).__name__}: {e}); trying next model")
        raise last_err

    def bind_tools(self, tools, **kwargs):
        bound = [(name, model.bind_tools(tools, **kwargs)) for name, model in self._chain]
        return FallbackChatModel(chain=bound)


def get_backbone_llm() -> BaseChatModel:
    """Primary 'answer' model with automatic fallback: qwen -> gemini -> gemma(local).

    Used by the planner (routing), synthesizer, and every worker so a single model
    outage can never crash a node.
    """
    chain: list = []
    for name in (LLM.OPENAI, LLM.GEMINI, LLM.FALLBACK):
        try:
            model = get_llm(name)
        except Exception:
            model = None
        if model is not None:
            chain.append((name.value, model))
    if not chain:
        raise ValueError("No LLM available for backbone (openai/gemini/fallback all missing)")
    return FallbackChatModel(chain=chain)


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
