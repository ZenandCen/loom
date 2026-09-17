"""Execution tracer — lets you READ the AI's thinking during task execution.

A LangChain callback handler that logs, as the pipeline runs:
  • LLM decisions   -> [TRACE:DECIDE]  (which tools the model chose to call + args)
  • reasoning       -> [TRACE:THINK]   (reasoning_content, when the model emits it)
  • LLM output      -> [TRACE:LLM]     (what the model actually said)
  • tool calls      -> [TRACE:TOOL]    (name, input, and result of each tool run)

Attach it once to the top-level runnable; LangChain propagates it to ALL nested
runs (planner, workers, ReAct agents, and every tool they call):

    await team.ainvoke(input, config={"callbacks": [LoomTracer()]})

Config via env:
    LOOM_TRACE=0/1      enable / disable (default: on)
    LOOM_TRACE_MAX=N    max chars per logged payload (default: 2000)
"""

import logging
import os

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("loom.trace")


def _clip(text: str, max_len: int) -> str:
    text = text or ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f" …(+{len(text) - max_len} chars)"


def _short_args(args, n: int = 200) -> str:
    try:
        s = str(args)
        return s if len(s) <= n else s[:n] + "…"
    except Exception:
        return ""


def _msg_text(message) -> str:
    """Plain text from a chat message (content may be a str or list of parts)."""
    c = getattr(message, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get("text", ""))
        return " ".join(parts)
    return str(c)


class LoomTracer(BaseCallbackHandler):
    """Logs LLM reasoning + tool calls. Safe: never raises into the pipeline."""

    always_thread_separate = False

    def __init__(self):
        self.max_len = int(os.getenv("LOOM_TRACE_MAX", "2000"))
        self.enabled = os.getenv("LOOM_TRACE", "1") == "1"
        # Content-based dedup: the fallback delegation's nested sub-model run re-emits
        # the same end-event as its parent, so consecutive identical payloads collapse
        # into one line.
        self._last_sig = None

    # ---------- LLM / chat model ----------
    def on_chat_model_start(self, serialized, messages, *, run_name=None, **kwargs):
        if not self.enabled:
            return
        try:
            model = (serialized or {}).get("name") or run_name or "chat"
            logger.info(f"[TRACE:LLM:{model}] ▶ thinking …")
        except Exception:
            pass

    def on_llm_start(self, serialized, prompts, *, run_name=None, **kwargs):
        if not self.enabled:
            return
        try:
            model = (serialized or {}).get("name") or run_name or "llm"
            logger.info(f"[TRACE:LLM:{model}] ▶ thinking …")
        except Exception:
            pass

    def _log_generation(self, response) -> bool:
        """Log reasoning / tool-decision / output of a generation.
        Consecutive identical payloads (nested sub-model re-emit) are collapsed."""
        try:
            gen = response.generations[0][0]
        except Exception:
            return False
        message = getattr(gen, "message", None)
        # chat models carry content in gen.message; plain LLMs in gen.text
        text = (_msg_text(message) if message is not None else (getattr(gen, "text", "") or "")).strip()

        # reasoning content (Qwen/o1-style thinking) if the model emits it
        reasoning = ""
        gi = getattr(gen, "generation_info", None) or {}
        reasoning = (gi.get("reasoning_content") or gi.get("reasoning") or "").strip()
        if message is not None:
            ak = getattr(message, "additional_kwargs", None) or {}
            reasoning = reasoning or (ak.get("reasoning_content") or "").strip()

        # the agent's decision: which tool(s) it wants to call next
        tcs = getattr(message, "tool_calls", None) or []
        tool_desc = ", ".join(f"{tc.get('name')}({_short_args(tc.get('args'))})" for tc in tcs)

        # Collapse consecutive duplicates (nested delegation run re-emits the parent's).
        sig = f"{reasoning}\x00{tool_desc}\x00{text}"
        if sig == self._last_sig:
            return False
        self._last_sig = sig

        logged = False
        if reasoning:
            logger.info(f"[TRACE:THINK] {_clip(reasoning, self.max_len)}")
            logged = True
        if tool_desc:
            logger.info(f"[TRACE:DECIDE] → call tools: {tool_desc}")
            logged = True
        if text:
            logger.info(f"[TRACE:LLM] ◀ {_clip(text, self.max_len)}")
            logged = True
        return logged

    def _handle_end(self, response):
        try:
            self._log_generation(response)
        except Exception as e:
            logger.debug(f"tracer end failed: {e}")

    def on_chat_model_end(self, response, *, run_id=None, **kwargs):
        if self.enabled:
            self._handle_end(response)

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        # NOTE: FallbackChatModel emits on_llm_end (not on_chat_model_end).
        if self.enabled:
            self._handle_end(response)

    # ---------- tools ----------
    def on_tool_start(self, serialized, input_str, *, run_name=None, run_id=None, **kwargs):
        if not self.enabled:
            return
        try:
            name = (serialized or {}).get("name") or run_name or "tool"
            logger.info(f"[TRACE:TOOL:{name}] ▶ {_clip(str(input_str), self.max_len)}")
        except Exception:
            pass

    def on_tool_end(self, output, *, run_id=None, **kwargs):
        if not self.enabled:
            return
        try:
            logger.info(f"[TRACE:TOOL] ◀ {_clip(str(output), self.max_len)}")
        except Exception:
            pass

    def on_tool_error(self, error, *, run_id=None, **kwargs):
        if not self.enabled:
            return
        try:
            logger.warning(f"[TRACE:TOOL] ✗ {error}")
        except Exception:
            pass
