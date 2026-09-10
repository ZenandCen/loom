"""LLM Usage Tracker — monitors calls, tokens, errors, and latency per model.

Usage:
    from utils.llm_usage import usage_tracker

    # After each LLM call:
    usage_tracker.record_call("openai", success=True, latency_ms=1200)
    usage_tracker.record_call("gemini", success=False, error="429 Rate limit")

    # Print summary at end of session:
    print(usage_tracker.report())

    # Check if a model is exhausted:
    if usage_tracker.is_exhausted("gemini"):
        print("Switch to fallback!")
"""

import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMStats:
    """Usage stats for a single LLM."""
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    last_error: str = ""
    last_call_time: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.successes == 0:
            return 0.0
        return self.total_latency_ms / self.successes

    @property
    def success_rate(self) -> float:
        if self.calls == 0:
            return 0.0
        return self.successes / self.calls

    def is_exhausted(self, error_threshold: int = 3) -> bool:
        """Check if model appears exhausted (consecutive errors or quota errors)."""
        if self.calls == 0:
            return False
        # Too many recent errors
        recent_errors = self.errors[-5:]
        error_count = len(recent_errors)
        if error_count >= error_threshold:
            return True
        # Specific quota/rate-limit keywords
        quota_keywords = ["quota", "rate limit", "429", "403", "billing", "exhausted", "no longer available"]
        if self.last_error:
            last_err_lower = self.last_error.lower()
            if any(kw in last_err_lower for kw in quota_keywords):
                return True
        return False


class LLMUsageTracker:
    """Tracks usage across all LLMs in the session."""

    def __init__(self):
        self._stats: dict[str, LLMStats] = {}

    def _get_stats(self, name: str) -> LLMStats:
        if name not in self._stats:
            self._stats[name] = LLMStats()
        return self._stats[name]

    def record_call(
        self,
        name: str,
        success: bool,
        latency_ms: float | None = None,
        error: str = "",
        note: str = "",
    ):
        """Record a single LLM call.

        Args:
            name: LLM identifier (e.g., "openai", "gemini", "ollama", "fallback").
            success: Whether the call succeeded.
            latency_ms: Call duration in milliseconds.
            error: Error message if failed.
            note: Optional annotation (e.g., "fallback for gemini").
        """
        stats = self._get_stats(name)
        stats.calls += 1
        stats.last_call_time = time.time()

        if success:
            stats.successes += 1
            if latency_ms is not None:
                stats.total_latency_ms += latency_ms
            if note:
                logger.debug(f"[usage] {name}: OK ({note})")
        else:
            stats.failures += 1
            stats.last_error = error
            if error:
                stats.errors.append(f"{time.strftime('%H:%M:%S')}: {error[:100]}")
                # Keep only last 20 errors
                if len(stats.errors) > 20:
                    stats.errors = stats.errors[-20:]
            logger.warning(f"[usage] {name}: FAILED - {error[:80]}")

    def is_exhausted(self, name: str) -> bool:
        """Check if a model appears to be exhausted/quota-limited."""
        stats = self._get_stats(name)
        return stats.is_exhausted()

    def get_stats(self, name: str) -> LLMStats:
        """Get stats for a specific LLM."""
        return self._get_stats(name)

    def report(self) -> str:
        """Generate a usage report string."""
        if not self._stats:
            return "  No LLM calls recorded."

        lines = []
        lines.append(f"  {'LLM':<12} {'Calls':>5} {'OK':>4} {'Fail':>5} {'Avg ms':>8} {'Rate':>6}  Status")
        lines.append(f"  {'-'*12} {'-'*5} {'-'*4} {'-'*5} {'-'*8} {'-'*6}  {'-'*12}")

        for name, stats in sorted(self._stats.items()):
            status = "OK"
            if stats.is_exhausted():
                status = "EXHAUSTED"
            elif stats.failures > 0:
                status = f"DEGRADED ({stats.failures} err)"

            rate = f"{stats.success_rate:.0%}" if stats.calls > 0 else "-"
            avg = f"{stats.avg_latency_ms:.0f}" if stats.successes > 0 else "-"
            lines.append(
                f"  {name:<12} {stats.calls:>5} {stats.successes:>4} {stats.failures:>5} "
                f"{avg:>8} {rate:>6}  {status}"
            )

        # Show last error for failed models
        for name, stats in self._stats.items():
            if stats.last_error:
                lines.append(f"    Last error [{name}]: {stats.last_error[:80]}")

        return "\n".join(lines)

    def reset(self):
        """Clear all stats (start fresh)."""
        self._stats.clear()


# Global singleton
usage_tracker = LLMUsageTracker()
