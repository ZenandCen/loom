# loom

> Dệt các agent thành một hệ thống hoàn chỉnh.

LangGraph 2026 project — Deep Agent với multi-agent orchestration, hybrid storage, layered middleware.

## Architecture

```
loom/
├── main.py                       # Entry point
├── agent/
│   ├── config.py                 # Model, backend, store
│   ├── prompt.py                 # System prompts
│   ├── tools.py                  # @tool definitions
│   ├── middleware/
│   │   ├── safety.py             # Budget, rate limit
│   │   ├── observability.py      # Logging, tracing
│   │   └── domain.py             # Business logic
│   ├── subagents/
│   │   ├── researcher.py         # Research subagent
│   │   └── workflow_pipeline.py  # Deterministic pipeline
│   └── build.py                  # COMPOSE: assemble everything
├── memory/
│   ├── schemas.py                # Typed memory schemas
│   ├── load.py                   # Load memory node
│   └── create.py                 # Create memory node
├── graphs/
│   ├── router.py                 # Triage / routing
│   └── main_graph.py             # Optional StateGraph wrapper
└── evals/
    ├── dataset.py
    ├── evaluators.py
    └── run_evals.py
```

## Quick Start

```bash
cp .env.example .env  # fill in API keys
pip install -e .
python main.py
```

## Key Decisions

| Concern | Choice | Why |
|---------|--------|-----|
| Agent | `create_deep_agent()` | Filesystem, planning, subagents built-in |
| Storage | `CompositeBackend` | Hybrid: state + disk + persistent store |
| Middleware | 5-layer stack | Safety → Observability → Domain → HITL |
| Multi-agent | Supervisor (subagents param) | Native delegation, no boilerplate |
| Memory | 2-tier (checkpointer + store) | Short-term per-thread, long-term per-user |
