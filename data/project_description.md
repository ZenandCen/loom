# Loom - LangGraph Deep Agent Framework

Loom is a production-ready LangGraph agent framework that combines LLM orchestration, RAG (Retrieval-Augmented Generation), and persistent memory into a unified system.

## Core Capabilities

1. **Deep Agent**: Autonomous agent that decides which tools to use (web search, RAG, memory, file operations) based on the user's question.

2. **RAG Pipeline**: Multi-level adaptive RAG with 4 pipeline levels:
   - Level 1: Basic retrieval + generation
   - Level 2: + Query rewriting (adaptives_rag)
   - Level 3: + Document grading (corrective_rag)
   - Level 4: + Hallucination check (self_rag)

3. **Memory System**: Two-tier memory:
   - Short-term: LangGraph Checkpointer (conversation state per thread)
   - Long-term: InMemoryStore (semantic search over user facts)

4. **Multi-Agent**: Supervisor pattern with specialized subagents (researcher, report-pipeline).

5. **Middleware**: Safety guards, observability logging, domain-specific context injection.

## Architecture

- **LangChain**: LLM abstraction, tool definitions, prompt management
- **LangGraph**: StateGraph orchestration, conditional routing, loops, parallel execution
- **Vector DB**: pgvector / Chroma / Qdrant for semantic search
- **Embeddings**: Ollama (nomic-embed-text) or OpenAI
- **LLM**: Qwen via OpenAI-compatible API

## Key Design Patterns

- ReAct Agent (tool calling loop)
- Self-RAG (hallucination detection)
- Adaptive RAG (query rewriting)
- Corrective RAG (document grading)
- Plan-and-Execute
- Human-in-the-Loop (interrupt + checkpoint)
