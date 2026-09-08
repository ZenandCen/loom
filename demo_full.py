"""COMPREHENSIVE DEMO — LangChain + LangGraph + Memory + RAG.

Mỗi section giải thích MỘT concept cốt lõi, kèm code thực tế.
Chạy: python demo_full.py

Sections:
  1. LangChain Core — LLM, Prompts, Messages
  2. LangChain Tools — Function Calling
  3. LangGraph — StateGraph, Nodes, Edges
  4. LangGraph — Conditional Routing + Loops
  5. Memory — Short-term (Checkpointer)
  6. Memory — Long-term (Store)
  7. RAG — Indexing + Retrieval + Generation
  8. RAG Pipeline — Adaptive (full flow)
  9. Full Agent — All together
"""

import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LANGCHAIN CORE
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  SECTION 1: LANGCHAIN CORE — LLM, Prompts, Messages")
print("=" * 70)

"""
┌─────────────────────────────────────────────────────────────────────────┐
│ LANGCHAIN LÀ GÌ?                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LangChain = abstraction layer cho LLMs + surrounding infrastructure.  │
│                                                                         │
│  NÓ BAO GỒM:                                                           │
│                                                                         │
│  1. ChatModels (LLM wrappers)                                         │
│     - Unified API cho mọi LLM provider (OpenAI, Anthropic, Ollama...)  │
│     - Bạn gọi .invoke(messages) → nhận AIMessage                        │
│     - Không cần quan tâm API khác nhau giữa providers                  │
│                                                                         │
│  2. Messages (message types)                                           │
│     - SystemMessage: instructions cho LLM (không show user)            │
│     - HumanMessage: input từ user                                      │
│     - AIMessage: response từ LLM                                       │
│     - ToolMessage: result từ tool (feedback vào LLM)                   │
│                                                                         │
│  3. Prompts (prompt templates)                                         │
│     - Chain-of-thought, few-shot, structured output                    │
│     - PromptTemplate: "Trả lời {question} một cách ngắn gọn"          │
│                                                                         │
│  4. Output Parsers                                                      │
│     - Parse LLM text output → structured data (JSON, Pydantic)        │
│     - Với Qwen: dùng structured output (function calling)             │
│                                                                         │
│  5. Tools (function calling)                                           │
│     - Wrap Python functions cho LLM call                               │
│     - LLM decide WHEN to call + WHAT arguments to pass                 │
│                                                                         │
│  6. Chains (sequential composition)                                    │
│     - LLMChain: prompt → LLM → parser → output                        │
│     - (LangGraph thay thế Chains cho complex flows)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.models import LLM, get_llm

# ─── 1a: Basic LLM Invocation ────────────────────────────────────────────────
# 
# get_llm() returns a ChatModel instance (wraps OpenAI-compatible API)
# .invoke() takes a list of messages → returns AIMessage
#
# Message flow:
#   [SystemMessage, HumanMessage] → LLM → AIMessage
#
llm = get_llm(LLM.OPENAI)

print("\n  1a. Basic LLM call:")
print("  " + "-" * 50)

messages = [
    # SystemMessage: sets the LLM's "persona" / behavior
    # → LLM sees this but user doesn't
    SystemMessage(content="You are a concise technical assistant. Answer in Vietnamese."),

    # HumanMessage: the actual user input
    HumanMessage(content="LangChain là gì? Trong 1 câu."),
]

start = time.time()
response = llm.invoke(messages)
elapsed = (time.time() - start) * 1000

print(f"  Q: LangChain là gì? Trong 1 câu.")
print(f"  A: {response.content}")
print(f"  ({elapsed:.0f}ms, model: {llm.model_name})")

# ─── 1b: Multi-turn Conversation ─────────────────────────────────────────────
#
# LLMs are STATELESS — mỗi lần .invoke() là 1 request mới.
# Để có "nhớ", bạn phải GỬI LẠI toàn bộ history mỗi lần.
#
# messages = [sys, user1, ai1, user2, ai2, user3] → LLM → ai3
#
# Đây là lý do LangGraph cần Checkpointer: tự lưu/load history.
#
print("\n  1b. Multi-turn (stateless LLM + manual history):")
print("  " + "-" * 50)

conversation = [
    SystemMessage(content="Trả lời ngắn gọn, 1 câu."),
]

turns = [
    "Capital of France?",
    "And of Japan?",
    "What language do they speak there?",
]

for turn in turns:
    conversation.append(HumanMessage(content=turn))
    response = llm.invoke(conversation)  # Gửi TOÀN BỘ history mỗi lần
    conversation.append(response)         # Lưu AI response vào history
    print(f"  U: {turn}")
    print(f"  A: {response.content}")

print(f"\n  → History length: {len(conversation)} messages")
print(f"  → Mỗi turn gửi {len(conversation)} messages (không tiết kiệm)")

# ─── 1c: Structured Output ───────────────────────────────────────────────────
#
# Problem: LLM trả về text tự do → parse khó.
# Solution: Pydantic schema → LLM phải trả về đúng structure.
#
# LangChain dùng "function calling" (tool calling) internally:
#   - Define a Pydantic model
#     → LangChain converts to JSON schema
#       → Sends to LLM as a "tool" the LLM must "call"
#         → LLM returns structured JSON matching the schema
#
from pydantic import BaseModel, Field

class ProductReview(BaseModel):
    """Structured output — LLM MUST return this format."""
    product: str = Field(description="Product name")
    rating: int = Field(description="Rating 1-5")
    summary: str = Field(description="One sentence summary")
    pros: list[str] = Field(description="List of pros")
    cons: list[str] = Field(description="List of cons")

print("\n  1c. Structured output (Pydantic schema):")
print("  " + "-" * 50)

# with_structured_output() → LLM returns a validated Pydantic object
structured_llm = llm.with_structured_output(ProductReview)
result = structured_llm.invoke("Review the Apple AirPods Pro 2")

print(f"  Product: {result.product}")
print(f"  Rating:  {result.rating}/5")
print(f"  Summary: {result.summary}")
print(f"  Pros:    {result.pros}")
print(f"  Cons:    {result.cons}")
print(f"\n  → Type: {type(result).__name__} (not string! validated by Pydantic)")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: LANGCHAIN TOOLS (Function Calling)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 2: LANGCHAIN TOOLS — Function Calling")
print("=" * 70)

"""
┌─────────────────────────────────────────────────────────────────────────┐
│ TOOL / FUNCTION CALLING                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Problem: LLM chỉ generate text. Nó không thể:                          │
│    - Search the web                                                      │
│    - Query a database                                                    │
│    - Send an email                                                       │
│                                                                         │
│  Solution: Tools                                                         │
│                                                                         │
│  1. You define Python functions with @tool decorator                    │
│     → LangChain reads the type hints + docstring                        │
│       → Generates a JSON schema describing the function                 │
│                                                                         │
│  2. You bind tools to the LLM:                                          │
│     llm_with_tools = llm.bind_tools([search, calc])                     │
│                                                                         │
│  3. When LLM decides it needs a tool:                                   │
│     - It returns a "tool_call" instead of text                          │
│     - AIMessage.tool_calls = [{"name": "search", "args": {...}}]       │
│                                                                         │
│  4. YOUR CODE executes the tool:                                        │
│     result = search.invoke({"query": "..."})                            │
│                                                                         │
│  5. You send the result back as a ToolMessage:                          │
│     messages.append(ToolMessage(content=result, tool_call_id=...))     │
│                                                                         │
│  6. LLM sees the tool result and generates final answer                 │
│                                                                         │
│  Flow:                                                                   │
│  User → LLM → [tool_call] → Execute → [tool_result] → LLM → Answer    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
"""

from langchain_core.tools import tool

# ─── 2a: Define a Tool ───────────────────────────────────────────────────────
#
# @tool decorator:
#   - Reads function signature (type hints)
#   - Reads docstring (for LLM to understand when to use it)
#   - Creates a BaseTool object with .name, .description, .args_schema
#
# parse_docstring=True: parse the Args section of docstring into descriptions
#
@tool(parse_docstring=True)
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: Math expression like '2 + 2 * 10'
    """
    # Simple eval for demo (NOT for production!)
    result = eval(expression)
    return str(result)

@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: City name
    """
    # Mock data (real would call an API)
    weather_data = {
        "hanoi": "28°C, sunny",
        "ho chi minh city": "32°C, humid",
        "tokyo": "22°C, light rain",
    }
    return weather_data.get(city.lower(), f"Weather in {city}: 25°C, clear")

# ─── 2b: LLM with Tools (Manual loop) ───────────────────────────────────────
#
# WITHOUT a framework like LangGraph/deepagents, you must
# manually implement the tool-calling loop:
#
#   while True:
#       response = llm.invoke(messages)
#       if response.tool_calls:
#           # Execute each tool
#           for tc in response.tool_calls:
#               result = tools[tc["name"]].invoke(tc["args"])
#               messages.append(ToolMessage(content=result, ...))
#       else:
#           # No more tool calls → final answer
#           break
#
# This loop is EXACTLY what LangGraph/deepagents automate for you.
#
from langchain_core.messages import ToolMessage

print("\n  2a. LLM decides to call tools:")
print("  " + "-" * 50)

llm_with_tools = llm.bind_tools([calculator, get_weather])

messages = [
    SystemMessage(content="You have access to tools. Use them when needed."),
    HumanMessage(content="What's 15 * 12 plus the temperature in Hanoi?"),
]

print(f"  Q: What's 15 * 12 plus the temperature in Hanoi?")
print(f"  (LLM will decide to call calculator AND get_weather)")
print()

tool_call_count = 0

# THE TOOL CALLING LOOP (this is what deepagents/LangGraph automates)
for step in range(5):  # Max 5 iterations (safety)
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tc in response.tool_calls:
            tool_call_count += 1
            tool_name = tc["name"]
            tool_args = tc["args"]

            # Find and execute the tool
            target_tool = calculator if tool_name == "calculator" else get_weather
            result = target_tool.invoke(tool_args)

            print(f"  → Tool call #{tool_call_count}: {tool_name}({tool_args})")
            print(f"  ← Result: {result}")

            # Send result back to LLM
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    else:
        # No tool calls → LLM has its final answer
        print(f"\n  Final answer: {response.content}")
        break

print(f"\n  Total tool calls: {tool_call_count}")
print(f"  Total messages in history: {len(messages)}")
print(f"  → LLM 'thought' about when to call tools, what args to pass")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: LANGGRAPH — STATEGRAPH, NODES, EDGES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 3: LANGGRAPH — StateGraph, Nodes, Edges")
print("=" * 70)

"""
┌─────────────────────────────────────────────────────────────────────────┐
│ LANGGRAPH LÀ GÌ?                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LangChain = building blocks (LLM, tools, prompts)                     │
│  LangGraph = ORCHESTRATION — how blocks talk to each other             │
│                                                                         │
│  Core concepts:                                                         │
│                                                                         │
│  1. State (TypedDict)                                                   │
│     - Shared data that flows through the graph                          │
│     - Each node reads from + writes to state                            │
│     - Like a "blackboard" all nodes can see                             │
│                                                                         │
│  2. Node (function)                                                     │
│     - A function: (state) → partial_state_update                        │
│     - Does ONE thing (call LLM, call tool, transform data)             │
│     - Returns dict of fields to UPDATE in state                         │
│                                                                         │
│  3. Edge (connection between nodes)                                     │
│     - Static: A → B (always go to B after A)                           │
│     - Conditional: A → router() → {B or C} (dynamic)                   │
│                                                                         │
│  4. StateGraph                                                          │
│     - The graph itself: nodes + edges + state schema                    │
│     - .compile() → CompiledGraph (executable)                           │
│     - .invoke(initial_state) → run the graph                            │
│                                                                         │
│  WHY not just a while loop?                                             │
│    - Explicit control flow (visualizable, debuggable)                   │
│    - State management (automatic, type-safe)                            │
│    - Persistence (checkpointing, time-travel)                           │
│    - Human-in-the-loop (pause, approve, resume)                         │
│    - Streaming (token-by-token output)                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

# ─── 3a: Simple Linear Graph ─────────────────────────────────────────────────
#
# A graph is: State → [Node1] → [Node2] → [Node3] → END
#
# Each node:
#   - Receives: current state (all fields)
#   - Returns: partial dict (only fields it changes)
#   - LangGraph MERGES the return into state
#
print("\n  3a. Linear graph: transcribe → translate → summarize")
print("  " + "-" * 50)

# STEP 1: Define State (the shared "blackboard")
class TranscriptionState(TypedDict, total=False):
    """State flows through all nodes. Each node updates what it needs."""
    text: str          # Input text
    translated: str    # After translation node
    summary: str       # After summary node
    step_log: list     # Track which nodes ran


# STEP 2: Define Nodes (each does ONE thing)

def transcribe_node(state: TranscriptionState) -> dict:
    """Simulate speech-to-text (just echo for demo)."""
    print(f"    [Node: transcribe] Processing: '{state['text'][:40]}...'")
    return {
        "translated": state["text"],  # "translate" (identity for demo)
        "step_log": state.get("step_log", []) + ["transcribe"],
    }


def translate_node(state: TranscriptionState) -> dict:
    """Translate text (uses LLM)."""
    # In real app: llm.invoke(f"Translate to English: {state['text']}")
    print(f"    [Node: translate] Translating...")
    return {
        "translated": f"[EN] {state['text'][:50]}",
        "step_log": state.get("step_log", []) + ["translate"],
    }


def summarize_node(state: TranscriptionState) -> dict:
    """Summarize the translated text (uses LLM)."""
    print(f"    [Node: summarize] Summarizing...")
    return {
        "summary": f"Summary of: {state['translated'][:30]}...",
        "step_log": state.get("step_log", []) + ["summarize"],
    }


# STEP 3: Build the Graph
#
# StateGraph(StateClass) → add nodes → add edges → compile
#
graph = StateGraph(TranscriptionState)

# Add nodes: (name, function)
graph.add_node("transcribe", transcribe_node)
graph.add_node("translate", translate_node)
graph.add_node("summarize", summarize_node)

# Add edges: (from, to)
# START is a special constant for the entry point
# END is a special constant for the exit
graph.add_edge(START, "transcribe")
graph.add_edge("transcribe", "translate")
graph.add_edge("translate", "summarize")
graph.add_edge("summarize", END)

# Compile → executable graph
app = graph.compile()

# STEP 4: Run it
result = app.invoke({"text": "Hello, this is a demo of LangGraph linear pipeline."})

print(f"\n  Result:")
print(f"    text:     {result['text'][:50]}")
print(f"    translated: {result['translated'][:50]}")
print(f"    summary:  {result['summary'][:50]}")
print(f"    steps:    {result['step_log']}")

# ─── 3b: Conditional Edges + Loop ───────────────────────────────────────────
#
# Real graphs have BRANCHES and LOOPS.
#
# Conditional edge:
#   graph.add_conditional_edges(
#       "node_name",       # from which node
#       routing_function,  # function that returns next node name
#       {                  # mapping: return_value → next_node
#           "option_a": "node_a",
#           "option_b": "node_b",
#       }
#   )
#
print("\n  3b. Conditional graph with loop (grade & retry):")
print("  " + "-" * 50)

class GradeState(TypedDict, total=False):
    question: str
    answer: str
    grade: str
    attempts: int
    final: bool

def generate_answer(state: GradeState) -> dict:
    """Generate an answer (simulated)."""
    attempts = state.get("attempts", 0) + 1
    # Simulate: first attempt is bad, second is good
    if attempts == 1:
        answer = f"Attempt {attempts}: I don't know."
    else:
        answer = f"Attempt {attempts}: The answer is 42."
    print(f"    [generate] {answer}")
    return {"answer": answer, "attempts": attempts}

def grade_answer(state: GradeState) -> dict:
    """Grade the answer. Stores decision in state."""
    if "don't know" in state["answer"]:
        print(f"    [grade] BAD → will retry")
        return {"grade": "retry"}
    else:
        print(f"    [grade] GOOD → will accept")
        return {"grade": "accept"}

def grade_router(state: GradeState) -> str:
    """Router: reads the grade from state, returns next node name."""
    return state.get("grade", "retry")

def final_node(state: GradeState) -> dict:
    """Mark as done."""
    return {"final": True}

# Build graph with conditional edge
g2 = StateGraph(GradeState)
g2.add_node("generate", generate_answer)
g2.add_node("grade", grade_answer)
g2.add_node("final", final_node)

# Entry: always start with generate
g2.add_edge(START, "generate")

# After generate → grade
g2.add_edge("generate", "grade")

# After grade → CONDITIONAL: accept or retry?
# Note: router function is SEPARATE from node function
# Node returns state update, router reads state to decide path
g2.add_conditional_edges(
    "grade",
    grade_router,  # Separate function that reads state["grade"]
    {
        "accept": "final",   # If good → done
        "retry": "generate", # If bad → loop back to generate!
    }
)

g2.add_edge("final", END)

app2 = g2.compile()
result2 = app2.invoke({"question": "What is the answer to life?"})

print(f"\n  Final answer: {result2['answer']}")
print(f"  Attempts:     {result2['attempts']}")
print(f"  → Loop ran {result2['attempts']}x until grade was 'accept'")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MEMORY — Short-term + Long-term
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 4: MEMORY — Checkpointer + Store")
print("=" * 70)

"""
┌─────────────────────────────────────────────────────────────────────────┐
│ MEMORY IN LOOM                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  2 TIERS:                                                               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ SHORT-TERM: Checkpointer (MemorySaver)                      │       │
│  │                                                             │       │
│  │ - Persists graph state between .invoke() calls             │       │
│  │ - Keyed by thread_id (one per conversation)                 │       │
│  │ - Auto-saves after each node execution                      │       │
│  │ - Lets you resume a graph where it left off                 │       │
│  │                                                             │       │
│  │ Use case: multi-turn chat, long-running workflows           │       │
│  │ Storage: in-memory (dev) / Postgres (prod)                 │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ LONG-TERM: Store (InMemoryStore)                            │       │
│  │                                                             │       │
│  │ - Key-value store with semantic search                      │       │
│  │ - Namespaced: (team, user, category)                        │       │
│  │ - Agent explicitly writes facts via tools                   │       │
│  │ - Agent explicitly reads via search                         │       │
│  │                                                             │       │
│  │ Use case: user preferences, learned facts, rules            │       │
│  │ Storage: in-memory (dev) / Postgres (prod)                 │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                         │
│  DIFFERENCE:                                                            │
│  - Checkpointer: AUTOMATIC (graph state saved after each step)         │
│  - Store: EXPLICIT (agent decides what to remember via tool call)     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# ─── 4a: Checkpointer (Short-term / Conversation Memory) ────────────────────
#
# Without checkpointer: each .invoke() is independent (no memory)
# With checkpointer: pass thread_id → state persists across calls
#
# The checkpointer saves state AFTER EACH NODE.
# Next .invoke() with same thread_id → loads last saved state → continues.
#
print("\n  4a. Checkpointer — conversation memory across calls:")
print("  " + "-" * 50)

checkpointer = MemorySaver()

# Simple chat graph: just echo/remember what user said
class ChatState(TypedDict, total=False):
    messages: list
    user_name: str

def chat_node(state: ChatState) -> dict:
    """Simple node that 'remembers' the user's name."""
    messages = state.get("messages", [])
    last_msg = messages[-1]["content"] if messages else ""

    if "my name is" in last_msg.lower():
        name = last_msg.split("my name is")[-1].strip()
        print(f"    [chat] Remembered name: {name}")
        return {"user_name": name, "messages": messages + [{"role": "ai", "content": f"Nice to meet you, {name}!"}]}
    elif state.get("user_name"):
        print(f"    [chat] Remembering you're {state['user_name']}")
        return {"messages": messages + [{"role": "ai", "content": f"Hello again, {state['user_name']}!"}]}
    else:
        return {"messages": messages + [{"role": "ai", "content": "What's your name?"}]}

chat_graph = StateGraph(ChatState)
chat_graph.add_node("chat", chat_node)
chat_graph.add_edge(START, "chat")
chat_graph.add_edge("chat", END)
chat_app = chat_graph.compile(checkpointer=checkpointer)  # ← KEY: pass checkpointer

# Turn 1: Tell your name
config1 = {"configurable": {"thread_id": "user_zenchung"}}
r1 = chat_app.invoke({"messages": [{"role": "user", "content": "Hi, my name is Zenchung"}]}, config1)
print(f"  Turn 1: 'Hi, my name is Zenchung'")
print(f"  AI: {r1['messages'][-1]['content']}")

# Turn 2: Ask a question (state REMEMBERS the name from turn 1!)
time.sleep(0.1)  # Simulate time passing
r2 = chat_app.invoke({"messages": [{"role": "user", "content": "How are you?"}]}, config1)
print(f"  Turn 2: 'How are you?'")
print(f"  AI: {r2['messages'][-1]['content']}")
print(f"  → It remembered 'Zenchung' from turn 1 (via checkpointer)")

# Different thread = different conversation (no memory sharing)
r3 = chat_app.invoke({"messages": [{"role": "user", "content": "How are you?"}]}, {"configurable": {"thread_id": "user_other"}})
print(f"  Turn 1 (different user): 'How are you?'")
print(f"  AI: {r3['messages'][-1]['content']}")
print(f"  → Different thread → no memory of 'Zenchung'")

# ─── 4b: Store (Long-term / Semantic Memory) ────────────────────────────────
#
# InMemoryStore: key-value with vector search
#
# .put(namespace, key, value): save a fact
# .search(namespace, query): semantic search (finds relevant facts)
#
# Namespace is a tuple: ("loom", "user_zenchung", "preferences")
# This enables per-user, per-category organization.
#
print("\n  4b. Store — long-term semantic memory:")
print("  " + "-" * 50)

store = InMemoryStore()

# Agent "remembers" facts (like the remember() tool in loom)
facts = [
    ("preference", "prefers_dark_mode", "User prefers dark mode in all editors"),
    ("preference", "uses_macos", "User is on macOS with Homebrew"),
    ("context", "current_project", "Working on loom - LangGraph agent framework"),
    ("rule", "language", "User communicates in Vietnamese"),
]

for category, key, value in facts:
    store.put(("loom", "zenchung", category), key, {"fact": value})

print(f"  Stored {len(facts)} facts")

# Semantic search: "find facts about my OS"
print(f"\n  Search: 'what OS does the user use?'")
results = store.search(("loom", "zenchung"), query="what OS does the user use?")
for item in results:
    score = f" (score: {item.score:.3f})" if item.score is not None else ""
    print(f"    → {item.value['fact']}{score}")

print(f"\n  Search: 'what language does user prefer?'")
results = store.search(("loom", "zenchung"), query="what language does user prefer?")
for item in results:
    score = f" (score: {item.score:.3f})" if item.score is not None else ""
    print(f"    → {item.value['fact']}{score}")

print(f"\n  → Store uses EMBEDDINGS for semantic search (not exact match)")
print(f"  → 'what OS' matches 'uses_macos' without keyword overlap")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: RAG — Indexing + Retrieval + Generation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 5: RAG — Index → Retrieve → Generate")
print("=" * 70)

"""
┌─────────────────────────────────────────────────────────────────────────┐
│ RAG (Retrieval-Augmented Generation)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Problem: LLM has knowledge cutoff + doesn't know your private docs.   │
│                                                                         │
│  Solution: RAG = 3 phases:                                             │
│                                                                         │
│  ┌──────────┐     ┌──────────────┐     ┌──────────────┐              │
│  │ INDEX    │     │ RETRIEVE     │     │ GENERATE     │              │
│  │ (offline)│     │ (online)     │     │ (online)     │              │
│  └──────────┘     └──────────────┘     └──────────────┘              │
│                                                                         │
│  INDEX:                                                                  │
│    1. Load documents (PDF, MD, CSV...)                                 │
│    2. Chunk (split into ~512 token pieces)                            │
│    3. Embed (text → 768-dim vector via Ollama)                        │
│    4. Store (vector + metadata → Chroma/pgvector)                     │
│                                                                         │
│  RETRIEVE:                                                               │
│    1. Embed the query (same model as indexing)                         │
│    2. Vector search (cosine similarity in vector DB)                   │
│    3. Rerank (cross-encoder or keyword overlap)                        │
│    4. Return top-k most relevant chunks                                │
│                                                                         │
│  GENERATE:                                                               │
│    1. Build prompt: "Answer using ONLY this context: [docs]"           │
│    2. LLM generates answer grounded in retrieved context               │
│    3. Cite sources                                                     │
│                                                                         │
│  KEY INSIGHT:                                                            │
│  - Same embedding model for INDEX and QUERY (must match!)              │
│  - Query "what is X?" → vector → find nearest document vectors         │
│  - Cosine similarity: 1.0 = identical direction, 0.0 = orthogonal      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
"""

from rag.indexing import index_documents, load_documents
from rag.config import get_rag_settings, get_embeddings
from rag.retrieval import get_retriever, rerank_documents

settings = get_rag_settings()

# ─── 5a: INDEXING ───────────────────────────────────────────────────────────
print("\n  5a. Indexing (load → chunk → embed → store):")
print("  " + "-" * 50)

# Load documents from data/
docs = load_documents(Path(settings.data_dir))
print(f"  Loaded {len(docs)} document units from {settings.data_dir}")
for d in docs[:3]:
    src = d.metadata.get("source", "?")
    page = d.metadata.get("page", "")
    print(f"    - {Path(src).name} {f'p{page}' if page else ''} ({len(d.page_content)} chars)")

# Index them
count = index_documents()
print(f"  Indexed {count} chunks into vector store")

# ─── 5b: RETRIEVAL ──────────────────────────────────────────────────────────
print(f"\n  5b. Retrieval (vector search + rerank):")
print("  " + "-" * 50)

# Get a retriever (wraps vector store similarity search)
retriever = get_retriever(k=3)

# Query
query = "LangGraph"
print(f"  Query: '{query}'")

# Step 1: Embed the query
emb = get_embeddings()
query_vector = emb.embed_query(query)
print(f"  Query embedded: {len(query_vector)} dimensions")
print(f"  First 5 dims:   {query_vector[:5]}")

# Step 2: Vector search (cosine similarity)
docs = retriever.invoke(query)
print(f"\n  Retrieved {len(docs)} documents:")
for i, doc in enumerate(docs):
    print(f"doc data {i+1}: {doc}")
    src = doc.metadata.get("source", "?")
    page = doc.metadata.get("page", "")
    print(f"    [{i+1}] {Path(src).name} {f'p{page}' if page else ''}: {doc.page_content[:80]}...")

# ─── 5c: GENERATION ─────────────────────────────────────────────────────────
print(f"\n  5c. Generation (LLM + context):")
print("  " + "-" * 50)

context = "\n\n".join(f"[{i+1}] {d.page_content[:500]}" for i, d in enumerate(docs))
prompt = (
    f"Answer using ONLY the context below. If not found, say 'I don't know.'\n\n"
    f"Context:\n{context}\n\n"
    f"Question: What is this project about?\n\n"
    f"Answer:"
)
response = llm.invoke(prompt)
print(f"  Q: What is this project about?")
print(f"  A: {response.content[:200]}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: FULL AGENT — Everything together
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 6: FULL AGENT — All systems working together")
print("=" * 70)

"""
┌─────────────────────────────────────────────────────────────────────────┐
│ HOW EVERYTHING FITS TOGETHER IN LOOM                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  User Input                                                             │
│     │                                                                    │
│     ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ MIDDLEWARE (wraps the agent)                                 │       │
│  │  - Safety: budget guard, dangerous tool gate                │       │
│  │  - Observability: log sessions, model calls, tool calls     │       │
│  │  - Domain: inject user context, sanitize output             │       │
│  └─────────────────────────────────────────────────────────────┘       │
│     │                                                                    │
│     ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ DEEP AGENT (LangGraph StateGraph under the hood)            │       │
│  │                                                             │       │
│  │  LLM (Qwen) ←→ Tools (tavily, rag_query, remember, recall) │       │
│  │       │          ←→ Subagents (researcher, report-pipeline) │       │
│  │       │          ←→ Filesystem (workspace/)                 │       │
│  │       │          ←→ Store (memories)                        │       │
│  │       │                                                       │       │
│  │  rag_query tool:                                              │       │
│  │     → run_rag() → StateGraph (adaptive pipeline)            │       │
│  │       → route → retrieve → grade → generate                 │       │
│  │       → (hallucination check in self_rag)                    │       │
│  │                                                             │       │
│  │  Checkpointer: saves agent state per thread_id              │       │
│  │  Store: long-term user facts                                │       │
│  └─────────────────────────────────────────────────────────────┘       │
│     │                                                                    │
│     ▼                                                                    │
│  Response → User                                                         │
│                                                                         │
│  The agent DECIDES which tool to use based on the question:            │
│    - "Search the web for X" → tavily_search                            │
│    - "What does our doc say about X?" → rag_query                       │
│    - "Remember that I prefer X" → remember                             │
│    - "What did I tell you about X?" → recall                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
"""

print("\n  6a. Agent architecture summary:")
print("  " + "-" * 50)

from agent.tools import all_tools
from agent.subagents import all_subagents

print(f"  Tools available to agent:")
for t in all_tools:
    print(f"    - {t.name}: {t.description[:60]}")

print(f"\n  Subagents:")
for sa in all_subagents:
    print(f"    - {sa['name']}: {sa['description'][:60]}")

print(f"\n  Memory:")
print(f"    - Checkpointer: MemorySaver (per-thread conversation)")
print(f"    - Store: InMemoryStore (per-user facts)")

print(f"\n  RAG:")
print(f"    - Pipeline: {settings.default_pipeline.value}")
print(f"    - Vector DB: {settings.vector_db_type.value}")
print(f"    - Embedding: {settings.embedding_model} ({settings.embedding_provider.value})")
print(f"    - Chunking: {settings.chunking_strategy.value} ({settings.chunk_size} tokens)")

print(f"\n  6b. Quick RAG query via tool:")
print("  " + "-" * 50)

from rag.pipeline import run_rag

result = run_rag("What is this project about?", level=None)
print(f"  Q: What is this project about?")
print(f"  A: {result['generation'][:200]}")
if result['sources']:
    print(f"  Sources: {result['sources'][:2]}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SUMMARY — What you now understand")
print("=" * 70)
print("""
  LAYER          | WHAT IT DOES                          | KEY CONCEPT
  ───────────────┼───────────────────────────────────────┼─────────────────────
  LangChain      | Abstract LLMs, tools, prompts         | .invoke(), @tool
  LangGraph      | Orchestrate multi-step flows          | StateGraph, nodes
  Memory         | Remember across calls                 | Checkpointer + Store
  RAG            | Ground answers in documents           | Embed + Vector Search
  Middleware     | Cross-cutting concerns                | Safety, Logging
  Agent (Deep)   | Autonomously decide + act             | Tool calling loop

  DATA FLOW:
  User → Middleware → Agent(LLM) → Tool(RAG) → VectorDB → LLM → Answer
                              ↕                    ↕
                        Memory (Store)     Embeddings (Ollama)

  KEY INSIGHT:
  - LangChain = BRICKS (LLM, tools, prompts)
  - LangGraph = BLUEPRINT (how bricks connect)
  - RAG = GROUNDING (answers must come from docs)
  - Memory = IDENTITY (knows who you are)
  - Agent = DECISION (chooses which tool when)
""")

print("  DEMO COMPLETE")
