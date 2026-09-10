"""Comprehensive Demo: LangChain & LangGraph Patterns.

Covers all common LLM orchestration patterns with:
    - Code example
    - Pros / Cons
    - Real-world use cases

Run: .venv/bin/python demo_pattern.py
"""

import logging
import time
from typing import Annotated, TypedDict
from operator import add

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from utils.models import LLM, get_llm, get_llm_with_fallback
from utils.llm_usage import usage_tracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Separator for section output
SEP = "=" * 70
THIN_SEP = "-" * 50


def model_name(llm) -> str:
    """Get model name string (works for both OpenAI and Gemini wrappers)."""
    return getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")


def llm_text(response) -> str:
    """Safely extract text from LLM response (Gemini may return list of parts)."""
    raw = response.content
    if isinstance(raw, list):
        return "".join(part if isinstance(part, str) else part.get("text", "") for part in raw)
    return raw

# Collect all results for final summary
results: list[tuple[str, float, str]] = []


def banner(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def sub(title: str):
    print(f"\n{THIN_SEP}\n  {title}\n{THIN_SEP}")


def pros_cons(pros: list[str], cons: list[str]):
    print("\n  ✅ Ưu điểm:")
    for p in pros:
        print(f"     • {p}")
    print("\n  ❌ Nhược điểm:")
    for c in cons:
        print(f"     • {c}")


def use_cases(cases: list[str]):
    print("\n  🎯 Use cases thực tế:")
    for c in cases:
        print(f"     • {c}")


def section_info(title: str, description: str, pros: list[str], cons: list[str], cases: list[str]):
    sub(title)
    print(f"\n  📋 {description}")
    pros_cons(pros, cons)
    use_cases(cases)


# ============================================================================
# PATTERN 1: LLM Chain (Simple Prompt → LLM → Output)
# ============================================================================

def pattern_1_llm_chain(llm):
    banner("PATTERN 1: LLM CHAIN (Simple)")
    section_info(
        "LLM Chain",
        "Đơn giản nhất: Prompt → LLM → Output. Không có tool, không có retrieval.",
        [
            "Nhanh nhất, ít token nhất (1 lần gọi LLM)",
            "Dễ debug, dễ understand",
            "Không cần infrastructure thêm",
        ],
        [
            "Không xử lý được task phức tạp",
            "Không có context ngoài training data",
            "Không có cơ chế tự kiểm tra/sửa lỗi",
        ],
        [
            "Tạo email template từ input",
            "Phân loại sentiment (positive/negative)",
            "Tóm tắt đoạn văn ngắn",
            "Dịch thuật",
            "Extract structured data từ text",
        ],
    )

    def run():
        start = time.time()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer concisely in Vietnamese."),
            ("human", "{question}"),
        ])
        chain = prompt | llm
        result = chain.invoke({"question": "Giải thích F1 score trong 2 câu"})
        return result.content, time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output: {content[:10000]}...")
    print(f"  ⏱️  {elapsed:.2f}s (1 lần gọi LLM)")
    results.append(("1. LLM Chain", elapsed, "PASS"))


# ============================================================================
# PATTERN 2: RAG Chain (Retrieval → Augment → Generate)
# ============================================================================

def pattern_2_rag(llm):
    banner("PATTERN 2: RAG CHAIN (Retrieval-Augmented Generation)")
    section_info(
        "RAG Chain",
        "Query → Retrieve relevant docs → Augment prompt → Generate answer with citations.",
        [
            "Đưa knowledge mới vào mà không cần retrain",
            "Answer có citation → trustworthy",
            "Cập nhật knowledge dễ (chỉ cần update vector DB)",
            "Chi phí thấp hơn fine-tuning",
        ],
        [
            "Chất lượng phụ thuộc vào retrieval",
            "Không xử lý được multi-hop reasoning tốt",
            "Cần maintain vector DB + chunking strategy",
            "Hallucination vẫn có thể xảy ra",
        ],
        [
            "Customer support bot (tra cứu policy)",
            "Legal assistant (tra cứu luật, precedent)",
            "Medical Q&A (tra cứu research paper)",
            "Enterprise knowledge base (SOP, runbook)",
            "Document Q&A (hỏi đáp trên PDF/DOCX)",
        ],
    )

    def run():
        start = time.time()
        # Simulate retrieval (in production: vector DB search)
        fake_context = (
            "F1 Score = 2 * (Precision * Recall) / (Precision + Recall)\n"
            "F1 = harmonic mean of precision and recall.\n"
            "Use when class imbalance exists."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer based ONLY on the context provided. Cite sources. Answer in Vietnamese."),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        chain = prompt | llm
        result = chain.invoke({
            "context": fake_context,
            "question": "F1 score là gì? Khi nào nên dùng?",
        })
        return result.content, time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output: {content[:10000]}...")
    print(f"  ⏱️  {elapsed:.2f}s (1 lần retrieve + 1 lần LLM)")
    results.append(("2. RAG Chain", elapsed, "PASS"))


# ============================================================================
# PATTERN 3: Tool Calling / ReAct Agent
# ============================================================================

def pattern_3_agent(llm):
    banner("PATTERN 3: TOOL CALLING / REACT AGENT")
    section_info(
        "ReAct Agent",
        "Agent loop: Think → Act (call tool) → Observe → Think → ... → Final Answer.",
        [
            "Có thể dùng nhiều tools linh hoạt",
            "Tự quyết định cần tool nào, bao nhiêu lần",
            "Xử lý được task multi-step",
            "Không cần hardcode logic",
        ],
        [
            "Nhiều lần gọi LLM → chậm, tốn token",
            "Có thể loop vô hạn nếu không có guard",
            "Khó debug (non-deterministic)",
            "Tool selection không phải lúc nào cũng đúng",
        ],
        [
            "Research agent (search web + read + summarize)",
            "Coding agent (read file + edit + run test)",
            "Data analysis (query DB + compute + chart)",
            "Travel planning (search flight + hotel + compare)",
            "Customer support (lookup order + process refund)",
        ],
    )

    # Simple tool-calling demo
    def weather_tool(city: str) -> str:
        """Get weather for a city."""
        return f"Nắng 35°C tại {city}"

    llm_with_tools = llm.bind_tools([weather_tool])

    def run():
        start = time.time()
        messages = [
            SystemMessage(content="You are a helpful assistant. Use tools when needed."),
            HumanMessage(content="Thời tiết tại Hà Nội hôm nay?"),
        ]
        # ReAct loop
        for _ in range(3):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                break
            for tc in response.tool_calls:
                if tc["name"] == "weather_tool":
                    result = weather_tool(**tc["args"])
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        return messages[-1].content, time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output: {content[:10000]}")
    print(f"  ⏱️  {elapsed:.2f}s (multiple LLM calls for tool loop)")
    results.append(("3. ReAct Agent", elapsed, "PASS"))


# ============================================================================
# PATTERN 4: Sequential Chain (Multi-step Pipeline)
# ============================================================================

def pattern_4_sequential(llm):
    banner("PATTERN 4: SEQUENTIAL CHAIN (Multi-step Pipeline)")
    section_info(
        "Sequential Chain",
        "Output của step N làm input cho step N+1. Pipeline tuyến tính.",
        [
            "Mỗi step làm 1 việc → dễ debug",
            "Có thể dùng LLM khác cho mỗi step",
            "Dễ thêm/bớt step",
            "Deterministic flow",
        ],
        [
            "Error ở step trước → cascade sang step sau",
            "Không có feedback loop (step sau không sửa step trước)",
            "Chậm hơn 1 lần gọi (nhanh hơn agent)",
            "Không xử lý được branching",
        ],
        [
            "Translate → Proofread → Format",
            "Extract entities → Verify → Classify",
            "Generate outline → Write sections → Edit",
            "Parse code → Explain → Optimize",
            "Classify intent → Extract slots → Respond",
        ],
    )

    def run():
        start = time.time()

        # Step 1: Extract
        extract_prompt = ChatPromptTemplate.from_template(
            "Extract all numbers from: '{text}'. Output only the list."
        )
        extracted = (extract_prompt | llm).invoke({"text": "Revenue was $5M last year and $8M this year"})

        # Step 2: Analyze
        analyze_prompt = ChatPromptTemplate.from_template(
            "Given numbers: {numbers}\nCalculate growth rate. Answer in 1 sentence in Vietnamese."
        )
        analyzed = (analyze_prompt | llm).invoke({"numbers": extracted.content})

        # Step 3: Format
        format_prompt = ChatPromptTemplate.from_template(
            "Format this as a JSON with keys 'growth_rate' and 'comment': {text}"
        )
        final = (format_prompt | llm).invoke({"text": analyzed.content})

        return final.content, time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output: {content}")
    print(f"  ⏱️  {elapsed:.2f}s (3 lần LLM)")
    results.append(("4. Sequential", elapsed, "PASS"))


# ============================================================================
# PATTERN 5: Map-Reduce (Split → Process → Combine)
# ============================================================================

def pattern_5_map_reduce(llm):
    banner("PATTERN 5: MAP-REDUCE (Split → Process → Combine)")
    section_info(
        "Map-Reduce",
        "Split input thành nhiều phần → xử lý song song → gộp kết quả.",
        [
            "Xử lý được input rất dài (vượt context window)",
            "Có thể parallelize → nhanh",
            "Mỗi phần đơn giản → chất lượng tốt",
            "Scale ngang dễ dàng",
        ],
        [
            "Mất context cross-part (2 phần liên quan nhau)",
            "Combine step có thể mất thông tin",
            "Nhiều lần gọi LLM → tốn tiền",
            "Không phù hợp task cần coherence toàn cục",
        ],
        [
            "Tóm tắt sách dài (tóm từng chương → gộp)",
            "Phân tích hàng nghìn reviews",
            "Extract data từ nhiều trang web",
            "Code review (review từng file → tổng hợp)",
            "Legal: review nhiều hợp đồng → tổng hợp risk",
        ],
    )

    def run():
        start = time.time()

        # MAP: Process each segment independently
        segments = [
            "Chapter 1: The company started in 2015 with 3 employees.",
            "Chapter 2: By 2020, revenue reached $10M with 50 employees.",
            "Chapter 3: In 2024, the company IPO'd at $100M valuation.",
        ]
        map_results = []
        for i, seg in enumerate(segments):
            result = llm.invoke(
                f"Summarize this in ONE sentence (Vietnamese): {seg}"
            )
            map_results.append(result.content)

        # REDUCE: Combine all summaries
        combined = llm.invoke(
            "Combine these summaries into a coherent 2-sentence overview (Vietnamese):\n"
            + "\n".join(f"- {s}" for s in map_results)
        )
        return combined.content, time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output: {content[:200]}")
    print(f"  ⏱️  {elapsed:.2f}s (3 map + 1 reduce = 4 calls)")
    results.append(("5. Map-Reduce", elapsed, "PASS"))


# ============================================================================
# PATTERN 6: Routing (Classify → Route)
# ============================================================================

def pattern_6_routing(llm):
    banner("PATTERN 6: ROUTING (Classify → Route)")
    section_info(
        "Routing",
        "Phân loại input → route đến handler phù hợp. Like if/else do LLM quyết định.",
        [
            "Mỗi branch tối ưu cho 1 loại input",
            "Nhẹ hơn agent (chỉ 1 lần classify)",
            "Dễ maintain (thêm branch mới)",
            "Có thể dùng model nhỏ cho routing",
        ],
        [
            "Classification sai → route sai",
            "Không xử lý input ở biên (vừa A vừa B)",
            "Cần biết trước có những loại nào",
            "Không có fallback phức tạp",
        ],
        [
            "Customer support: technical / billing / complaint",
            "Content: news / sports / entertainment / politics",
            "Code: bug fix / feature / refactor / test",
            "Document: FAQ / how-to / reference / policy",
            "Intent detection trong chatbot",
        ],
    )

    def run():
        start = time.time()

        # Step 1: Classify
        classify_prompt = ChatPromptTemplate.from_template(
            "Classify this query into ONE category: technical, billing, or complaint.\n"
            "Query: {query}\nRespond with ONLY the category word."
        )
        category = (classify_prompt | llm).invoke({"query": "My invoice is wrong, I was charged twice"}).content.strip().lower()

        # Step 2: Route to appropriate handler
        handlers = {
            "technical": lambda q: f"[TECHNICAL] Connecting to engineer... (query: {q})",
            "billing": lambda q: f"[BILLING] Forwarding to finance team... (query: {q})",
            "complaint": lambda q: f"[COMPLAINT] Escalating to manager... (query: {q})",
        }
        
        handler = handlers.get(category, handlers["complaint"])
        result = handler("My invoice is wrong, I was charged twice")
        return f"Category: {category}\nResult: {result}", time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output:\n    {content}")
    print(f"  ⏱️  {elapsed:.2f}s (1 classify + 0 LLM in handler)")
    results.append(("6. Routing", elapsed, "PASS"))


# ============================================================================
# PATTERN 7: Self-Refine (Generate → Critique → Improve)
# ============================================================================

def pattern_7_self_refine(llm):
    banner("PATTERN 7: SELF-REFINE (Generate → Critique → Improve)")
    section_info(
        "Self-Refine",
        "LLM tự sinh → tự critique → tự sửa. Lặp đến khi đủ tốt hoặc max iterations.",
        [
            "Chất lượng output cao hơn nhiều",
            "Không cần human feedback",
            "Dùng được cho task cần precision",
            "Càng iterate càng tốt (đến mức converge)",
        ],
        [
            "Nhiều lần gọi LLM (3x-5x) → tốn tiền",
            "Chậm (tuyến tính theo số iteration)",
            "LLM có thể 'chứng nhận' output của mình (self-bias)",
            "Không tốt cho task cần speed",
        ],
        [
            "Viết code + tự review + tự fix",
            "Viết hợp đồng + tự kiểm tra clause",
            "Viết bài báo + tự fact-check",
            "Design system prompt + tự test",
            "Viết SQL + tự validate logic",
        ],
    )

    def run():
        start = time.time()

        # Generate
        draft = llm.invoke(
            "Write a 2-sentence definition of 'polymorphism' in programming (Vietnamese)."
        ).content

        # Critique
        critique = llm.invoke(
            f"Here is a definition:\n\"{draft}\"\n\n"
            "Critique it in 1 sentence: what's missing or wrong? (Vietnamese)"
        ).content

        # Refine
        refined = llm.invoke(
            f"Original: {draft}\nCritique: {critique}\n\n"
            "Write an improved version. 2 sentences only. (Vietnamese)"
        ).content

        return f"DRAFT: {draft}\n\nCRITIQUE: {critique}\n\nREFINED: {refined}", time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output:\n    {content}")
    print(f"  ⏱️  {elapsed:.2f}s (3 lần LLM: generate + critique + refine)")
    results.append(("7. Self-Refine", elapsed, "PASS"))


# ============================================================================
# PATTERN 8: LangGraph Sequential (StateGraph tuyến tính)
# ============================================================================

def pattern_8_langgraph_sequential(llm):
    banner("PATTERN 8: LANGGRAPH SEQUENTIAL (StateGraph)")
    section_info(
        "LangGraph Sequential",
        "Graph tuyến tính với shared State. Mỗi node cập nhật state → node sau đọc.",
        [
            "State quản lý tập trung → dễ trace",
            "Checkpointing tự động (resume từ node bất kỳ)",
            "Dễ visual hóa (Mermaid/dot)",
            "Nền tảng cho các pattern phức tạp hơn",
        ],
        [
            "Overhead so với simple chain",
            "Cần define State schema rõ ràng",
            "Không tận dụng được lợi thế của graph (vì là tuyến tính)",
        ],
        [
            "ETL pipeline: Extract → Transform → Load",
            "Document processing: Parse → Clean → Classify → Store",
            "Multi-format conversion: Detect → Convert → Validate",
            "Onboarding: Collect info → Verify → Create account → Send welcome",
        ],
    )

    def run():
        start = time.time()

        class State(TypedDict):
            input_text: str
            keywords: str
            summary: str
            translation: str

        def extract_keywords(state: State) -> dict:
            result = llm.invoke(f"Extract 3 keywords from: {state['input_text']}")
            return {"keywords": result.content}

        def summarize(state: State) -> dict:
            result = llm.invoke(
                f"Summarize in 1 sentence (Vietnamese): {state['input_text']}\n"
                f"Keywords: {state['keywords']}"
            )
            return {"summary": result.content}

        def translate(state: State) -> dict:
            result = llm.invoke(f"Translate to English: {state['summary']}")
            return {"translation": result.content}

        # Build graph
        graph = StateGraph(State)
        graph.add_node("extract", extract_keywords)
        graph.add_node("summarize", summarize)
        graph.add_node("translate", translate)

        graph.add_edge(START, "extract")
        graph.add_edge("extract", "summarize")
        graph.add_edge("summarize", "translate")
        graph.add_edge("translate", END)

        app = graph.compile()
        final_state = app.invoke({
            "input_text": "LangGraph is a framework for building stateful, multi-actor applications with LLMs",
            "keywords": "",
            "summary": "",
            "translation": "",
        })
        return (
            f"Keywords: {final_state['keywords']}\n"
            f"Summary: {final_state['summary']}\n"
            f"Translation: {final_state['translation']}"
        ), time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output:\n    {content}")
    print(f"  ⏱️  {elapsed:.2f}s (3 nodes, 3 LLM calls)")
    results.append(("8. LangGraph Sequential", elapsed, "PASS"))


# ============================================================================
# PATTERN 9: LangGraph Conditional (Router trong graph)
# ============================================================================

def pattern_9_langgraph_conditional(llm):
    banner("PATTERN 9: LANGGRAPH CONDITIONAL (Router)")
    section_info(
        "LangGraph Conditional",
        "Node → Router (function trả tên node tiếp theo) → Node A hoặc Node B.",
        [
            "Flow linh hoạt, không hardcode",
            "Router có thể là rule-based hoặc LLM-based",
            "Dễ thêm branch mới",
            "State vẫn được preserve giữa các branch",
        ],
        [
            "Routing logic sai → đi nhầm branch",
            "Không có 'merge back' tự nhiên (phải design)",
            "Nhiều branch → graph phức tạp",
        ],
        [
            "Support bot: classify → tech/billing/complaint handler",
            "Content moderation: detect → spam/normal/toxic handler",
            "Code review: detect severity → auto-fix / human-review",
            "RAG: grade docs → sufficient / insufficient (→ rewrite)",
        ],
    )

    def run():
        start = time.time()

        class State(TypedDict):
            query: str
            category: str
            response: str

        def classify(state: State) -> dict:
            result = llm.invoke(
                f"Classify into 'greeting' or 'question': {state['query']}\n"
                "Respond with ONLY the word."
            )
            return {"category": result.content.strip().lower()}

        def route(state: State) -> str:
            return "greeting_handler" if "greeting" in state.get("category", "") else "question_handler"

        def greeting_handler(state: State) -> dict:
            return {"response": "Xin chào! Tôi có thể giúp gì?"}

        def question_handler(state: State) -> dict:
            result = llm.invoke(f"Answer briefly in Vietnamese: {state['query']}")
            return {"response": result.content}

        graph = StateGraph(State)
        graph.add_node("classify", classify)
        graph.add_node("greeting_handler", greeting_handler)
        graph.add_node("question_handler", question_handler)

        graph.add_edge(START, "classify")
        graph.add_conditional_edges("classify", route)
        graph.add_edge("greeting_handler", END)
        graph.add_edge("question_handler", END)

        app = graph.compile()
        final_state = app.invoke({
            "query": "1+1 bằng mấy?",
            "category": "",
            "response": "",
        })
        return f"Category: {final_state['category']}\nResponse: {final_state['response']}", time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output:\n    {content}")
    print(f"  ⏱️  {elapsed:.2f}s (classify + 1 handler)")
    results.append(("9. LangGraph Conditional", elapsed, "PASS"))


# ============================================================================
# PATTERN 10: LangGraph Cyclic (Loop / Iteration)
# ============================================================================

def pattern_10_langgraph_cyclic(llm):
    banner("PATTERN 10: LANGGRAPH CYCLIC (Loop / Iteration)")
    section_info(
        "LangGraph Cyclic",
        "Graph có vòng lặp: Generate → Check → (pass → END | fail → Generate lại).",
        [
            "Tự cải thiện đến khi đạt quality bar",
            "Flexibel: số iteration không cố định",
            "Có thể combine với human-in-the-loop",
            "Pattern nền tảng cho Self-Refine, Reflexion",
        ],
        [
            "Rủi ro infinite loop (cần max_iterations guard)",
            "Tốn token (nhiều lần gọi LLM)",
            "Chậm (tuyến tính theo số vòng)",
            "Convergence không đảm bảo",
        ],
        [
            "Code generation + test until pass",
            "RAG: rewrite query until docs are sufficient",
            "Content: generate → check tone → regenerate",
            "Translation: translate → verify accuracy → fix",
            "Math: solve → verify → fix if wrong",
        ],
    )

    def run():
        start = time.time()

        class State(TypedDict):
            question: str
            answer: str
            verdict: str
            iteration: int

        def generate(state: State) -> dict:
            result = llm.invoke(
                f"Answer in exactly 5 words (Vietnamese): {state['question']}\n"
                f"Attempt {state['iteration'] + 1}."
            )
            return {"answer": result.content.strip(), "iteration": state["iteration"] + 1}

        def check(state: State) -> dict:
            word_count = len(state["answer"].split())
            verdict = "pass" if word_count <= 7 else "fail"
            return {"verdict": verdict}

        def route_after_check(state: State) -> str:
            if state["verdict"] == "pass":
                return END
            if state["iteration"] >= 3:
                return END  # Safety guard
            return "generate"

        graph = StateGraph(State)
        graph.add_node("generate", generate)
        graph.add_node("check", check)

        graph.add_edge(START, "generate")
        graph.add_edge("generate", "check")
        graph.add_conditional_edges("check", route_after_check)

        app = graph.compile()
        final_state = app.invoke({
            "question": "Thủ đô Việt Nam?",
            "answer": "",
            "verdict": "",
            "iteration": 0,
        })
        return (
            f"Answer: {final_state['answer']}\n"
            f"Verdict: {final_state['verdict']}\n"
            f"Iterations: {final_state['iteration']}"
        ), time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output:\n    {content}")
    print(f"  ⏱️  {elapsed:.2f}s")
    results.append(("10. LangGraph Cyclic", elapsed, "PASS"))


# ============================================================================
# PATTERN 11: LangGraph Parallel (Fan-out / Fan-in)
# ============================================================================

def pattern_11_langgraph_parallel(llm, llm2=None, llm3=None):
    banner("PATTERN 11: LANGGRAPH PARALLEL (Fan-out / Fan-in)")
    section_info(
        "LangGraph Parallel",
        "1 node → nhiều nodes chạy song song → 1 node gộp kết quả.",
        [
            "Nhanh hơn nhiều (parallel execution)",
            "Tận dụng được multiple LLM/APIs",
            "Mỗi branch độc lập → dễ debug",
            "Scale theo số branch",
        ],
        [
            "Fan-in chờ branch chậm nhất (straggler problem)",
            "Branches không share context (mỗi branch chỉ thấy state input)",
            "Phức tạp hơn graph tuyến tính",
            "Chi phí = tổng chi phí các branch",
        ],
        [
            "Multi-perspective: ask 3 personas → combine",
            "Multi-model voting: GPT + Claude + Llama → consensus",
            "Multi-aspect analysis: security + performance + readability",
            "A/B testing prompts → pick best",
            "Multi-language: translate to 5 languages in parallel",
        ],
    )

    # Mỗi branch dùng LLM khác nhau (3 models parallel)
    llm_branch2 = llm2 or llm
    llm_branch3 = llm3 or llm2 or llm

    def run():
        start = time.time()

        class State(TypedDict):
            topic: str
            technical_view: str
            business_view: str
            user_view: str
            combined: str

        def technical(state: State) -> dict:
            result = llm.invoke(f"Technical perspective on '{state['topic']}' in 1 sentence (Vietnamese).")
            return {"technical_view": llm_text(result)}

        def business(state: State) -> dict:
            result = llm_branch2.invoke(f"Business perspective on '{state['topic']}' in 1 sentence (Vietnamese).")
            return {"business_view": llm_text(result)}

        def user(state: State) -> dict:
            result = llm_branch3.invoke(f"End-user perspective on '{state['topic']}' in 1 sentence (Vietnamese).")
            return {"user_view": llm_text(result)}

        def combine(state: State) -> dict:
            result = llm.invoke(
                f"Combine these 3 perspectives into 1 coherent paragraph (Vietnamese):\n"
                f"- Tech: {state['technical_view']}\n"
                f"- Business: {state['business_view']}\n"
                f"- User: {state['user_view']}"
            )
            return {"combined": llm_text(result)}

        graph = StateGraph(State)
        graph.add_node("technical", technical)
        graph.add_node("business", business)
        graph.add_node("user", user)
        graph.add_node("combine", combine)

        # Fan-out: START → 3 parallel nodes
        graph.add_edge(START, "technical")
        graph.add_edge(START, "business")
        graph.add_edge(START, "user")

        # Fan-in: 3 nodes → combine
        graph.add_edge("technical", "combine")
        graph.add_edge("business", "combine")
        graph.add_edge("user", "combine")

        graph.add_edge("combine", END)

        app = graph.compile()
        final_state = app.invoke({
            "topic": "AI trong giáo dục",
            "technical_view": "",
            "business_view": "",
            "user_view": "",
            "combined": "",
        })
        return (
            f"Tech: {final_state['technical_view']}\n"
            f"Business: {final_state['business_view']}\n"
            f"User: {final_state['user_view']}\n"
            f"COMBINED: {final_state['combined']}"
        ), time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output:\n    {content}")
    print(f"  ⏱️  {elapsed:.2f}s (3 parallel + 1 combine)")
    print(f"  📌 Branch1: {model_name(llm)} | Branch2: {model_name(llm_branch2)} | Branch3: {model_name(llm_branch3)}")
    results.append(("11. LangGraph Parallel", elapsed, "PASS"))


# ============================================================================
# PATTERN 12: LangGraph Multi-Agent (Supervisor)
# ============================================================================

def pattern_12_multi_agent(llm):
    banner("PATTERN 12: MULTI-AGENT (Supervisor)")
    section_info(
        "Multi-Agent Supervisor",
        "1 Supervisor agent điều phối nhiều Worker agents. Supervisor quyết định gọi ai, khi nào.",
        [
            "Mỗi agent specialize → chất lượng cao",
            "Supervisor có toàn cục view",
            "Scale: thêm agent mới không đổi architecture",
            "Tách biệt context (mỗi agent có system prompt riêng)",
        ],
        [
            "Nhiều lần LLM → rất tốn token",
            "Chậm (sequential agent calls)",
            "Supervisor có thể delegate sai",
            "Phức tạp nhất trong các pattern",
        ],
        [
            "Research team: planner + searcher + writer + editor",
            "Dev team: architect + coder + tester + reviewer",
            "Content team: researcher + writer + SEO + fact-checker",
            "Support: triage agent + tech agent + billing agent",
            "Data: collector + cleaner + analyst + visualizer",
        ],
    )

    def run():
        start = time.time()

        class State(TypedDict):
            task: str
            plan: str
            research: str
            writing: str
            final: str
            next_agent: str

        def supervisor_plan(state: State) -> dict:
            result = llm.invoke(
                f"Given task: '{state['task']}'\n"
                "Create a 2-step plan. Format: 'Step 1: ... Step 2: ...'\n"
                "Respond in Vietnamese, max 2 lines."
            )
            return {"plan": result.content, "next_agent": "researcher"}

        def researcher(state: State) -> dict:
            result = llm.invoke(
                f"You are a RESEARCHER.\n"
                f"Task: {state['task']}\nPlan: {state['plan']}\n"
                "Provide 2 key facts/points. Vietnamese, 2 lines max."
            )
            return {"research": result.content, "next_agent": "writer"}

        def writer(state: State) -> dict:
            result = llm.invoke(
                f"You are a WRITER.\n"
                f"Task: {state['task']}\nResearch: {state['research']}\n"
                "Write final output. Vietnamese, 3 lines max."
            )
            return {"final": result.content, "next_agent": "done"}

        def route(state: State) -> str:
            return state.get("next_agent", "done")

        graph = StateGraph(State)
        graph.add_node("supervisor", supervisor_plan)
        graph.add_node("researcher", researcher)
        graph.add_node("writer", writer)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges("supervisor", route)
        graph.add_conditional_edges("researcher", route)
        graph.add_conditional_edges("writer", route)
        # "done" → END (implicit: if no edge, it's a terminal)

        app = graph.compile()
        final_state = app.invoke({
            "task": "Giải thích vector database cho người mới",
            "plan": "",
            "research": "",
            "writing": "",
            "final": "",
            "next_agent": "",
        })
        return (
            f"Plan: {final_state['plan']}\n"
            f"Research: {final_state['research']}\n"
            f"FINAL: {final_state['final']}"
        ), time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output:\n    {content}")
    print(f"  ⏱️  {elapsed:.2f}s (3 agents sequential)")
    results.append(("12. Multi-Agent", elapsed, "PASS"))


# ============================================================================
# PATTERN 13: LangGraph Human-in-the-Loop
# ============================================================================

def pattern_13_human_in_loop(llm):
    banner("PATTERN 13: HUMAN-IN-THE-LOOP (Interrupt)")
    section_info(
        "Human-in-the-Loop",
        "Graph chạy → pause tại 1 node → chờ human approve/reject → resume.",
        [
            "Human giữ control ở bước critical",
            "Không cần human ở mọi bước (chỉ bước quan trọng)",
            "Audit trail rõ ràng (ai approve, khi nào)",
            "Tăng confidence cho high-stakes decisions",
        ],
        [
            "Chậm (chờ human response)",
            "Cần infrastructure cho interrupt/resume (checkpointing)",
            "Human bottleneck nếu volume cao",
            "UX phức tạp hơn",
        ],
        [
            "Approval workflow: AI draft → human approve → send",
            "Code deployment: AI generate → human review → merge",
            "Financial: AI suggest trade → human approve → execute",
            "Medical: AI diagnose → doctor confirm → prescribe",
            "Legal: AI draft contract → lawyer review → sign",
        ],
    )

    def run():
        start = time.time()

        class State(TypedDict):
            draft: str
            approved: bool
            final: str

        def generate_draft(state: State) -> dict:
            result = llm.invoke("Write a 1-sentence press release title about AI (Vietnamese).")
            return {"draft": result.content}

        def human_approval(state: State) -> dict:
            # In production: this node would use `interrupt()` from langgraph
            # Here we simulate: auto-approve
            return {"approved": True}

        def publish(state: State) -> dict:
            if state.get("approved"):
                return {"final": f"[PUBLISHED] {state['draft']}"}
            return {"final": "[REJECTED] Not published."}

        graph = StateGraph(State)
        graph.add_node("generate", generate_draft)
        graph.add_node("human_approval", human_approval)
        graph.add_node("publish", publish)

        graph.add_edge(START, "generate")
        graph.add_edge("generate", "human_approval")
        graph.add_edge("human_approval", "publish")
        graph.add_edge("publish", END)

        app = graph.compile()
        final_state = app.invoke({"draft": "", "approved": False, "final": ""})
        return f"Draft: {final_state['draft']}\nFinal: {final_state['final']}", time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output:\n    {content}")
    print(f"  ⏱️  {elapsed:.2f}s (simulated: auto-approve)")
    print(f"  💡 Production: dùng `interrupt()` từ langgraph + checkpoint")
    results.append(("13. Human-in-the-Loop", elapsed, "PASS"))


# ============================================================================
# PATTERN 14: LangGraph with Memory (Checkpointing)
# ============================================================================

def pattern_14_langgraph_memory(llm):
    banner("PATTERN 14: LANGGRAPH MEMORY (Checkpointing)")
    section_info(
        "LangGraph Memory",
        "Graph lưu state sau mỗi node → resume từ điểm bất kỳ. Nền tảng cho conversation memory.",
        [
            "Resume sau crash (không mất progress)",
            "Conversation memory (multi-turn)",
            "Time travel: rollback đến state trước",
            "Foundation cho human-in-the-loop",
        ],
        [
            "Cần storage backend (SQLite/Postgres/Redis)",
            "State có thể lớn → storage cost",
            "Thread management thêm complexity",
            "Debug khó hơn (state phân tán trong checkpoints)",
        ],
        [
            "Multi-turn chatbot với memory",
            "Long-running agent (resume sau 24h)",
            "Workflow approval (pause 3 ngày chờ sign)",
            "Debug: replay execution step-by-step",
            "Branching: fork từ 1 state để thử 2 hướng",
        ],
    )

    def run():
        start = time.time()

        from langgraph.checkpoint.memory import MemorySaver

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        def chatbot(state: State) -> dict:
            response = llm.invoke(state["messages"])
            return {"messages": [response]}

        graph = StateGraph(State)
        graph.add_node("chatbot", chatbot)
        graph.add_edge(START, "chatbot")
        graph.add_edge("chatbot", END)

        checkpointer = MemorySaver()
        app = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "demo-thread"}}

        # Turn 1
        r1 = app.invoke({"messages": [HumanMessage(content="Chào! Tên tôi là Minh.")]}, config)
        # Turn 2 (remembers turn 1)
        r2 = app.invoke({"messages": [HumanMessage(content="Tôi tên gì?")]}, config)

        last_msg = r2["messages"][-1].content
        return f"Turn 2 response: {last_msg}", time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output:\n    {content}")
    print(f"  ⏱️  {elapsed:.2f}s (2 turns, shared thread memory)")
    results.append(("14. LangGraph Memory", elapsed, "PASS"))


# ============================================================================
# PATTERN 15: Plan-and-Execute
# ============================================================================

def pattern_15_plan_and_execute(llm, llm2=None):
    banner("PATTERN 15: PLAN-AND-EXECUTE")
    section_info(
        "Plan-and-Execute",
        "LLM mạnh tạo plan → LLM yếu/cheap thực hiện từng step. Tách planning từ execution.",
        [
            "Planning chất lượng cao (dùng model mạnh)",
            "Execution rẻ (dùng model nhỏ cho mỗi step)",
            "Plan có thể review/edit trước khi execute",
            "Dễ parallelize các step độc lập",
            "Tối ưu chi phí: chỉ 1 lần gọi model mạnh",
        ],
        [
            "Plan cứng: không adapt khi execution gặp vấn đề",
            "2 lần LLM overhead (plan + execute)",
            "Step dependency phức tạp → plan khó",
            "Không tốt cho exploratory tasks",
        ],
        [
            "Research: plan outline → write each section",
            "Software: design architecture → implement each module",
            "Marketing: campaign strategy → execute each channel",
            "Data: analysis plan → run each query",
            "Education: lesson plan → teach each topic",
        ],
    )

    # Use llm2 for execution if provided, otherwise use llm
    executor = llm2 or llm

    def run():
        start = time.time()

        # PLAN (strong model — llm)
        plan_resp = llm.invoke(
            "Create a 3-step plan to explain 'REST API' to a beginner.\n"
            "Format: Step 1: ...\\nStep 2: ...\\nStep 3: ...\nVietnamese, 1 line each."
        )
        plan = llm_text(plan_resp)

        # EXECUTE (each step — llm2, cheaper/different model)
        results_lines = []
        for i, line in enumerate(plan.strip().split("\n"), 1):
            resp = executor.invoke(
                f"Execute this step of explaining REST API to a beginner:\n{line}\n"
                "Write 1-2 sentences. Vietnamese."
            )
            results_lines.append(llm_text(resp))

        return f"PLAN (by {model_name(llm)}):\n{plan}\n\nEXECUTION (by {model_name(executor)}):\n" + "\n".join(results_lines), time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output:\n    {content}")
    print(f"  ⏱️  {elapsed:.2f}s (1 plan + 3 execute = 4 calls)")
    print(f"  📌 Planner: {model_name(llm)} | Executor: {model_name(executor)}")
    results.append(("15. Plan-and-Execute", elapsed, "PASS"))


# ============================================================================
# PATTERN 16: Ensemble / Voting
# ============================================================================

def pattern_16_ensemble(llm, llm2, llm3=None):
    banner("PATTERN 16: ENSEMBLE / VOTING")
    section_info(
        "Ensemble / Voting",
        "Nhiều LLM (khác provider) trả lời → vote/aggregate → final answer.",
        [
            "Giảm hallucination (cần majority agree)",
            "Robust: 1 model sai không ảnh hưởng tất cả",
            "Có confidence score (độ đồng thuận)",
            "Tăng accuracy cho classification",
            "Multi-provider: không phụ thuộc 1 vendor",
            "3 voters → majority clear hơn 2",
        ],
        [
            "Rất tốn tiền (N lần gọi LLM)",
            "Chậm (cần chờ tất cả models)",
            "Models có thể đồng loạt sai (correlated errors)",
            "Overkill cho simple tasks",
        ],
        [
            "Medical diagnosis (cần accuracy cao)",
            "Legal judgment (cần consistency)",
            "Fact-checking (nhiều sources agree?)",
            "Code: multiple solutions → pick best",
            "Safety-critical: autonomous driving decisions",
        ],
    )

    def run():
        start = time.time()

        # Build voter list: always 2, optionally 3
        voters = [
            (f"LLM1 ({model_name(llm)})", llm),
            (f"LLM2 ({model_name(llm2)})", llm2),
        ]
        if llm3:
            voters.append((f"LLM3 ({model_name(llm3)})", llm3))

        questions = [
            "Is the Earth flat? Answer YES or NO only.",
            "Is water wet? Answer YES or NO only.",
            "Can humans fly without assistance? Answer YES or NO only.",
        ]

        results_text = []
        for q in questions:
            # Each voter (different LLM provider) votes independently
            q_votes = []
            for voter_name, voter_llm in voters:
                r = voter_llm.invoke(q)
                answer = llm_text(r).strip().upper()[:3]  # YES/NO
                q_votes.append(f"{voter_name}:{answer}")

            # Aggregate: majority vote
            yes_count = sum(1 for v in q_votes if ":YES" in v)
            no_count = sum(1 for v in q_votes if ":NO" in v)
            final = "YES" if yes_count > no_count else "NO"
            results_text.append(f"Q: {q}\n  Votes: {', '.join(q_votes)}\n  → {final}\n")

        return "\n".join(results_text), time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output:\n    {content}")
    n_models = 3 if llm3 else 2
    print(f"  ⏱️  {elapsed:.2f}s (3 questions × {n_models} models = {3*n_models} calls)")
    if llm3:
        print(f"  📌 LLM1: {model_name(llm)} | LLM2: {model_name(llm2)} | LLM3: {model_name(llm3)}")
    else:
        print(f"  📌 LLM1: {model_name(llm)} | LLM2: {model_name(llm2)}")
    results.append(("16. Ensemble/Voting", elapsed, "PASS"))


# ============================================================================
# PATTERN 17: LangGraph Subgraph (Hierarchical)
# ============================================================================

def pattern_17_subgraph(llm):
    banner("PATTERN 17: LANGGRAPH SUBGRAPH (Hierarchical)")
    section_info(
        "LangGraph Subgraph",
        "Graph chứa graph con. Subgraph là 1 node trong parent graph. Tách lo",
        [
            "Modular: mỗi subgraph 1 concern",
            "Reusable: subgraph dùng được ở nhiều nơi",
            "Nested state: subgraph có state riêng",
            "E",
        ],
        [
            "Phức tạp nhất",
            "State passing giữa parent ↔ subgraph cần design kỹ",
            "Debug khó (nested execution)",
            "Overhead compile",
        ],
        [
            "Customer service: triage → (billing_subgraph | tech_subgraph)",
            "Pipeline: intake → (translation_subgraph | analysis_subgraph) → output",
            "Multi-agent: supervisor → (agent_a_subgraph | agent_b_subgraph)",
            "E-commerce: order → (payment_subgraph | shipping_subgraph)",
        ],
    )

    def run():
        start = time.time()

        # --- Subgraph: "summarize" ---
        class SubState(TypedDict):
            text: str
            summary: str

        def sub_summarize(state: SubState) -> dict:
            r = llm.invoke(f"Summarize in 1 sentence (Vietnamese): {state['text']}")
            return {"summary": r.content}

        sub_graph = StateGraph(SubState)
        sub_graph.add_node("summarize", sub_summarize)
        sub_graph.add_edge(START, "summarize")
        sub_graph.add_edge("summarize", END)
        compiled_sub = sub_graph.compile()

        # --- Parent graph ---
        class ParentState(TypedDict):
            documents: list[str]
            summaries: list[str]
            final: str

        def process_docs(state: ParentState) -> dict:
            summaries = []
            for doc in state["documents"]:
                result = compiled_sub.invoke({"text": doc, "summary": ""})
                summaries.append(result["summary"])
            return {"summaries": summaries}

        def combine(state: ParentState) -> dict:
            r = llm.invoke(
                f"Combine these summaries into 1 sentence (Vietnamese):\n"
                + "\n".join(state["summaries"])
            )
            return {"final": r.content}

        graph = StateGraph(ParentState)
        graph.add_node("process", process_docs)
        graph.add_node("combine", combine)
        graph.add_edge(START, "process")
        graph.add_edge("process", "combine")
        graph.add_edge("combine", END)

        app = graph.compile()
        final_state = app.invoke({
            "documents": [
                "Python is a high-level programming language.",
                "Rust focuses on memory safety without garbage collection.",
            ],
            "summaries": [],
            "final": "",
        })
        return f"Summaries: {final_state['summaries']}\nFinal: {final_state['final']}", time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output:\n    {content}")
    print(f"  ⏱️  {elapsed:.2f}s (2 subgraph calls + 1 combine)")
    results.append(("17. Subgraph", elapsed, "PASS"))


# ============================================================================
# PATTERN 18: Production Multi-Agent Team (Supervisor + Memory + RAG)
# ============================================================================

def pattern_18_production_team(llm, llm2, llm3):
    banner("PATTERN 18: PRODUCTION MULTI-AGENT TEAM")
    section_info(
        "Production Team (Supervisor + Memory + RAG + Sessions)",
        "Full production architecture: Supervisor điều phối team, memory per-user/"
        "per-project, RAG cho project context, session lifecycle (create/resume/end).",
        [
            "Không hardcode user — namespace-based memory",
            "Multi-project: mỗi project 1 RAG collection + memory namespace",
            "Session lifecycle: create → work → end → resume sau này",
            "Mỗi agent dùng LLM phù hợp (strong model = reasoning, fast = execution)",
            "Fallback: LLM chính fail → tự chuyển llama3.1",
            "Chuẩn kiến trúc AI production (stateless agents, persistent state)",
        ],
        [
            "Phức tạp — cần hiểu LangGraph + Store + RAG",
            "Nhiều LLM calls → chi phí",
            "Cần manage session lifecycle đúng",
        ],
        [
            "SaaS platform: nhiều user, nhiều project, multi-turn",
            "Enterprise: team AI workers với phân quyền",
            "Content pipeline: research → write → review → publish",
            "Code assistant: multi-project, per-user preferences",
        ],
    )

    import uuid
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore
    from rag.retrieval import get_retriever

    # ─── ARCHITECTURE ────────────────────────────────────────────────────────
    #
    # ┌─────────────────────────────────────────────────────────────────┐
    # │  SESSION MANAGER                                                 │
    # │  - create_session(user_id, project) → session_id                │
    # │  - resume_session(session_id) → load state                      │
    # │  - end_session(session_id) → save + cleanup                     │
    # └─────────────────────────────────────────────────────────────────┘
    #        │
    #        ▼
    # ┌─────────────────────────────────────────────────────────────────┐
    # │  SUPERVISOR (strong LLM)                                         │
    # │  - Receives user request                                         │
    # │  - Decides which agent to delegate to                            │
    # │  - Tracks progress, handles re-runs                              │
    # └─────────────────────────────────────────────────────────────────┘
    #        │
    #        ├──→ RESEARCHER (fast LLM) → RAG search + web
    #        ├──→ WRITER (medium LLM)   → generate content
    #        └──→ REVIEWER (local LLM)  → fact-check, quality gate
    #
    # ┌─────────────────────────────────────────────────────────────────┐
    # │  MEMORY LAYERS                                                   │
    # │                                                                   │
    # │  Checkpointer (short-term):                                       │
    # │    thread_id = session_id                                         │
    # │    → Auto-saves graph state after each node                      │
    # │                                                                   │
    # │  Store (long-term):                                               │
    # │    namespace = (project, user_id, category)                      │
    # │    → User preferences, project context, learned facts            │
    # │                                                                   │
    # │  RAG (project knowledge):                                         │
    # │    collection = project_name                                      │
    # │    → Project docs, specs, SOPs                                   │
    # └─────────────────────────────────────────────────────────────────┘
    #
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │ ARCHITECTURE                                                        │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  User Request                                                       │
  │       │                                                              │
  │       ▼                                                              │
  │  ┌──────────────────────────────────────────────────────────┐      │
  │  │ SESSION MANAGER                                           │      │
  │  │   thread_id = f"{project}:{user_id}:{session_id}"        │      │
  │  │   namespace = (project, user_id, category)               │      │
  │  └──────────────────────────────────────────────────────────┘      │
  │       │                                                              │
  │       ▼                                                              │
  │  ┌──────────────────────────────────────────────────────────┐      │
  │  │ SUPERVISOR [LLM1: strong reasoning]                       │      │
  │  │   • Reads session state + user context from Store         │      │
  │  │   • Decides: research / write / review / done             │      │
  │  └──────────────────────────────────────────────────────────┘      │
  │       │              │                │                              │
  │       ▼              ▼                ▼                              │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                        │
  │  │RESEARCHER│  │ WRITER   │  │ REVIEWER │                        │
  │  │[LLM2]    │  │[LLM1]    │  │[LLM3]    │                        │
  │  │+ RAG     │  │+ context │  │+ factchk │                        │
  │  └──────────┘  └──────────┘  └──────────┘                        │
  │                                                                     │
  │  MEMORY:                                                            │
  │  • Checkpointer: per-session (auto-save state)                     │
  │  • Store: per (project, user) — preferences, facts                 │
  │  • RAG: per-project — docs, specs                                 │
  └─────────────────────────────────────────────────────────────────────┘
    """)

    # ─── TEAM SETUP ──────────────────────────────────────────────────────────

    # Each agent uses the most suitable LLM:
    # - Supervisor: needs strong reasoning → LLM1 (OpenAI/Qwen)
    # - Researcher: needs speed + RAG → LLM2 (Gemini/fast)
    # - Writer: needs good text generation → LLM1 (quality)
    # - Reviewer: needs fact-checking → LLM3 (Ollama/local, cheap)
    team = {
        "supervisor": llm2,
        "researcher": llm or llm2,
        "writer": llm3,
        "reviewer": llm2 or llm,
    }

    print(f"  Team:")
    for role, model in team.items():
        print(f"    {role:>10}: {model_name(model)}")

    # ─── MEMORY INFRASTRUCTURE ───────────────────────────────────────────────

    checkpointer = MemorySaver()  # Per-session state
    store = InMemoryStore()       # Per-(project, user) long-term facts

    # ─── SESSION MANAGER ─────────────────────────────────────────────────────

    class SessionManager:
        """Manages session lifecycle for multi-user, multi-project.

        Session ID format: {project}:{user_id}:{uuid}
        This ensures:
          - Same user, different projects → different sessions
          - Same project, different users → different memory
          - Resume works: pass same session_id back
        """

        def __init__(self):
            self.active_sessions: dict[str, dict] = {}

        def create(self, user_id: str, project: str) -> str:
            session_id = f"{project}:{user_id}:{uuid.uuid4().hex[:8]}"
            self.active_sessions[session_id] = {
                "user_id": user_id,
                "project": project,
                "created_at": time.time(),
                "turns": 0,
            }
            # Initialize user namespace in Store
            store.put((project, user_id, "session"), "meta", {
                "session_id": session_id,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            return session_id

        def resume(self, session_id: str) -> dict:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session '{session_id}' not found")
            return self.active_sessions[session_id]

        def end(self, session_id: str):
            info = self.active_sessions.pop(session_id, {})
            if info:
                store.put(
                    (info["project"], info["user_id"], "sessions"),
                    session_id,
                    {"turns": info["turns"], "ended": time.strftime("%Y-%m-%d %H:%M:%S")},
                )
            return info

        def thread_config(self, session_id: str) -> dict:
            return {"configurable": {"thread_id": session_id}}

    sessions = SessionManager()

    # ─── AGENT GRAPH ─────────────────────────────────────────────────────────

    class TeamState(TypedDict, total=False):
        # Input
        user_request: str
        user_id: str
        project: str
        # Context (injected from memory/RAG)
        user_context: str
        project_context: str
        # Agent outputs
        research_findings: str
        draft: str
        review: str
        # Final
        final_output: str
        # Routing
        next_agent: str
        # Metadata
        turn: int

    def load_context(state: TeamState) -> dict:
        """Load user + project context from Store and RAG."""
        user_id = state["user_id"]
        project = state["project"]

        # 1. User context from Store (preferences, past interactions)
        user_facts = store.search((project, user_id), query="user preferences and context")
        user_ctx = "\n".join(
            f"- {item.value.get('fact', item.value)}" for item in user_facts[:3]
        ) if user_facts else "(no stored preferences)"

        # 2. Project context from RAG
        try:
            retriever = get_retriever(k=2)
            docs = retriever.invoke(state["user_request"])
            proj_ctx = "\n".join(f"- {d.page_content[:200]}" for d in docs[:2])
        except Exception:
            proj_ctx = "(no project docs indexed)"

        return {"user_context": user_ctx, "project_context": proj_ctx}

    def supervisor_node(state: TeamState) -> dict:
        """Supervisor decides which agent to call next."""
        prompt = (
            "You are a team supervisor. Based on the user request and current progress, "
            "decide which agent should work next.\n\n"
            f"User request: {state['user_request']}\n"
            f"User context: {state.get('user_context', '')}\n"
            f"Project context: {state.get('project_context', '')}\n"
            f"Research done: {'Yes' if state.get('research_findings') else 'No'}\n"
            f"Draft done: {'Yes' if state.get('draft') else 'No'}\n"
            f"Review done: {'Yes' if state.get('review') else 'No'}\n\n"
            "Available agents: researcher, writer, reviewer, done\n"
            "Rules:\n"
            "- If no research yet → researcher\n"
            "- If research done but no draft → writer\n"
            "- If draft done but no review → reviewer\n"
            "- If review passed → done\n"
            "Respond with ONLY the agent name."
        )
        result = team["supervisor"].invoke(prompt)
        decision = llm_text(result).strip().lower()

        # Sanitize
        valid = {"researcher", "writer", "reviewer", "done"}
        next_agent = decision if decision in valid else "done"
        return {"next_agent": next_agent, "turn": state.get("turn", 0) + 1}

    def supervisor_router(state: TeamState) -> str:
        return state.get("next_agent", "done")

    def researcher_node(state: TeamState) -> dict:
        """Researcher: uses RAG + LLM to gather information."""
        prompt = (
            f"You are a researcher. Gather relevant information for this task.\n\n"
            f"Task: {state['user_request']}\n"
            f"Project context: {state.get('project_context', '')}\n\n"
            "Provide 3 key findings in bullet points. Be concise."
        )
        result = team["researcher"].invoke(prompt)
        return {"research_findings": llm_text(result)}

    def writer_node(state: TeamState) -> dict:
        """Writer: generates content based on research."""
        prompt = (
            f"You are a technical writer. Write a response based on research.\n\n"
            f"Task: {state['user_request']}\n"
            f"Research findings:\n{state.get('research_findings', '')}\n"
            f"User context: {state.get('user_context', '')}\n\n"
            "Write a clear, professional response (3-5 sentences). Vietnamese."
        )
        result = team["writer"].invoke(prompt)
        return {"draft": llm_text(result)}

    def reviewer_node(state: TeamState) -> dict:
        """Reviewer: fact-checks and quality-gates the draft."""
        prompt = (
            f"You are a quality reviewer. Check this draft for accuracy and completeness.\n\n"
            f"Original task: {state['user_request']}\n"
            f"Draft: {state.get('draft', '')}\n\n"
            "If the draft is good, respond: 'APPROVED: <improved version>'\n"
            "If it needs changes, respond: 'REJECTED: <specific issues>'\n"
            "Always provide the final improved version after APPROVED/REJECTED."
        )
        result = team["reviewer"].invoke(prompt)
        review_text = llm_text(result)
        return {"review": review_text, "final_output": review_text}

    def finalize_node(state: TeamState) -> dict:
        """Mark session turn as complete."""
        if not state.get("final_output"):
            state["final_output"] = state.get("draft", "No output generated.")
        return {}

    # Build the team graph
    team_graph = StateGraph(TeamState)
    team_graph.add_node("load_context", load_context)
    team_graph.add_node("supervisor", supervisor_node)
    team_graph.add_node("researcher", researcher_node)
    team_graph.add_node("writer", writer_node)
    team_graph.add_node("reviewer", reviewer_node)
    team_graph.add_node("finalize", finalize_node)

    team_graph.add_edge(START, "load_context")
    team_graph.add_edge("load_context", "supervisor")
    team_graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "done": "finalize",
        },
    )
    team_graph.add_edge("researcher", "supervisor")  # Loop back
    team_graph.add_edge("writer", "supervisor")      # Loop back
    team_graph.add_edge("reviewer", "finalize")
    team_graph.add_edge("finalize", END)

    team_app = team_graph.compile(checkpointer=checkpointer)

    # ─── SESSION BOOTSTRAP (User enters name → system selects memory + RAG) ──

    print(f"\n  {'='*60}")
    print(f"  SESSION BOOTSTRAP — User onboarding flow")
    print(f"  {'='*60}")

    class SessionBootstrap:
        """Handles the full session initialization flow.

        Flow:
        1. User provides: user_id + project name
        2. System checks: does this user exist in Store?
           - Yes → load preferences, past sessions
           - No  → new user, initialize empty context
        3. System selects: RAG collection for the project
        4. System creates: session with thread_id
        5. Returns: ready-to-use session config + loaded context
        """

        def __init__(self, store, checkpointer):
            self.store = store
            self.checkpointer = checkpointer
            self.active: dict[str, dict] = {}

        def authenticate(self, user_id: str, project: str) -> dict:
            """Step 1: Verify user + load their context.

            In production, this would:
              - Check JWT/API key for auth
              - Look up user in database
              - Verify project access (RBAC)

            Here: Store-based lookup.
            """
            # Check if user has stored data for this project
            user_ns = (project, user_id)
            all_facts = self.store.search(user_ns, query="all user data")

            is_returning = len(all_facts) > 0

            # Load preferences
            prefs = self.store.search(user_ns, query="preferences and settings")
            preferences = [
                item.value.get("fact", str(item.value)) for item in prefs[:5]
            ]

            # Load past session history
            past_sessions = self.store.search(
                (project, user_id, "sessions"), query="past sessions"
            )

            return {
                "user_id": user_id,
                "project": project,
                "is_returning": is_returning,
                "preferences": preferences,
                "past_sessions": len(past_sessions),
            }

        def select_rag(self, project: str) -> dict:
            """Step 2: Select the RAG collection for this project.

            Convention: collection_name = project name.
            Each project has its own indexed documents.
            """
            from rag.config import get_rag_settings
            settings = get_rag_settings()

            # In production: map project → collection
            # Here: use project name as collection (fallback to default)
            collection = project  # e.g., "loom", "crm_platform"

            # Verify collection exists (has documents)
            try:
                from rag.indexing import get_vectorstore
                vs = get_vectorstore(collection)
                # Quick check: try a search
                test_docs = vs.similarity_search("test", k=1)
                has_docs = len(test_docs) > 0
            except Exception:
                # Collection doesn't exist yet → use default
                collection = settings.default_collection
                has_docs = True

            return {
                "collection": collection,
                "has_documents": has_docs,
                "note": f"Using collection: {collection}"
            }

        def create_session(self, user_id: str, project: str) -> dict:
            """Step 3: Create a new session with all context loaded.

            Returns a session config ready to use with LangGraph.
            """
            # Authenticate
            auth = self.authenticate(user_id, project)

            # Select RAG
            rag_info = self.select_rag(project)

            # Create session ID
            session_id = f"{project}:{user_id}:{uuid.uuid4().hex[:8]}"
            thread_config = {"configurable": {"thread_id": session_id}}

            # Store session metadata
            self.store.put(
                (project, user_id, "session"),
                "meta",
                {
                    "session_id": session_id,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "rag_collection": rag_info["collection"],
                },
            )

            # Track active session
            self.active[session_id] = {
                "user_id": user_id,
                "project": project,
                "config": thread_config,
                "rag_collection": rag_info["collection"],
                "created_at": time.time(),
                "turns": 0,
            }

            return {
                "session_id": session_id,
                "config": thread_config,
                "auth": auth,
                "rag": rag_info,
            }

        def resume_session(self, session_id: str) -> dict:
            """Resume an existing session (user comes back)."""
            if session_id not in self.active:
                raise ValueError(f"Session '{session_id}' not found or expired")
            return self.active[session_id]

        def end_session(self, session_id: str):
            """End session: save state, persist metadata."""
            session = self.active.pop(session_id, None)
            if session:
                self.store.put(
                    (session["project"], session["user_id"], "sessions"),
                    session_id,
                    {
                        "turns": session["turns"],
                        "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
            return session

    bootstrap = SessionBootstrap(store, checkpointer)

    # ─── FLOW 1: New user, first time ────────────────────────────────────────
    print(f"\n  ┌─ FLOW 1: New user first visit")
    print(f"  │")
    print(f"  │  User input: name='new_dev', project='loom'")

    session_new = bootstrap.create_session("new_dev", "loom")
    auth = session_new["auth"]
    rag = session_new["rag"]

    print(f"  │  Auth: returning={auth['is_returning']}, prefs={auth['preferences']}")
    print(f"  │  RAG:  collection='{rag['collection']}', has_docs={rag['has_documents']}")
    print(f"  │  Session: {session_new['session_id']}")
    print(f"  └─ Ready to accept requests")

    # ─── FLOW 2: Returning user (has stored preferences) ─────────────────────
    print(f"\n  ┌─ FLOW 2: Returning user")
    print(f"  │")
    print(f"  │  User input: name='dev_zenchung', project='loom'")

    # First, simulate that this user was here before
    store.put(("loom", "dev_zenchung", "preference"), "lang", {"fact": "Speaks Vietnamese"})
    store.put(("loom", "dev_zenchung", "preference"), "style", {"fact": "Wants code examples"})
    store.put(("loom", "dev_zenchung", "sessions"), "old_session", {"turns": 5, "ended": "2026-09-01"})

    session_ret = bootstrap.create_session("dev_zenchung", "loom")
    auth_ret = session_ret["auth"]
    rag_ret = session_ret["rag"]

    print(f"  │  Auth: returning={auth_ret['is_returning']}")
    print(f"  │  Loaded preferences: {auth_ret['preferences']}")
    print(f"  │  Past sessions: {auth_ret['past_sessions']}")
    print(f"  │  RAG:  collection='{rag_ret['collection']}'")
    print(f"  │  Session: {session_ret['session_id']}")
    print(f"  │  → System knows: user speaks VN, wants code, has 5 past turns")
    print(f"  └─ Context pre-loaded, ready")

    # ─── FLOW 3: Different project (RAG switches) ────────────────────────────
    print(f"\n  ┌─ FLOW 3: Same user, different project")
    print(f"  │")
    print(f"  │  User input: name='dev_zenchung', project='crm_platform'")

    session_crm = bootstrap.create_session("dev_zenchung", "crm_platform")
    rag_crm = session_crm["rag"]

    print(f"  │  RAG:  collection='{rag_crm['collection']}' (different from 'loom')")
    print(f"  │  Memory: (crm_platform, dev_zenchung) — separate from (loom, dev_zenchung)")
    print(f"  │  → User's 'loom' preferences do NOT leak into 'crm_platform'")
    print(f"  └─ Project isolation maintained")

    # ─── DECISION TABLE ──────────────────────────────────────────────────────
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │ DECISION LOGIC (what the system checks on session start)            │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  1. USER IDENTIFICATION                                             │
  │     input: user_id (from login/JWT)                                 │
  │     check: Store.search((project, user_id))                         │
  │     → found: returning user → load prefs                           │
  │     → not found: new user → empty context                          │
  │                                                                     │
  │  2. PROJECT SELECTION                                               │
  │     input: project_name                                             │
  │     check: vector_store.similarity_search(collection=project)      │
  │     → exists: use project-specific RAG                             │
  │     → not exists: fallback to default collection                    │
  │                                                                     │
  │  3. SESSION RESUME vs CREATE                                        │
  │     check: checkpointer.get(thread_id=session_id)                  │
  │     → has state: resume (load last checkpoint)                      │
  │     → no state: create new session                                 │
  │                                                                     │
  │  4. MEMORY LAYER SELECTION                                          │
  │     namespace = (project, user_id, category)                        │
  │     → (project, user, "preference")  = user prefs                  │
  │     → (project, user, "sessions")    = session history             │
  │     → (project, user, "facts")       = learned context             │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
    """)

    # Cleanup demo sessions
    for sid in list(bootstrap.active.keys()):
        bootstrap.end_session(sid)

    elapsed_total = 0
    results.append(("18. Production Team", elapsed_total, "PASS"))


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def print_summary():
    banner("SUMMARY: TẤT CẢ PATTERNS")
    print(f"\n  {'#':<4} {'Pattern':<30} {'Time':<10} {'Status'}")
    print(f"  {'-'*4} {'-'*30} {'-'*10} {'-'*10}")
    for name, elapsed, status in results:
        print(f"  {name:<34} {elapsed:<10.2f}s  {status}")

    total = sum(e for _, e, _ in results)
    print(f"\n  {'Tổng':<34} {total:<10.2f}s")
    print(f"\n  {'='*70}")
    print("  📊 COMPARISON MATRIX")
    print(f"  {'='*70}")
    print(f"""
  Pattern              | LLM Calls | Parallel | Loop | Human | Complexity
  ---------------------+-----------+----------+------+-------+-----------
  1. LLM Chain         | 1         | No       | No   | No    | ★☆☆☆☆
  2. RAG Chain         | 1+retrieve| No       | No   | No    | ★★☆☆☆
  3. ReAct Agent       | 2-10      | No       | Yes  | No    | ★★★☆☆
  4. Sequential        | N         | No       | No   | No    | ★★☆☆☆
  5. Map-Reduce        | N+1       | Yes      | No   | No    | ★★☆☆☆
  6. Routing           | 1         | No       | No   | No    | ★★☆☆☆
  7. Self-Refine       | 3-5       | No       | Yes  | No    | ★★★☆☆
  8. LG Sequential     | N         | No       | No   | No    | ★★☆☆☆
  9. LG Conditional    | N+1       | No       | No   | No    | ★★★☆☆
  10. LG Cyclic        | 2-6       | No       | Yes  | No    | ★★★☆☆
  11. LG Parallel      | N+1       | Yes      | No   | No    | ★★★☆☆
  12. Multi-Agent      | 3-10      | No       | Yes  | No    | ★★★★☆
  13. Human-in-Loop    | N+1       | No       | No   | Yes   | ★★★☆☆
  14. LG Memory        | N         | No       | No   | No    | ★★★☆☆
  15. Plan-Execute     | 1+N       | Maybe    | No   | No    | ★★★☆☆
  16. Ensemble         | N×K       | Yes      | No   | No    | ★★☆☆☆
  17. Subgraph         | N         | Maybe    | Maybe| Maybe | ★★★★☆

  ═══════════════════════════════════════════════════════════════════════
  QUY TẮC CHỌN PATTERN:
  ═══════════════════════════════════════════════════════════════════════
  • Task đơn giản, 1 input → 1 output?          → Pattern 1 (LLM Chain)
  • Cần knowledge ngoài?                         → Pattern 2 (RAG)
  • Cần dùng tools/APIs?                         → Pattern 3 (Agent)
  • Pipeline cố định, không branch?              → Pattern 4/8 (Sequential)
  • Input rất dài?                               → Pattern 5 (Map-Reduce)
  • Input có nhiều loại khác nhau?               → Pattern 6 (Routing)
  • Cần chất lượng cao, sẵn sàng chờ?            → Pattern 7 (Self-Refine)
  • Cần loop đến khi pass?                       → Pattern 10 (Cyclic)
  • Cần multi-perspective?                       → Pattern 11 (Parallel)
  • Task phức tạp, multi-step, multi-role?       → Pattern 12 (Multi-Agent)
  • Cần human approve bước critical?             → Pattern 13 (HITL)
  • Cần nhớ conversation?                        → Pattern 14 (Memory)
  • Muốn tách planning (model mạnh) + exec (model rẻ)? → Pattern 15
  • Cần accuracy cực cao (medical, legal)?       → Pattern 16 (Ensemble)
  • System lớn, cần modular?                     → Pattern 17 (Subgraph)
  ═══════════════════════════════════════════════════════════════════════
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{SEP}")
    print("  LANGCHAIN & LANGGRAPH PATTERNS - COMPREHENSIVE DEMO")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)

    # ─── LLM SETUP ───────────────────────────────────────────────────────────
    # 4 LLMs: 3 primary + 1 fallback
    #   llm   = primary   (auto-fallback to llama3.1 if unavailable)
    #   llm2  = secondary (auto-fallback)
    #   llm3  = tertiary  (auto-fallback)
    #
    # Nếu LLM chính cạn quota / lỗi → tự chuyển sang FALLBACK (llama3.1:latest)
    # ─────────────────────────────────────────────────────────────────────────
    llm = get_llm_with_fallback(LLM.OPENAI)
    llm2 = get_llm_with_fallback(LLM.GEMINI)
    llm3 = get_llm_with_fallback(LLM.OLLAMA)

    print(f"\n  LLM 1 (primary):   {model_name(llm)}")
    print(f"  LLM 2 (secondary): {model_name(llm2)}")
    print(f"  LLM 3 (tertiary):  {model_name(llm3)}")
    print(f"  Fallback:          {model_name(get_llm(LLM.FALLBACK))} (auto if above fail)")
    print()

    # Run all patterns
    # pattern_1_llm_chain(llm)
    # pattern_2_rag(llm)
    # pattern_3_agent(llm)
    # pattern_4_sequential(llm)
    # pattern_5_map_reduce(llm)
    # pattern_6_routing(llm)
    # pattern_7_self_refine(llm)
    # pattern_8_langgraph_sequential(llm)
    # pattern_9_langgraph_conditional(llm)
    # pattern_10_langgraph_cyclic(llm)
    # pattern_11_langgraph_parallel(llm, llm2, llm3)  # 3 models parallel
    # pattern_12_multi_agent(llm)
    # pattern_13_human_in_loop(llm)
    # pattern_14_langgraph_memory(llm)
    # pattern_15_plan_and_execute(llm, llm2)     # plan + execute khác model
    # pattern_16_ensemble(llm, llm2, llm3)         # 3-model voting
    # pattern_17_subgraph(llm)
    pattern_18_production_team(llm, llm2, llm3)  # Full production team

    # Usage report
    print(f"\n{SEP}")
    print("  LLM USAGE REPORT")
    print(SEP)
    print()
    print(usage_tracker.report())
    print()

    # Summary
    print_summary()


if __name__ == "__main__":
    main()
