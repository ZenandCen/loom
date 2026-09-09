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

from utils.models import LLM, get_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Separator for section output
SEP = "=" * 70
THIN_SEP = "-" * 50

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
    print(f"\n  🤖 Output: {content[:200]}...")
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
    print(f"\n  🤖 Output: {content[:200]}...")
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
    print(f"\n  🤖 Output: {content[:200]}")
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
    print(f"\n  🤖 Output: {content[:200]}")
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

def pattern_11_langgraph_parallel(llm):
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
            return {"technical_view": result.content}

        def business(state: State) -> dict:
            result = llm.invoke(f"Business perspective on '{state['topic']}' in 1 sentence (Vietnamese).")
            return {"business_view": result.content}

        def user(state: State) -> dict:
            result = llm.invoke(f"End-user perspective on '{state['topic']}' in 1 sentence (Vietnamese).")
            return {"user_view": result.content}

        def combine(state: State) -> dict:
            result = llm.invoke(
                f"Combine these 3 perspectives into 1 coherent paragraph (Vietnamese):\n"
                f"- Tech: {state['technical_view']}\n"
                f"- Business: {state['business_view']}\n"
                f"- User: {state['user_view']}"
            )
            return {"combined": result.content}

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

def pattern_15_plan_and_execute(llm):
    banner("PATTERN 15: PLAN-AND-EXECUTE")
    section_info(
        "Plan-and-Execute",
        "LLM mạnh tạo plan → LLM yếu/cheap thực hiện từng step. Tách planning từ execution.",
        [
            "Planning chất lượng cao (dùng model mạnh)",
            "Execution rẻ (dùng model nhỏ cho mỗi step)",
            "Plan có thể review/edit trước khi execute",
            "Dễ parallelize các step độc lập",
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

    def run():
        start = time.time()

        # PLAN (strong model)
        plan = llm.invoke(
            "Create a 3-step plan to explain 'REST API' to a beginner.\n"
            "Format: Step 1: ...\\nStep 2: ...\\nStep 3: ...\nVietnamese, 1 line each."
        ).content

        # EXECUTE (each step)
        results_lines = []
        for i, line in enumerate(plan.strip().split("\n"), 1):
            result = llm.invoke(
                f"Execute this step of explaining REST API to a beginner:\n{line}\n"
                "Write 1-2 sentences. Vietnamese."
            ).content
            results_lines.append(result)

        return f"PLAN:\n{plan}\n\nEXECUTION:\n" + "\n".join(results_lines), time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output:\n    {content}")
    print(f"  ⏱️  {elapsed:.2f}s (1 plan + 3 execute = 4 calls)")
    results.append(("15. Plan-and-Execute", elapsed, "PASS"))


# ============================================================================
# PATTERN 16: Ensemble / Voting
# ============================================================================

def pattern_16_ensemble(llm):
    banner("PATTERN 16: ENSEMBLE / VOTING")
    section_info(
        "Ensemble / Voting",
        "Nhiều LLM (hoặc cùng LLM với temp khác) trả lời → vote/aggregate → final answer.",
        [
            "Giảm hallucination (cần majority agree)",
            "Robust: 1 model sai không ảnh hưởng tất cả",
            "Có confidence score (độ đồng thuận)",
            "Tăng accuracy cho classification",
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

        # 3 "voters" (same model, different prompts to simulate diversity)
        questions = [
            "Is the Earth flat? Answer YES or NO only.",
            "Is water wet? Answer YES or NO only.",
            "Can humans fly without assistance? Answer YES or NO only.",
        ]

        votes = []
        for q in questions:
            # 3 votes per question
            q_votes = []
            for _ in range(3):
                r = llm.invoke(q)
                q_votes.append(r.content.strip().upper()[:3])  # YES/NO
            votes.append(q_votes)

        # Aggregate
        results_text = []
        for i, (q, v) in enumerate(zip(questions, votes)):
            yes_count = sum(1 for x in v if x.startswith("YES"))
            final = "YES" if yes_count > 1 else "NO"
            results_text.append(f"Q{i+1} ({v}): → {final}")

        return "\n".join(results_text), time.time() - start

    content, elapsed = run()
    print(f"\n  🤖 Output:\n    {content}")
    print(f"  ⏱️  {elapsed:.2f}s (3 questions × 3 votes = 9 calls)")
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

    llm = get_llm(LLM.OPENAI)

    # Run all patterns
    pattern_1_llm_chain(llm)
    pattern_2_rag(llm)
    pattern_3_agent(llm)
    pattern_4_sequential(llm)
    pattern_5_map_reduce(llm)
    pattern_6_routing(llm)
    pattern_7_self_refine(llm)
    pattern_8_langgraph_sequential(llm)
    pattern_9_langgraph_conditional(llm)
    pattern_10_langgraph_cyclic(llm)
    pattern_11_langgraph_parallel(llm)
    pattern_12_multi_agent(llm)
    pattern_13_human_in_loop(llm)
    pattern_14_langgraph_memory(llm)
    pattern_15_plan_and_execute(llm)
    pattern_16_ensemble(llm)
    pattern_17_subgraph(llm)

    # Summary
    print_summary()


if __name__ == "__main__":
    main()
