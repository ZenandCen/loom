%% RAG Integration — Sequence Diagrams
%% Format: Mermaid (render on GitHub, VS Code, or https://mermaid.live)

%% ═══════════════════════════════════════════════════════════════
%% 1. INDEXING (Offline — chạy trước khi query)
%% ═══════════════════════════════════════════════════════════════
sequenceDiagram
    title RAG Indexing Pipeline (Offline)

    participant User
    participant CLI as index_documents()
    participant Loader as load_documents()
    participant Chunker as chunk_documents()
    participant Embedder as OpenAIEmbeddings
    participant VectorDB as Vector Store

    User->>CLI: Index documents from ./data/
    CLI->>Loader: Load all files (PDF, MD, TXT, HTML, DOCX, CSV)
    Loader->>Loader: Recursive scan, format detection
    Loader-->>CLI: List[Document]
    CLI->>Chunker: Apply strategy (recursive/header/parent_child/contextual)
    Chunker->>Chunker: Split text into chunks
    Chunker-->>CLI: List[Document] (chunks with metadata)
    CLI->>Embedder: Generate vectors for each chunk
    Embedder-->>CLI: List[vector]
    CLI->>VectorDB: add_documents(chunks)
    VectorDB-->>CLI: OK
    CLI-->>User: "Indexed N chunks"

%% ═══════════════════════════════════════════════════════════════
%% 2. AGENT TOOL CALL (Entry point — agent gọi RAG)
%% ═══════════════════════════════════════════════════════════════
sequenceDiagram
    title Agent Calls RAG Tool

    participant User
    participant Agent as Loom Agent (LLM)
    participant Tool as rag_query()
    participant Pipeline as run_rag()
    participant Graph as StateGraph

    User->>Agent: "What is the X policy?"
    Agent->>Agent: LLM decides: use rag_query tool
    Agent->>Tool: rag_query(query="What is the X policy?")
    Tool->>Pipeline: run_rag(question, level=ADAPTIVE)
    Pipeline->>Graph: pipeline.invoke({"question": ..., "query_rewrite_count": 0})
    Graph-->>Pipeline: {"generation": ..., "sources": [...]}
    Pipeline-->>Tool: result dict
    Tool-->>Agent: Formatted answer + sources
    Agent-->>User: Answer with citations

%% ═══════════════════════════════════════════════════════════════
%% 3. ADAPTIVE PIPELINE (Level 2 — default)
%% ═══════════════════════════════════════════════════════════════
sequenceDiagram
    title Adaptive RAG Pipeline (Level 2)

    participant Start as START
    participant Route as route_question()
    participant Direct as direct_answer()
    participant Retrieve as retrieve()
    participant VectorDB as Vector Store
    participant Rerank as rerank_documents()
    participant Grade as grade_documents()
    participant Generate as generate()
    participant Rewrite as rewrite_query()
    participant End as END

    Start->>Route: state.question
    alt General knowledge question
        Route-->>Start: RouteDecision.DIRECT_ANSWER
        Start->>Direct: state.question
        Direct->>Direct: LLM answers without context
        Direct-->>End: {"generation": ..., "sources": []}
    else Knowledge base question
        Route-->>Start: RouteDecision.RETRIEVE
        Start->>Retrieve: state.question
        Retrieve->>VectorDB: vector search (top-k)
        VectorDB-->>Retrieve: List[Document]
        Retrieve->>Rerank: rerank(query, docs)
        Rerank-->>Retrieve: reranked docs
        Retrieve-->>Start: {"documents": [...], "question": ...}
        Start->>Grade: state.documents
        alt Documents are relevant
            Grade-->>Start: GradeDecision.GENERATE
            Start->>Generate: state.documents + state.question
            Generate->>Generate: LLM generates answer from context
            Generate-->>End: {"generation": ..., "sources": [...]}
        else Documents not relevant
            Grade-->>Start: GradeDecision.REWRITE
            Start->>Rewrite: state (question + bad docs)
            Rewrite->>Rewrite: LLM rewrites query
            Rewrite-->>Start: {"question": new_query, "query_rewrite_count": +1}
            Note over Start,Rewrite: Loop back to retrieve (max 2 times)
        end
    end

%% ═══════════════════════════════════════════════════════════════
%% 4. SELF-RAG PIPELINE (Level 3 — self-correcting)
%% ═══════════════════════════════════════════════════════════════
sequenceDiagram
    title Self-RAG Pipeline (Level 3)

    participant Start as START
    participant Route as route_question()
    participant Retrieve as retrieve()
    participant Grade as grade_documents()
    participant Generate as generate()
    participant Halluc as check_hallucination()
    participant Quality as check_answer_quality()
    participant Rewrite as rewrite_query()
    participant End as END

    Start->>Route: state.question
    Route-->>Start: RouteDecision.RETRIEVE
    Start->>Retrieve: state.question
    Retrieve-->>Start: {"documents": [...]}
    Start->>Grade: state.documents
    Grade-->>Start: GradeDecision.GENERATE
    Start->>Generate: state.documents + question
    Generate-->>Start: {"generation": ..., "sources": [...]}

    Start->>Halluc: check if answer is grounded
    alt Answer has hallucination
        Halluc-->>Start: GradeDecision.REWRITE
        Start->>Rewrite: rewrite query
        Rewrite-->>Start: loop back to retrieve
    else Answer is grounded
        Halluc-->>Start: "check_quality"
        Start->>Quality: check if answer is useful
        alt Answer is useful
            Quality-->>End: QualityCheckDecision.FINISH
        else Answer is not useful
            Quality-->>Start: QualityCheckDecision.REWRITE
            Start->>Rewrite: rewrite query
            Rewrite-->>Start: loop back to retrieve
        end
    end

%% ═══════════════════════════════════════════════════════════════
%% 5. FULL SYSTEM (Agent + RAG + Middleware)
%% ═══════════════════════════════════════════════════════════════
sequenceDiagram
    title Full System: User → Agent → Middleware → RAG → Response

    participant User
    participant Middleware as Middleware Stack
    participant Agent as Deep Agent (LLM)
    participant RAG_Tool as rag_query()
    participant RAG_Pipeline as RAG StateGraph
    participant VectorDB as Chroma/Qdrant
    participant LLM as LLM (Qwen)

    User->>Middleware: "Tell me about X"
    Middleware->>Middleware: Safety check (budget guard)
    Middleware->>Middleware: Log session start
    Middleware->>Middleware: Inject context (user, date)
    Middleware->>Agent: Processed input
    Agent->>Agent: LLM reasoning: "I should use rag_query"
    Agent->>RAG_Tool: rag_query("Tell me about X")
    RAG_Tool->>RAG_Pipeline: run_rag(question, ADAPTIVE)
    RAG_Pipeline->>LLM: route_question (LLM call)
    LLM-->>RAG_Pipeline: "retrieve"
    RAG_Pipeline->>VectorDB: vector search
    VectorDB-->>RAG_Pipeline: top-k docs
    RAG_Pipeline->>LLM: grade_documents (LLM call)
    LLM-->>RAG_Pipeline: "generate"
    RAG_Pipeline->>LLM: generate answer (LLM call)
    LLM-->>RAG_Pipeline: answer text
    RAG_Pipeline-->>RAG_Tool: {"generation": ..., "sources": [...]}
    RAG_Tool-->>Agent: Formatted result
    Agent->>Agent: LLM composes final response
    Agent-->>Middleware: Agent response
    Middleware->>Middleware: Sanitize output (unicode fix)
    Middleware->>Middleware: Log session end
    Middleware-->>User: Final answer with sources
