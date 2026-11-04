"""LangGraph StateGraph definition for the DocuMind query pipeline.

Wires together all pipeline nodes with conditional edges for the
verify → retry loop. The graph handles:
  classify → retrieve → rerank → generate → verify
    ├─ grounded → finalize
    └─ not grounded → reformulate → retrieve (max 2 retries)
        └─ still not grounded → abstain
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from loguru import logger

from documind.generation.grounding import GroundingChecker, get_grounding_checker
from documind.generation.llm import LLMGenerator, get_llm_generator
from documind.ingestion.embedder import Embedder
from documind.orchestrator.nodes import PipelineNodes
from documind.retrieval.hybrid import HybridRetriever
from documind.retrieval.reranker import Reranker
from documind.retrieval.sparse import BM25Index
from documind.vectorstore.qdrant import QdrantStore


def should_retry(state: dict) -> str:
    """Conditional edge: decide whether to retry or finalize.

    Returns:
        "reformulate" if not grounded and retries remain.
        "finalize" if grounded or retries exhausted.
    """
    if state.get("is_grounded") or state.get("abstained"):
        return "finalize"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if retry_count < max_retries:
        logger.info(f"Not grounded — retrying ({retry_count + 1}/{max_retries})")
        return "reformulate"
    else:
        logger.warning("Not grounded — max retries reached, abstaining")
        return "finalize"


def build_pipeline(
    qdrant_store: QdrantStore,
    bm25_index: BM25Index,
    embedder: Embedder,
    reranker: Reranker | None = None,
    generator: LLMGenerator | None = None,
    grounding_checker: GroundingChecker | None = None,
) -> StateGraph:
    """Build the LangGraph pipeline.

    Args:
        qdrant_store: Initialized QdrantStore.
        bm25_index: Built BM25Index.
        embedder: Initialized Embedder.
        reranker: Reranker instance (created if None).
        generator: LLMGenerator instance (created if None).
        grounding_checker: GroundingChecker instance (created if None).

    Returns:
        A compiled LangGraph StateGraph ready to invoke.
    """
    # Initialize components with defaults
    retriever = HybridRetriever(
        qdrant_store=qdrant_store,
        bm25_index=bm25_index,
        embedder=embedder,
    )
    reranker = reranker or Reranker()
    generator = generator or get_llm_generator()
    grounding_checker = grounding_checker or get_grounding_checker()

    # Create node functions
    nodes = PipelineNodes(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        grounding_checker=grounding_checker,
    )

    # Build the graph
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("classify", nodes.classify_query)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("rerank", nodes.rerank)
    graph.add_node("generate", nodes.generate)
    graph.add_node("verify", nodes.verify_grounding)
    graph.add_node("reformulate", nodes.reformulate)
    graph.add_node("finalize", nodes.finalize)

    # Add edges (linear flow with a conditional retry loop)
    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "verify")

    # Conditional: retry or finalize
    graph.add_conditional_edges(
        "verify",
        should_retry,
        {
            "reformulate": "reformulate",
            "finalize": "finalize",
        },
    )

    # After reformulation, go back to retrieve
    graph.add_edge("reformulate", "retrieve")

    # Finalize ends the graph
    graph.add_edge("finalize", END)

    compiled = graph.compile()
    compiled._documind_nodes = nodes
    logger.info("LangGraph pipeline compiled successfully")

    return compiled


def run_query(
    pipeline,
    question: str,
    source_file: str | None = None,
) -> dict:
    """Run a query through the pipeline.

    Args:
        pipeline: Compiled LangGraph pipeline.
        question: The user's question.
        source_file: Optional: restrict search to a specific document.

    Returns:
        Dict with final_answer, final_citations, final_confidence, abstained.
    """
    initial_state = {
        "question": question,
        "source_file": source_file,
        "transformed_question": "",
        "query_type": "",
        "retrieved_results": [],
        "reranked_results": [],
        "answer": "",
        "citations": [],
        "context_used": "",
        "is_grounded": False,
        "grounding_confidence": 0.0,
        "grounding_details": {},
        "retry_count": 0,
        "max_retries": 2,
        "reformulated_question": "",
        "final_answer": "",
        "final_citations": [],
        "final_confidence": 0.0,
        "abstained": False,
        "stage_timings": {},
        "error": None,
    }

    result = pipeline.invoke(initial_state)

    return {
        "answer": result.get("final_answer", ""),
        "citations": result.get("final_citations", []),
        "confidence": result.get("final_confidence", 0.0),
        "abstained": result.get("abstained", False),
        "query_type": result.get("query_type", ""),
        "retry_count": result.get("retry_count", 0),
        "stage_timings": result.get("stage_timings", {}),
        "latency_s": result.get("latency_s", 0.0),
        "error": result.get("error"),
    }


def run_query_stream(
    pipeline,
    question: str,
    source_file: str | None = None,
):
    """Run pipeline query with real-time SSE streaming events.

    Yields:
        {"event": "stage", "stage": "classifying", "message": "Analyzing & expanding query..."}
        {"event": "stage", "stage": "retrieving", "message": "Executing parallel hybrid search (dense + sparse)..."}
        {"event": "stage", "stage": "reranking", "message": "Cross-encoder reranking candidates..."}
        {"event": "stage", "stage": "generating", "message": "Synthesizing answer from verified evidence..."}
        {"event": "token", "token": "..."}
        {"event": "final", "data": {...}}
    """
    nodes: PipelineNodes = getattr(pipeline, "_documind_nodes", None)
    if not nodes:
        # Fallback to non-streaming invoke
        res = run_query(pipeline, question, source_file)
        yield {"event": "token", "token": res["answer"]}
        yield {"event": "final", "data": res}
        return

    import time
    t_start = time.perf_counter()
    state = {
        "question": question,
        "source_file": source_file,
        "transformed_question": "",
        "query_type": "",
        "retrieved_results": [],
        "reranked_results": [],
        "answer": "",
        "citations": [],
        "context_used": "",
        "is_grounded": False,
        "grounding_confidence": 0.0,
        "grounding_details": {},
        "retry_count": 0,
        "max_retries": 2,
        "reformulated_question": "",
        "final_answer": "",
        "final_citations": [],
        "final_confidence": 0.0,
        "abstained": False,
        "stage_timings": {},
        "error": None,
    }

    # 1. Classify / Transform
    yield {"event": "stage", "stage": "classifying", "message": "Analyzing & expanding query..."}
    state = nodes.classify_query(state)

    # 2. Retrieve (Hybrid Dense + Lexical)
    yield {"event": "stage", "stage": "retrieving", "message": "Executing parallel hybrid search (Qdrant + BM25)..."}
    state = nodes.retrieve(state)

    # 3. Cross-Encoder Reranking
    num_candidates = len(state.get("retrieved_results", []))
    yield {
        "event": "stage",
        "stage": "reranking",
        "message": f"Cross-encoder reranking {num_candidates} candidates for high precision...",
    }
    state = nodes.rerank(state)

    reranked = state.get("reranked_results", [])
    if not reranked:
        state["answer"] = "I don't have enough information to answer this question."
        state["abstained"] = True
        nodes.finalize(state)
        yield {"event": "token", "token": state["answer"]}
        total_latency = round(time.perf_counter() - t_start, 3)
        state["latency_s"] = total_latency
        state["stage_timings"]["total_s"] = total_latency
        yield {
            "event": "final",
            "data": {
                "answer": state["final_answer"],
                "citations": state["final_citations"],
                "confidence": state["final_confidence"],
                "abstained": state["abstained"],
                "query_type": state["query_type"],
                "retry_count": 0,
                "stage_timings": state["stage_timings"],
                "latency_s": total_latency,
                "error": None,
            },
        }
        return

    # 4. Synthesizing Answer with live token streaming
    yield {
        "event": "stage",
        "stage": "generating",
        "message": f"Synthesizing answer from top {len(reranked)} grounded chunks...",
    }
    t_gen = time.perf_counter()
    full_answer_text = ""
    citations = []
    context_used = ""

    if hasattr(nodes.generator, "generate_answer_stream"):
        for item in nodes.generator.generate_answer_stream(question=question, results=reranked):
            if item.get("type") == "token":
                yield {"event": "token", "token": item["token"]}
            elif item.get("type") == "final":
                full_answer_text = item["answer"]
                citations = item["citations"]
                context_used = item["context_used"]
    else:
        out = nodes.generator.generate_answer(question=question, results=reranked)
        full_answer_text = out["answer"]
        citations = out["citations"]
        context_used = out["context_used"]
        yield {"event": "token", "token": full_answer_text}

    state.setdefault("stage_timings", {})["generate_s"] = round(time.perf_counter() - t_gen, 3)
    state["answer"] = full_answer_text
    state["citations"] = citations
    state["context_used"] = context_used

    # 5. Verify & Finalize
    yield {"event": "stage", "stage": "verifying", "message": "Verifying grounding & citations..."}
    state = nodes.verify_grounding(state)
    state = nodes.finalize(state)

    total_latency = round(time.perf_counter() - t_start, 3)
    state["latency_s"] = total_latency
    state["stage_timings"]["total_s"] = total_latency

    yield {
        "event": "final",
        "data": {
            "answer": state["final_answer"],
            "citations": state["final_citations"],
            "confidence": state["final_confidence"],
            "abstained": state["abstained"],
            "query_type": state["query_type"],
            "retry_count": state.get("retry_count", 0),
            "stage_timings": state["stage_timings"],
            "latency_s": total_latency,
            "error": state.get("error"),
        },
    }
