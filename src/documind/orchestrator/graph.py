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
        "error": result.get("error"),
    }
