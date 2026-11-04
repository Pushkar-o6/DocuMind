"""Typed state schema for the LangGraph orchestration pipeline.

All nodes in the graph read from and write to this shared state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineState:
    """Shared state passed between LangGraph nodes.

    Each node reads what it needs and writes its outputs here.
    """

    # ── Input ──
    question: str = ""
    source_file: str | None = None  # Optional: restrict to specific document
    transformed_question: str = ""

    # ── Classification ──
    query_type: str = ""  # "text", "visual", "mixed"

    # ── Retrieval ──
    retrieved_results: list[Any] = field(default_factory=list)  # HybridResult list
    reranked_results: list[Any] = field(default_factory=list)  # After reranking

    # ── Generation ──
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    context_used: str = ""

    # ── Grounding ──
    is_grounded: bool = False
    grounding_confidence: float = 0.0
    grounding_details: dict = field(default_factory=dict)

    # ── Retry state ──
    retry_count: int = 0
    max_retries: int = 2
    reformulated_question: str = ""

    # ── Final output ──
    final_answer: str = ""
    final_citations: list[dict] = field(default_factory=list)
    final_confidence: float = 0.0
    abstained: bool = False  # True if system couldn't ground an answer

    # ── Error handling ──
    error: str | None = None
