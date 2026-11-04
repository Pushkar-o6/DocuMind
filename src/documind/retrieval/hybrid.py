"""Hybrid retrieval – fuses dense (vector) and sparse (BM25) results.

Uses Reciprocal Rank Fusion (RRF) to combine rankings from both
retrieval methods into a single ranked list.

Industry-ready improvement: source_file filter is now passed through
to BM25 search so cross-document contamination is prevented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from documind.ingestion.embedder import Embedder
from documind.retrieval.sparse import BM25Index
from documind.vectorstore.qdrant import QdrantStore


@dataclass
class HybridResult:
    """A fused result from hybrid retrieval."""

    chunk_id: str
    text: str
    page_number: int
    block_type: str
    source_file: str
    section_heading: str = ""
    parent_text: str = ""
    bbox: tuple[float, float, float, float] | None = None
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    metadata: dict = field(default_factory=dict)


class HybridRetriever:
    """Combines dense (Qdrant) and sparse (BM25) retrieval using RRF.

    Reciprocal Rank Fusion (RRF) formula:
        score(d) = Σ 1 / (k + rank_i(d))
    where k is a constant (default 60) and rank_i is the rank from system i.
    """

    def __init__(
        self,
        qdrant_store: QdrantStore,
        bm25_index: BM25Index,
        embedder: Embedder,
        rrf_k: int = 60,
    ):
        self.qdrant = qdrant_store
        self.bm25 = bm25_index
        self.embedder = embedder
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 15,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        source_file: str | None = None,
        section_heading: str | None = None,
    ) -> list[HybridResult]:
        """Run hybrid retrieval with Reciprocal Rank Fusion (RRF) and metadata pre-filtering.

        Args:
            query: The search query.
            top_k: Number of candidates to pass to the next stage (reranker).
            dense_top_k: Initial candidates from dense vector retrieval.
            sparse_top_k: Initial candidates from sparse lexical retrieval.
            dense_weight: Weight for dense RRF ranking.
            sparse_weight: Weight for sparse RRF ranking.
            source_file: Optional hard filter by document name.
            section_heading: Optional hard filter by section heading.

        Returns:
            List of HybridResult sorted by fused RRF score.
        """
        # ── Dense retrieval (Qdrant) ──
        query_embedding = self.embedder.embed_text(query)
        dense_results = self.qdrant.search(
            query_embedding=query_embedding,
            top_k=dense_top_k,
            source_file=source_file,
            section_heading=section_heading,
        )

        # ── Sparse retrieval (BM25) ──
        sparse_results = self.bm25.search(
            query,
            top_k=sparse_top_k,
            source_file=source_file,
            section_heading=section_heading,
        )

        # ── RRF Fusion ──
        fused: dict[str, HybridResult] = {}

        # Dense candidates
        for rank, result in enumerate(dense_results):
            rrf_score = dense_weight / (self.rrf_k + rank + 1)
            fused[result.chunk_id] = HybridResult(
                chunk_id=result.chunk_id,
                text=result.text,
                page_number=result.page_number,
                block_type=result.block_type,
                source_file=result.source_file,
                section_heading=getattr(result, "section_heading", ""),
                parent_text=getattr(result, "parent_text", ""),
                bbox=result.bbox,
                dense_score=result.score,
                rrf_score=rrf_score,
                metadata=result.metadata,
            )

        # Merge sparse candidates
        for rank, (chunk, bm25_score) in enumerate(sparse_results):
            rrf_score = sparse_weight / (self.rrf_k + rank + 1)
            if chunk.chunk_id in fused:
                fused[chunk.chunk_id].sparse_score = bm25_score
                fused[chunk.chunk_id].rrf_score += rrf_score
            else:
                fused[chunk.chunk_id] = HybridResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    block_type=chunk.block_type,
                    source_file=chunk.source_file,
                    section_heading=getattr(chunk, "section_heading", ""),
                    parent_text=getattr(chunk, "parent_text", ""),
                    bbox=chunk.bbox,
                    sparse_score=bm25_score,
                    rrf_score=rrf_score,
                    metadata=getattr(chunk, "metadata", {}),
                )

        sorted_results = sorted(
            fused.values(), key=lambda r: r.rrf_score, reverse=True
        )[:top_k]

        logger.info(
            f"Hybrid search: dense={len(dense_results)}, "
            f"sparse={len(sparse_results)}, fused={len(sorted_results)}"
            + (f" [source: {source_file}]" if source_file else "")
            + (f" [section: {section_heading}]" if section_heading else "")
        )

        return sorted_results
