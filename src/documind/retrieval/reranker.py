"""Cross-encoder reranker using BGE-Reranker.

Reranks the top-k candidates from hybrid retrieval using a
cross-encoder model for much higher precision in the final ranking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    pass

from documind.config import settings
from documind.retrieval.hybrid import HybridResult


class Reranker:
    """Cross-encoder reranker using BGE-Reranker-Base.

    Runs on CPU by default to preserve the RTX 3050's 4GB VRAM for Ollama LLM inference.
    Includes score threshold filtering to guard against hallucination on irrelevant queries.
    """

    def __init__(self, model_name: str | None = None, min_score: float | None = None):
        self.model_name = model_name or settings.reranker_model
        self.min_score = min_score if min_score is not None else getattr(settings, "reranker_min_score", -4.0)
        self.device = getattr(settings, "reranker_device", "cpu")
        self._model = None

    def _load_model(self):
        """Lazy-load the cross-encoder model on designated device (CPU to conserve VRAM)."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker model: {self.model_name} on device: {self.device}")
            self._model = CrossEncoder(self.model_name, max_length=512, device=self.device)
            logger.info("Reranker model loaded successfully")
        return self._model

    def rerank(
        self,
        query: str,
        results: list[HybridResult],
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[HybridResult]:
        """Rerank hybrid results using the cross-encoder.

        Args:
            query: The original search query.
            results: Candidates from hybrid retrieval.
            top_k: Number of results to keep after reranking.
            score_threshold: Minimum cross-encoder score cutoff.

        Returns:
            Reranked list of HybridResult (filtered by score threshold).
        """
        if not results:
            return []

        model = self._load_model()
        cutoff = score_threshold if score_threshold is not None else self.min_score

        # Create query-document pairs (using raw text without metadata prefixes for pure semantic match)
        pairs = [(query, r.text) for r in results]

        # Score all pairs
        scores = model.predict(pairs)

        # Attach scores and sort
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Apply score threshold filtering (hallucination defense)
        filtered_results = [item for item in scored_results if item[1] >= cutoff]

        if not filtered_results and scored_results:
            logger.warning(
                f"All {len(scored_results)} candidates fell below reranker cutoff ({cutoff:.2f}). "
                f"Top score was {scored_results[0][1]:.2f}. Abstention recommended."
            )
            return []

        if top_k:
            filtered_results = filtered_results[:top_k]

        # Deduplication / MMR-style diversity: remove chunks with >85% word overlap to higher chunks
        deduped: list[HybridResult] = []
        seen_words: list[set[str]] = []

        for result, score in filtered_results:
            words = set(result.text.lower().split())
            is_dup = False
            for prev_words in seen_words:
                overlap = len(words & prev_words) / max(1, len(words | prev_words))
                if overlap > 0.85:
                    is_dup = True
                    break
            if not is_dup:
                result.metadata["reranker_score"] = float(score)
                deduped.append(result)
                seen_words.append(words)

        logger.info(
            f"Reranked {len(results)} → {len(deduped)} candidates. Top score: {scored_results[0][1]:.4f}"
        )
        return deduped
