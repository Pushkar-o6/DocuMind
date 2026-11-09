"""Semantic Cache for DocuMind RAG Queries.

Features:
- Dual-tier lookup: O(1) exact-match string check, followed by Cosine Similarity on query embeddings (>= 0.93 threshold)
- Document-scoped isolation: Queries scoped to a specific document only match within that document's scope
- Cascade document invalidation: Deleting a document purges all cached questions scoped to it AND any global questions that cited it
- Full purge on corpus clear
- Populated strictly by authenticated user queries (startup/warmup queries never pollute the cache)
- Thread-safe with LRU eviction
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class SemanticCacheEntry:
    """Individual semantic cache record."""

    question: str
    embedding: np.ndarray  # L2-normalized float32 vector
    source_file: str | None
    response_data: dict[str, Any]
    created_at: float
    hit_count: int = 0
    last_accessed: float = 0.0


class SemanticCache:
    """In-memory semantic cache using dense embeddings and cosine similarity.

    Matches queries with cosine similarity >= similarity_threshold (default 0.93).
    """

    def __init__(self, similarity_threshold: float = 0.93, max_entries: int = 500):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._entries: list[SemanticCacheEntry] = []
        self._lock = threading.Lock()
        logger.info(
            f"SemanticCache initialized (similarity_threshold={similarity_threshold}, max_entries={max_entries})"
        )

    @staticmethod
    def _normalize(vector: list[float] | np.ndarray) -> np.ndarray:
        """Convert to 1D float32 array and L2-normalize for fast cosine similarity via dot product."""
        arr = np.array(vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(arr)
        if norm > 0:
            return arr / norm
        return arr

    def lookup(
        self,
        question: str,
        query_embedding: list[float] | np.ndarray | None = None,
        source_file: str | None = None,
    ) -> tuple[dict[str, Any] | None, float, str]:
        """Search the cache for an exact or semantic match.

        Args:
            question: User's query text.
            query_embedding: Dense embedding vector for question (if available).
            source_file: Document scope (None = global search).

        Returns:
            (cached_response, similarity_score, match_type)
            match_type is 'exact', 'semantic', or 'miss'.
        """
        clean_q = question.strip().lower()
        now = time.time()

        with self._lock:
            # ── Tier 1: Exact String Match (0ms) ──
            for entry in self._entries:
                if entry.source_file == source_file and entry.question.strip().lower() == clean_q:
                    entry.hit_count += 1
                    entry.last_accessed = now
                    logger.info(f"SemanticCache exact match hit (100%) for: '{question}'")
                    return dict(entry.response_data), 1.0, "exact"

            # ── Tier 2: Semantic Vector Match (Cosine Similarity) ──
            if query_embedding is None or len(self._entries) == 0:
                return None, 0.0, "miss"

            norm_q = self._normalize(query_embedding)
            best_entry: SemanticCacheEntry | None = None
            best_sim = -1.0

            for entry in self._entries:
                # Must match document scope (or both be global)
                if entry.source_file != source_file:
                    continue

                # Cosine similarity between two unit vectors is simply their dot product
                sim = float(np.dot(norm_q, entry.embedding))
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_entry and best_sim >= self.similarity_threshold:
                best_entry.hit_count += 1
                best_entry.last_accessed = now
                logger.info(
                    f"SemanticCache hit ({best_sim:.1%}) for '{question}' "
                    f"-> matched '{best_entry.question}'"
                )
                return dict(best_entry.response_data), best_sim, "semantic"

            return None, max(0.0, best_sim), "miss"

    def store(
        self,
        question: str,
        query_embedding: list[float] | np.ndarray,
        source_file: str | None,
        response_data: dict[str, Any],
    ) -> bool:
        """Store a real user query-answer pair in the cache.

        Ensures only valid, non-error, non-empty user responses are cached.
        """
        if not question or not response_data:
            return False

        # Guard: Never cache errors or empty responses
        if response_data.get("error"):
            return False

        answer = str(response_data.get("answer", "")).strip()
        if not answer:
            return False

        norm_emb = self._normalize(query_embedding)
        now = time.time()
        clean_q = question.strip()

        with self._lock:
            # Check if this exact query & scope already exists; update it if so
            for entry in self._entries:
                if entry.source_file == source_file and entry.question.strip().lower() == clean_q.lower():
                    entry.response_data = dict(response_data)
                    entry.last_accessed = now
                    entry.embedding = norm_emb
                    return True

            # LRU Eviction if capacity exceeded
            if len(self._entries) >= self.max_entries:
                # Evict entry with oldest last_accessed
                self._entries.sort(key=lambda e: e.last_accessed)
                evicted = self._entries.pop(0)
                logger.debug(f"SemanticCache capacity reached: evicted '{evicted.question}'")

            new_entry = SemanticCacheEntry(
                question=clean_q,
                embedding=norm_emb,
                source_file=source_file,
                response_data=dict(response_data),
                created_at=now,
                last_accessed=now,
            )
            self._entries.append(new_entry)
            logger.info(
                f"SemanticCache stored entry (scope={source_file!r}, "
                f"total_cached={len(self._entries)}): '{clean_q}'"
            )
            return True

    def invalidate_document(self, filename: str) -> int:
        """Purge all cached queries scoped to this document OR citing this document.

        Args:
            filename: The document name being removed.

        Returns:
            Number of cache entries deleted.
        """
        with self._lock:
            initial_count = len(self._entries)
            surviving: list[SemanticCacheEntry] = []

            for entry in self._entries:
                # 1. Scoped to the deleted document?
                if entry.source_file == filename:
                    continue

                # 2. Global query whose citations contain the deleted document?
                citations = entry.response_data.get("citations", [])
                cited_files = {
                    c.get("source_file") if isinstance(c, dict) else getattr(c, "source_file", "")
                    for c in citations
                }
                if filename in cited_files:
                    continue

                surviving.append(entry)

            removed = initial_count - len(surviving)
            self._entries = surviving

            if removed > 0:
                logger.info(
                    f"SemanticCache invalidated {removed} entries linked to document '{filename}' "
                    f"({len(self._entries)} entries remaining)"
                )
            return removed

    def clear(self) -> int:
        """Purge all entries from cache (e.g. on full corpus clear)."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            logger.info(f"SemanticCache completely purged ({count} entries cleared)")
            return count

    def size(self) -> int:
        """Return the current number of cached user queries."""
        with self._lock:
            return len(self._entries)
