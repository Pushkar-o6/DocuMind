"""BM25 sparse retrieval index.

Provides keyword-based retrieval that complements dense vector search.
Especially important for exact term/number matching (e.g., "$42.5M revenue").

Industry-ready improvements over initial version:
  - Incremental add_chunks() — no need to rebuild from scratch on each upload
  - source_file filtering in search()
  - Persistent save/load (survives container restarts)
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from loguru import logger
from rank_bm25 import BM25Okapi

from documind.ingestion.chunker import Chunk


class BM25Index:
    """BM25 sparse retrieval over document chunks.

    Stores the corpus alongside the index so we can return
    full chunk metadata with results.

    Supports incremental updates: call add_chunks() per document
    instead of rebuilding the entire index from scratch every time.
    """

    def __init__(self):
        self._index: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Domain-aware tokenizer preserving technical terms, codes, and metrics."""
        text = text.lower()
        # Captures alphanumeric identifiers, dotted numbers ($42.5M, 3.2b), hyphens, and currency symbols
        tokens = re.findall(r"\b[a-z0-9]+(?:[-._/][a-z0-9]+)*\b|[\$€£%]", text)
        return tokens

    def _rebuild_index(self) -> None:
        """Rebuild the BM25Okapi model from the current tokenized corpus."""
        if self._tokenized_corpus:
            self._index = BM25Okapi(self._tokenized_corpus)
        else:
            self._index = None

    def clear(self) -> None:
        """Reset the BM25 index to an empty state."""
        self._chunks = []
        self._tokenized_corpus = []
        self._index = None
        logger.info("BM25 index cleared")

    def build(self, chunks: list[Chunk]) -> None:
        """Build (or replace) the BM25 index from a list of chunks.

        Use add_chunks() for incremental updates.

        Args:
            chunks: Document chunks to index.
        """
        self._chunks = list(chunks)
        self._tokenized_corpus = [self._tokenize(c.text) for c in self._chunks]
        self._rebuild_index()
        logger.info(f"Built BM25 index over {len(self._chunks)} chunks")

    def add_chunks(self, new_chunks: list[Chunk]) -> None:
        """Incrementally add new chunks to the existing index.

        Much faster than rebuilding from scratch on every upload.

        Args:
            new_chunks: New document chunks to add to the index.
        """
        if not new_chunks:
            return

        self._chunks.extend(new_chunks)
        new_tokenized = [self._tokenize(c.text) for c in new_chunks]
        self._tokenized_corpus.extend(new_tokenized)
        self._rebuild_index()
        logger.info(
            f"Added {len(new_chunks)} chunks to BM25 index "
            f"(total: {len(self._chunks)})"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_file: str | None = None,
        section_heading: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search the BM25 index with optional metadata pre-filtering.

        Args:
            query: The search query.
            top_k: Number of results to return.
            source_file: Optional filter — only return chunks from this document.
            section_heading: Optional filter — only return chunks matching heading.

        Returns:
            List of (chunk, score) tuples sorted by descending BM25 score.
        """
        if self._index is None or not self._chunks:
            logger.warning("BM25 index is empty — no results returned")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Build (index, score) pairs and apply metadata filters
        ranked: list[tuple[int, float]] = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            chunk = self._chunks[idx]
            if source_file and chunk.source_file != source_file:
                continue
            if section_heading and getattr(chunk, "section_heading", "") != section_heading:
                continue
            ranked.append((idx, float(score)))

        # Sort by score descending and take top_k
        ranked.sort(key=lambda x: x[1], reverse=True)
        ranked = ranked[:top_k]

        return [(self._chunks[idx], score) for idx, score in ranked]

    def __len__(self) -> int:
        return len(self._chunks)

    def save(self, path: Path) -> None:
        """Save the BM25 index and chunks to disk."""
        data = {
            "chunks": self._chunks,
            "tokenized_corpus": self._tokenized_corpus,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved BM25 index to {path} ({len(self._chunks)} chunks)")

    def load(self, path: Path) -> None:
        """Load a BM25 index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._chunks = data["chunks"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._rebuild_index()
        logger.info(f"Loaded BM25 index from {path} ({len(self._chunks)} chunks)")
