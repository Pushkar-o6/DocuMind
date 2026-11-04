"""Qdrant vector database wrapper.

Manages collection creation, document upsert, and similarity search.
Stores chunk metadata alongside vectors for citation retrieval.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from documind.config import settings
from documind.ingestion.chunker import Chunk


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    chunk_id: str
    text: str
    score: float
    page_number: int
    block_type: str
    source_file: str
    section_heading: str = ""
    parent_text: str = ""
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict = field(default_factory=dict)


class QdrantStore:
    """Qdrant vector store for DocuMind chunks.

    Handles collection creation, upserting embedded chunks,
    and similarity search with optional metadata filtering.
    """

    def __init__(
        self,
        url: str | None = None,
        collection_name: str | None = None,
        api_key: str | None = None,
    ):
        self.url = url or settings.qdrant_url
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.api_key = api_key or getattr(settings, "qdrant_api_key", None) or None
        self._is_embedded = False

        # If pointing to remote cloud (https://), connect directly with api_key
        if self.url and not self.url.startswith("http://localhost") and not self.url.startswith("http://127.0.0.1"):
            self.client = QdrantClient(url=self.url, api_key=self.api_key, check_compatibility=False)
        else:
            # Localhost URL: try connecting to local Docker server first
            try:
                test_client = QdrantClient(url=self.url, timeout=1.0, check_compatibility=False)
                test_client.get_collections()
                self.client = test_client
                logger.info(f"Connected to local Qdrant server at {self.url}")
            except Exception:
                # Docker is not running — fall back to embedded local Qdrant
                from pathlib import Path
                storage_path = Path("./storage/qdrant_db")
                storage_path.mkdir(parents=True, exist_ok=True)

                # Clean up stale .lock files from previous crashed sessions
                lock_file = storage_path / ".lock"
                if lock_file.exists():
                    try:
                        lock_file.unlink()
                        logger.info("Cleaned up stale Qdrant .lock file")
                    except Exception:
                        pass

                try:
                    self.client = QdrantClient(path=str(storage_path))
                    self._is_embedded = True
                    logger.info(
                        "Local Qdrant server unreachable (Docker not running). "
                        "Using embedded local storage at ./storage/qdrant_db (no Docker needed!)"
                    )
                except Exception as lock_err:
                    logger.warning(
                        f"Embedded disk storage locked ({lock_err}). Using in-memory Qdrant storage."
                    )
                    self.client = QdrantClient(":memory:")
                    self._is_embedded = True

    def ensure_collection(self, embedding_dim: int) -> bool:
        """Create the collection if it doesn't exist, or recreate if dimension changed.

        Args:
            embedding_dim: Dimension of the embedding vectors.

        Returns:
            bool: True if collection was created or recreated, False if existing collection was kept.
        """
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            # Verify the existing collection has the correct vector dimension
            try:
                info = self.client.get_collection(self.collection_name)
                existing_dim = None
                vectors = info.config.params.vectors
                if hasattr(vectors, "size"):
                    existing_dim = vectors.size
                elif isinstance(vectors, dict):
                    first = next(iter(vectors.values()), None)
                    if hasattr(first, "size"):
                        existing_dim = first.size

                if existing_dim == embedding_dim:
                    logger.info(
                        f"Collection '{self.collection_name}' already exists "
                        f"(dim={existing_dim})"
                    )
                    return False
                else:
                    logger.warning(
                        f"Collection '{self.collection_name}' has dim={existing_dim} "
                        f"but current embedding model produces dim={embedding_dim}. "
                        f"Recreating collection (old vectors are incompatible)."
                    )
                    self.client.delete_collection(self.collection_name)
            except Exception as e:
                logger.warning(f"Could not verify collection dimension: {e}. Recreating collection.")
                try:
                    self.client.delete_collection(self.collection_name)
                except Exception:
                    pass

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            f"Created Qdrant collection '{self.collection_name}' "
            f"(dim={embedding_dim})"
        )
        return True

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """Upsert embedded chunks into Qdrant.

        Args:
            chunks: List of Chunk objects.
            embeddings: Corresponding embedding vectors.

        Returns:
            Number of points upserted.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                f"count mismatch"
            )

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "block_type": chunk.block_type,
                    "source_file": chunk.source_file,
                    "section_heading": getattr(chunk, "section_heading", ""),
                    "parent_text": getattr(chunk, "parent_text", ""),
                    "token_count_est": getattr(chunk, "token_count_est", 0),
                    "created_at": getattr(chunk, "created_at", ""),
                    "bbox": list(chunk.bbox) if chunk.bbox else None,
                    **chunk.metadata,
                },
            )
            points.append(point)

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info(f"Upserted {len(points)} chunks to '{self.collection_name}'")
        return len(points)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        source_file: str | None = None,
        block_type: str | None = None,
        section_heading: str | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks with metadata pre-filtering.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            source_file: Optional hard filter by source file.
            block_type: Optional filter by block type.
            section_heading: Optional filter by section heading.

        Returns:
            List of SearchResult objects, sorted by descending score.
        """
        # Build hard metadata pre-filters
        conditions = []
        if source_file:
            conditions.append(
                FieldCondition(key="source_file", match=MatchValue(value=source_file))
            )
        if block_type:
            conditions.append(
                FieldCondition(key="block_type", match=MatchValue(value=block_type))
            )
        if section_heading:
            conditions.append(
                FieldCondition(key="section_heading", match=MatchValue(value=section_heading))
            )

        query_filter = Filter(must=conditions) if conditions else None

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        search_results = []
        for point in results.points:
            payload = point.payload or {}
            bbox_raw = payload.get("bbox")
            search_results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", ""),
                    text=payload.get("text", ""),
                    score=point.score,
                    page_number=payload.get("page_number", 0),
                    block_type=payload.get("block_type", ""),
                    source_file=payload.get("source_file", ""),
                    section_heading=payload.get("section_heading", ""),
                    parent_text=payload.get("parent_text", ""),
                    bbox=tuple(bbox_raw) if bbox_raw else None,
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k
                        not in {
                            "chunk_id",
                            "text",
                            "page_number",
                            "block_type",
                            "source_file",
                            "section_heading",
                            "bbox",
                        }
                    },
                )
            )

        return search_results

    def get_collection_info(self) -> dict:
        """Get info about the current collection.

        Returns a dict with collection stats, or a 'not_found' status if
        the collection hasn't been created yet (no documents ingested).
        Never raises — safe to call from the health check at any time.
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": getattr(info, "points_count", 0) or 0,
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0) or 0,
                "status": str(getattr(info, "status", "unknown")),
            }
        except Exception as e:
            err = str(e).lower()
            # Collection simply doesn't exist yet — not an infrastructure error
            if "not found" in err or "404" in err or "doesn't exist" in err:
                return {"name": self.collection_name, "status": "not_created", "points_count": 0}
            return {"error": str(e)}

    def delete_by_source_file(self, source_file: str) -> int:
        """Delete all chunks belonging to a specific source document.

        Args:
            source_file: The filename to delete chunks for.

        Returns:
            Number of points deleted.
        """
        from qdrant_client.models import FilterSelector

        info_before = self.get_collection_info()
        count_before = info_before.get("points_count", 0) or 0

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=source_file),
                        )
                    ]
                )
            ),
        )

        info_after = self.get_collection_info()
        count_after = info_after.get("points_count", 0) or 0
        deleted = count_before - count_after
        logger.info(f"Deleted {deleted} chunks for source_file='{source_file}'")
        return deleted

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        logger.warning(f"Deleted collection '{self.collection_name}'")
