"""Embedding generation — Google Gemini (cloud) or Ollama (local).

Provider is selected by ``settings.provider``:
  - "gemini"  → GeminiEmbedder  (text-embedding-004, 768-dim, free tier)
  - "ollama"  → OllamaEmbedder  (nomic-embed-text, requires local Ollama)

Call ``get_embedder()`` to obtain the correct instance automatically.
"""

from __future__ import annotations

from loguru import logger

from documind.config import settings
from documind.ingestion.chunker import Chunk

# ── Gemini Embedder ──────────────────────────────────────────────────────────

class GeminiEmbedder:
    """Embed text using Google Gemini Embedding API via the Generative AI SDK.

    Free tier: 1 500 requests/day, up to 100 texts per batch call.
    Embedding dimension is detected dynamically from the configured model
    (e.g. gemini-embedding-001 → 3072-dim).
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        import google.generativeai as genai  # lazy import — not needed for Ollama path

        _key = api_key or settings.gemini_api_key
        if not _key:
            raise ValueError(
                "GEMINI_API_KEY is required when provider='gemini'. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        genai.configure(api_key=_key)
        self._genai = genai
        self.model = model or settings.gemini_embed_model
        self._embedding_dim: int | None = None
        logger.info(f"GeminiEmbedder ready (model={self.model})")

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            sample = self.embed_text("test")
            self._embedding_dim = len(sample)
        return self._embedding_dim

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string (used for query embedding at query time)."""
        result = self._genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]

    def embed_texts(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Embed multiple texts, batched for efficiency.

        Gemini supports up to 100 texts per call — we stay at that limit.
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self._genai.embed_content(
                model=self.model,
                content=batch,
                task_type="retrieval_document",
            )
            # When content is a list, result["embedding"] is a list-of-lists
            embeddings = result["embedding"]
            if isinstance(embeddings[0], float):
                # Single-item batch returns a flat list — wrap it
                embeddings = [embeddings]
            all_embeddings.extend(embeddings)

            logger.debug(
                f"Gemini embedded batch {i // batch_size + 1}/"
                f"{(len(texts) + batch_size - 1) // batch_size}"
            )

        return all_embeddings

    def embed_chunks(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        """Embed a list of Chunk objects, returning (chunk, embedding) pairs."""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)
        logger.info(f"Gemini embedded {len(chunks)} chunks (dim={self._embedding_dim})")
        return list(zip(chunks, embeddings))

    def ensure_model_available(self) -> bool:
        """Gemini models are always available if the API key is valid."""
        return bool(settings.gemini_api_key)


# ── Ollama Embedder ──────────────────────────────────────────────────────────

class OllamaEmbedder:
    """Embed text using a locally-running Ollama server.

    Requires: ollama server running + model pulled (e.g. nomic-embed-text).
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ):
        from ollama import Client  # lazy import — not needed for Gemini path

        self.model = model or settings.ollama_embed_model
        self.base_url = base_url or settings.ollama_base_url
        self.client = Client(host=self.base_url)
        self._embedding_dim: int | None = None
        logger.info(f"OllamaEmbedder ready (model={self.model}, url={self.base_url})")

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            test = self.embed_text("test")
            self._embedding_dim = len(test)
        return self._embedding_dim

    def embed_text(self, text: str, is_query: bool = True) -> list[float]:
        """Embed a single text string.

        Appends the prefix required by nomic-embed-text:
        - 'search_query: ' for search queries at query time
        - 'search_document: ' for document chunks at index time
        """
        if self.model.startswith("nomic-embed-text"):
            prefix = "search_query: " if is_query else "search_document: "
            if not text.startswith("search_query:") and not text.startswith("search_document:"):
                text = prefix + text

        response = self.client.embed(model=self.model, input=text)
        return response.embeddings[0]

    def embed_texts(self, texts: list[str], batch_size: int = 32, is_query: bool = False) -> list[list[float]]:
        """Embed multiple texts in batches with nomic prefix formatting."""
        all_embeddings: list[list[float]] = []
        prefix = "search_query: " if is_query else "search_document: "

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if self.model.startswith("nomic-embed-text"):
                formatted_batch = [
                    (prefix + t) if not t.startswith("search_query:") and not t.startswith("search_document:") else t
                    for t in batch
                ]
            else:
                formatted_batch = batch

            response = self.client.embed(model=self.model, input=formatted_batch)
            all_embeddings.extend(response.embeddings)
            logger.debug(
                f"Ollama embedded batch {i // batch_size + 1}/"
                f"{(len(texts) + batch_size - 1) // batch_size}"
            )
        return all_embeddings

    def embed_chunks(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts, is_query=False)
        logger.info(f"Ollama embedded {len(chunks)} chunks (dim={len(embeddings[0])})")
        return list(zip(chunks, embeddings))

    def ensure_model_available(self) -> bool:
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models]
            if any(self.model in name for name in model_names):
                logger.info(f"Embedding model '{self.model}' is available")
                return True
            logger.warning(
                f"Embedding model '{self.model}' not found. "
                f"Run: ollama pull {self.model}"
            )
            return False
        except Exception as e:
            logger.error(f"Cannot connect to Ollama at {self.base_url}: {e}")
            return False


# ── Type alias used by the rest of the codebase ─────────────────────────────

# Both classes share the same interface; use this union as a type hint.
Embedder = GeminiEmbedder | OllamaEmbedder


def get_embedder() -> Embedder:
    """Factory: return the correct Embedder for the configured provider."""
    if settings.using_gemini:
        return GeminiEmbedder()
    return OllamaEmbedder()

