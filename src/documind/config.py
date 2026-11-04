"""Centralized configuration via pydantic-settings.

All settings are read from environment variables or a `.env` file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from env vars / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Provider selection ───────────────────────────────
    # "ollama"  → Local Ollama server (optimized for RTX 3050 GPU, 100% private, free)
    # "gemini"  → Google Gemini API (cloud fallback)
    provider: str = "ollama"

    # ── Google Gemini ────────────────────────────────────
    gemini_api_key: str = ""
    gemini_embed_model: str = "models/gemini-embedding-001"  # active Gemini embedding model
    gemini_llm_model: str = "gemini-flash-lite-latest"     # active fast Gemini Flash model

    # ── Ollama (Local GPU accelerated via RTX 3050) ───────
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_llm_model: str = "llama3.2:3b"
    ollama_vlm_model: str = "moondream"

    # ── Qdrant ──────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "documind"

    # ── Reranker ────────────────────────────────────────
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_enabled: bool = True  # Enabled for 2-stage precision retrieval
    reranker_min_score: float = -4.0 # Score floor cutoff for abstention (raw cross-encoder logit)
    reranker_device: str = "cpu"    # Keep reranker on CPU to guarantee full 4GB VRAM for Ollama LLM

    # ── OCR / Vision ────────────────────────────────────
    # "tesseract" → Local OCR (default when offline)
    # "gemini"    → Cloud vision OCR
    ocr_provider: str = "tesseract"

    # ── Ingestion & Chunking ─────────────────────────────
    # Strategy options: "structure_aware" | "recursive" | "semantic"
    chunking_strategy: str = "structure_aware"
    chunk_size: int = 512       # ~256-512 for factoid, 512-1024 for analytical
    chunk_overlap: int = 64     # 10-20% overlap
    chunk_min_tokens: int = 150 # Size floor to avoid tiny fragments
    chunk_query_type: str = "factoid" # "factoid" (256-512) or "analytical" (512-1024)
    max_pages: int = 500

    # ── API ─────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Paths ───────────────────────────────────────────
    upload_dir: Path = Path("./data/uploads")

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @property
    def using_gemini(self) -> bool:
        """True when Gemini is the active provider."""
        return self.provider.lower() == "gemini"

    @property
    def using_ollama(self) -> bool:
        """True when Ollama is the active provider."""
        return self.provider.lower() == "ollama"


# Singleton instance — import this everywhere
settings = Settings()

