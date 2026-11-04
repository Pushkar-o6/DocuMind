"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Request Schemas ──


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(..., description="The question to ask about uploaded documents")
    source_file: str | None = Field(
        None, description="Optional: restrict search to a specific document filename"
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of results to consider")


# ── Response Schemas ──


class Citation(BaseModel):
    """A single citation reference."""

    page: int
    block_type: str
    source_file: str
    snippet: str
    section_heading: str = ""
    parent_text: str = ""
    bbox: list[float] | None = None


class UploadResponse(BaseModel):
    """Response after successful document upload and ingestion."""

    filename: str
    total_pages: int
    total_chunks: int
    skipped: bool = False
    message: str


class QueryResponse(BaseModel):
    """Response from the /query endpoint."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    abstained: bool = False
    query_type: str = ""
    retry_count: int = 0
    request_id: str = ""
    latency_s: float = 0.0
    error: str | None = None


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str
    provider: str = "gemini"
    provider_connected: bool = False
    ollama_connected: bool = False
    qdrant_connected: bool = False
    documents_indexed: int = 0
    collection_info: dict = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    """Metadata about a single indexed document."""

    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Response from GET /api/v1/documents."""

    documents: list[DocumentInfo]
    total: int
