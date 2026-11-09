"""FastAPI application – main entry point and routes.

Industry-ready improvements over initial version:
  - Async ingestion via threadpool (non-blocking event loop)
  - Duplicate document detection by SHA-256 content hash
  - BM25 persistence to disk (survives restarts)
  - Citations properly passed through to API response
  - Per-request tracing (X-Request-ID header)
  - GET /api/v1/documents endpoint
  - Singleton reuse in /health check
  - API versioned under /api/v1/
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from qdrant_client import models

from documind.api.schemas import (
    Citation,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from documind.config import settings
from documind.ingestion.chunker import DocumentChunker
from documind.ingestion.embedder import Embedder, get_embedder
from documind.ingestion.ocr import ocr_pdf_pages
from documind.ingestion.parser import PDFParser
from documind.orchestrator.graph import build_pipeline, run_query, run_query_stream
from documind.retrieval.cache import SemanticCache
from documind.retrieval.reranker import Reranker
from documind.retrieval.sparse import BM25Index
from documind.vectorstore.qdrant import QdrantStore

# ── Thread pool for CPU-bound ingestion work ──
_executor = ThreadPoolExecutor(max_workers=4)

# ── BM25 index persistence path ──
BM25_INDEX_PATH = Path("./storage/bm25_index.pkl")

# ── Content-hash registry to detect duplicate uploads ──
# Maps SHA-256 hex → filename; persisted to disk
HASH_REGISTRY_PATH = Path("./storage/doc_hashes.json")

# ── Singleton component cache ──
_components: dict = {}

# ── Dynamic Semantic Query Cache (vectorized similarity >= 0.92, document-scoped) ──
semantic_cache = SemanticCache(similarity_threshold=0.92, max_entries=1000)
_corpus_version: int = 1


def _invalidate_cache(filename: str | None = None):
    """Invalidate semantic query cache when documents are uploaded or deleted."""
    global _corpus_version
    _corpus_version += 1
    if filename:
        removed = semantic_cache.invalidate_document(filename)
        logger.info(f"Semantic cache invalidated for '{filename}': {removed} queries removed")
    else:
        semantic_cache.clear()
        logger.info("Semantic cache fully cleared")


# ── App definition ──

app = FastAPI(
    title="DocuMind API",
    description="Agentic multimodal RAG for visually rich enterprise documents",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down to specific frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request ID middleware ──

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Inject a unique X-Request-ID header for tracing."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Startup / Shutdown ──

@app.on_event("startup")
async def startup():
    """Pre-initialize components and restore BM25 index on startup."""
    settings.ensure_dirs()
    Path("./storage").mkdir(parents=True, exist_ok=True)
    logger.info("DocuMind API starting up...")

    # ── Validate Gemini API key early ──
    if settings.using_gemini:
        key = settings.gemini_api_key
        if not key:
            logger.error(
                "GEMINI_API_KEY is empty! Set it in .env. "
                "Get a free key at https://aistudio.google.com/apikey"
            )

    # Initialize components eagerly so first request is fast
    try:
        components = get_components()
        # Restore BM25 index from disk if available
        bm25: BM25Index = components["bm25_index"]
        if BM25_INDEX_PATH.exists():
            bm25.load(BM25_INDEX_PATH)
            logger.info(f"Restored BM25 index from {BM25_INDEX_PATH}")

        # Pre-warm embedder and reranker
        try:
            embedder: Embedder = components["embedder"]
            embedder.embed_text("DocuMind warm-up query")
            logger.info("Pre-warmed embedder model")
        except Exception as emb_err:
            logger.warning(f"Embedder warmup skipped: {emb_err}")

        # Pre-warm reranker on CPU so user queries never suffer cold-start latency
        if settings.reranker_enabled:
            components["reranker"]._load_model()
            logger.info("Pre-warmed reranker model on CPU")

        # Warm up Ollama LLM and pin in VRAM
        if settings.using_ollama:
            try:
                from ollama import Client as OllamaClient
                oc = OllamaClient(host=settings.ollama_base_url)
                logger.info(f"Pinning Ollama LLM ({settings.ollama_llm_model}) in GPU VRAM (keep_alive=24h)...")
                oc.chat(
                    model=settings.ollama_llm_model,
                    messages=[{"role": "user", "content": "ping"}],
                    options={"temperature": 0.0, "num_ctx": 2048, "num_predict": 5},
                    keep_alive="24h",
                )
                logger.info("Ollama LLM pre-warmed & pinned in GPU VRAM (keep_alive=24h)")
            except Exception as llm_err:
                logger.warning(f"Ollama LLM warmup skipped: {llm_err}")

        logger.info("All components initialized and pre-warmed")
    except Exception as e:
        logger.warning(f"Startup initialization incomplete (will recover on first request): {e}")

    # ── Auto-pull embedding model if missing (only when using local Ollama) ──
    if settings.using_ollama:
        try:
            from ollama import Client as OllamaClient
            oc = OllamaClient(host=settings.ollama_base_url)
            model_names = [m.model for m in oc.list().models]
            if not any(settings.ollama_embed_model in n for n in model_names):
                logger.info(f"Pulling embedding model '{settings.ollama_embed_model}'...")
                oc.pull(settings.ollama_embed_model)
                logger.info(f"Pulled '{settings.ollama_embed_model}' successfully")
        except Exception as e:
            logger.warning(
                f"Could not auto-pull {settings.ollama_embed_model}: {e}"
            )


# ── Component initialization ──

def get_components() -> dict:
    """Get or initialize all pipeline components (singleton)."""
    if not _components:
        settings.ensure_dirs()

        embedder = get_embedder()
        qdrant_store = QdrantStore()
        bm25_index = BM25Index()
        reranker = Reranker()

        # Try to ensure the Qdrant collection exists
        try:
            dim = embedder.embedding_dim
            recreated = qdrant_store.ensure_collection(embedding_dim=dim)
            if recreated:
                logger.info("Collection initialized or recreated; synchronizing BM25 index and hash registry")
                bm25_index.clear()
                if BM25_INDEX_PATH.exists():
                    try:
                        BM25_INDEX_PATH.unlink()
                    except Exception:
                        pass
                _save_hash_registry({})
            elif BM25_INDEX_PATH.exists() and len(bm25_index._chunks) == 0:
                bm25_index.load(BM25_INDEX_PATH)
                logger.info(f"Loaded persistent BM25 index ({len(bm25_index._chunks)} docs) from {BM25_INDEX_PATH}")
        except Exception as e:
            logger.warning(f"Could not initialize Qdrant collection: {e}")

        _components.update({
            "embedder": embedder,
            "qdrant_store": qdrant_store,
            "bm25_index": bm25_index,
            "reranker": reranker,
            "parser": PDFParser(image_output_dir=settings.upload_dir / "images"),
            "chunker": DocumentChunker(),
        })

    return _components


def get_pipeline():
    """Get or build the LangGraph pipeline (singleton)."""
    if "pipeline" not in _components:
        components = get_components()
        pipeline = build_pipeline(
            qdrant_store=components["qdrant_store"],
            bm25_index=components["bm25_index"],
            embedder=components["embedder"],
            reranker=components["reranker"],
        )
        _components["pipeline"] = pipeline
    return _components["pipeline"]


# ── Hash registry helpers ──

def _load_hash_registry() -> dict[str, str]:
    """Load content-hash → filename map from disk."""
    if HASH_REGISTRY_PATH.exists():
        try:
            return json.loads(HASH_REGISTRY_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_hash_registry(registry: dict[str, str]) -> None:
    """Persist the hash registry to disk."""
    HASH_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HASH_REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Ingestion helper (runs in thread pool) ──

def _ingest_pdf_sync(
    save_path: Path,
    components: dict,
) -> dict:
    """Synchronous ingestion logic — runs in thread pool to not block event loop.

    Returns a dict with ingestion stats.
    """
    parser: PDFParser = components["parser"]
    chunker: DocumentChunker = components["chunker"]
    embedder: Embedder = components["embedder"]
    qdrant: QdrantStore = components["qdrant_store"]
    bm25: BM25Index = components["bm25_index"]

    # Parse PDF
    parsed_doc = parser.parse(save_path)

    # OCR scanned pages
    scanned_pages = [p.page_number for p in parsed_doc.pages if p.is_scanned]
    ocr_texts = {}
    if scanned_pages:
        logger.info(f"Running OCR on {len(scanned_pages)} scanned pages...")
        ocr_texts = ocr_pdf_pages(save_path, page_numbers=scanned_pages)

    # Chunk
    chunks = chunker.chunk_document(parsed_doc, ocr_texts=ocr_texts)
    if not chunks:
        raise ValueError("No content could be extracted from the PDF")

    # Embed
    chunk_embedding_pairs = embedder.embed_chunks(chunks)
    embeddings = [emb for _, emb in chunk_embedding_pairs]

    # Store in Qdrant
    qdrant.upsert_chunks(chunks, embeddings)

    # Rebuild BM25 incrementally (adds new chunks to existing corpus)
    bm25.add_chunks(chunks)

    # Persist updated BM25 to disk
    bm25.save(BM25_INDEX_PATH)

    return {
        "total_pages": parsed_doc.total_pages,
        "total_chunks": len(chunks),
    }


# ── Routes ──


@app.post("/api/v1/upload", response_model=UploadResponse)
@app.post("/upload", response_model=UploadResponse, include_in_schema=False)  # backwards compat
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Upload and ingest a PDF document.

    Steps:
    1. Validate file type
    2. Check for duplicate content (SHA-256 hash)
    3. Save file to disk
    4. Parse, chunk, embed, and store asynchronously
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    components = get_components()
    settings.ensure_dirs()

    # ── Lazily ensure Qdrant collection exists ──
    try:
        dim = components["embedder"].embedding_dim
        recreated = components["qdrant_store"].ensure_collection(embedding_dim=dim)
        if recreated:
            logger.info("Collection recreated during upload; synchronizing BM25 index and hash registry")
            components["bm25_index"].clear()
            if BM25_INDEX_PATH.exists():
                try:
                    BM25_INDEX_PATH.unlink()
                except Exception:
                    pass
            _save_hash_registry({})
    except Exception as e:
        provider_hint = (
            "Check GEMINI_API_KEY in .env"
            if settings.using_gemini
            else f"Is Ollama running with '{settings.ollama_embed_model}' pulled?"
        )
        raise HTTPException(
            status_code=503,
            detail=f"Vector store not ready: {e}. {provider_hint}",
        )

    # Save uploaded file to a temp path first (to compute hash before committing)
    save_path = settings.upload_dir / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"Saved uploaded file: {save_path}")

    # ── Duplicate detection ──
    content_hash = _file_sha256(save_path)
    registry = _load_hash_registry()

    if content_hash in registry:
        existing_filename = registry[content_hash]
        logger.info(f"Duplicate detected: {file.filename} matches existing {existing_filename}")
        return UploadResponse(
            filename=file.filename,
            total_pages=0,
            total_chunks=0,
            skipped=True,
            message=(
                f"Duplicate content detected — this document was already indexed as "
                f"'{existing_filename}'. Skipping re-ingestion."
            ),
        )

    try:
        # ── Run ingestion in thread pool (non-blocking) ──
        loop = asyncio.get_running_loop()
        stats = await loop.run_in_executor(
            _executor,
            _ingest_pdf_sync,
            save_path,
            components,
        )

        # Register content hash after successful ingestion
        registry[content_hash] = file.filename
        _save_hash_registry(registry)
        _invalidate_cache()

        return UploadResponse(
            filename=file.filename,
            total_pages=stats["total_pages"],
            total_chunks=stats["total_chunks"],
            message=(
                f"Successfully ingested '{file.filename}': "
                f"{stats['total_pages']} pages, {stats['total_chunks']} chunks"
            ),
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/v1/query", response_model=QueryResponse)
@app.post("/query", response_model=QueryResponse, include_in_schema=False)  # backwards compat
async def query_documents(request: Request, body: QueryRequest):
    """Ask a question about uploaded documents with dual-tier semantic caching.

    Checks:
    1. Exact query match in cache (0ms)
    2. Dense vector cosine similarity match in cache (sim >= 0.92)
    3. If miss: Runs full LangGraph pipeline & stores user query in semantic cache
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"[{request_id}] Query: {body.question!r} (scope: {body.source_file})")
    t0 = time.time()

    components = get_components()
    embedder: Embedder = components["embedder"]

    # 1. Tier 1: Exact match lookup (0ms)
    cached_data, sim_score, match_type = semantic_cache.lookup(body.question, None, body.source_file)
    query_emb = None
    if cached_data is None:
        # 2. Tier 2: Semantic vector match
        query_emb = embedder.embed_text(body.question)
        cached_data, sim_score, match_type = semantic_cache.lookup(body.question, query_emb, body.source_file)

    if cached_data is not None:
        logger.info(f"[{request_id}] Serving query from {match_type} semantic cache (score={sim_score:.2f})")
        raw_c = cached_data.get("citations", [])
        return QueryResponse(
            answer=cached_data["answer"],
            citations=[Citation(**c) if isinstance(c, dict) else c for c in raw_c],
            confidence=cached_data.get("confidence", 0.0),
            abstained=cached_data.get("abstained", False),
            query_type=cached_data.get("query_type", "text"),
            retry_count=cached_data.get("retry_count", 0),
            request_id=request_id,
            latency_s=0.01,
            stage_timings={"semantic_cache_s": 0.005},
            cached=True,
            error=None,
        )

    try:
        pipeline = get_pipeline()
        result = run_query(
            pipeline=pipeline,
            question=body.question,
            source_file=body.source_file,
        )

        latency_s = round(time.time() - t0, 2)

        # ── Build Citation objects from the raw dicts ──
        raw_citations: list[dict] = result.get("citations", [])
        citations = []
        for c in raw_citations:
            try:
                citations.append(Citation(
                    page=c["page"],
                    block_type=c.get("block_type", "text"),
                    source_file=c.get("source_file", ""),
                    snippet=c.get("snippet", ""),
                    section_heading=c.get("section_heading", ""),
                    parent_text=c.get("parent_text", ""),
                    bbox=c.get("bbox"),
                ))
            except Exception as citation_err:
                logger.warning(f"[{request_id}] Skipping malformed citation: {citation_err}")

        # Store user query into semantic cache (only real user queries fill the cache)
        if query_emb is None:
            query_emb = embedder.embed_text(body.question)

        cache_payload = {
            "answer": result["answer"],
            "citations": raw_citations,
            "confidence": result["confidence"],
            "abstained": result["abstained"],
            "query_type": result["query_type"],
            "retry_count": result["retry_count"],
            "stage_timings": result.get("stage_timings", {}),
        }
        semantic_cache.store(body.question, query_emb, body.source_file, cache_payload)

        return QueryResponse(
            answer=result["answer"],
            citations=citations,
            confidence=result["confidence"],
            abstained=result["abstained"],
            query_type=result["query_type"],
            retry_count=result["retry_count"],
            request_id=request_id,
            latency_s=latency_s,
            stage_timings=result.get("stage_timings", {}),
            cached=False,
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"[{request_id}] Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/api/v1/query/stream")
@app.post("/query/stream", include_in_schema=False)
async def query_documents_stream(request: Request, body: QueryRequest):
    """Stream answer generation token-by-token with dual-tier semantic caching via SSE."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"[{request_id}] Streaming query: {body.question!r} (scope: {body.source_file})")

    components = get_components()
    embedder: Embedder = components["embedder"]

    # 1. Tier 1: Exact string match (0ms)
    cached_data, sim_score, match_type = semantic_cache.lookup(body.question, None, body.source_file)
    query_emb = None
    if cached_data is None:
        # 2. Tier 2: Semantic vector match (cosine sim >= 0.92)
        query_emb = embedder.embed_text(body.question)
        cached_data, sim_score, match_type = semantic_cache.lookup(body.question, query_emb, body.source_file)

    if cached_data is not None:
        logger.info(f"[{request_id}] Serving streaming query from {match_type} semantic cache ({sim_score:.1%})")

        async def _cached_stream():
            msg = (
                f"⚡ Semantic Cache Hit ({sim_score:.1%} match)"
                if match_type == "semantic"
                else "⚡ Exact Match Cache Hit (0ms)"
            )
            yield f"data: {json.dumps({'event': 'stage', 'stage': 'cached', 'message': msg})}\n\n"
            yield f"data: {json.dumps({'event': 'token', 'token': cached_data['answer']})}\n\n"
            final_data = dict(cached_data)
            final_data["cached"] = True
            final_data["latency_s"] = 0.01
            final_data["request_id"] = request_id
            yield f"data: {json.dumps({'event': 'final', 'data': final_data})}\n\n"

        return StreamingResponse(
            _cached_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Request-ID": request_id},
        )

    pipeline = get_pipeline()

    async def _sse_streamer():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _worker():
            try:
                for event in run_query_stream(pipeline, body.question, body.source_file):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:
                logger.error(f"[{request_id}] Streaming error in worker: {exc}")
                loop.call_soon_threadsafe(queue.put_nowait, {"event": "error", "message": str(exc)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # Offload pipeline stream generation to worker thread
        loop.run_in_executor(_executor, _worker)

        while True:
            item = await queue.get()
            if item is None:
                break
            if item.get("event") == "final":
                data = item.get("data", {})
                data["request_id"] = request_id
                # Store user query into semantic cache (only actual user queries fill the cache)
                nonlocal query_emb
                if query_emb is None:
                    query_emb = embedder.embed_text(body.question)
                semantic_cache.store(body.question, query_emb, body.source_file, data)
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        _sse_streamer(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Request-ID": request_id},
    )


@app.get("/api/v1/documents", response_model=DocumentListResponse)
@app.get("/documents", response_model=DocumentListResponse, include_in_schema=False)
async def list_documents():
    """List all indexed documents and their chunk counts."""
    try:
        store: QdrantStore = get_components()["qdrant_store"]
        # Scroll through all payloads to aggregate by source_file
        offset = None
        file_counts: dict[str, int] = {}

        while True:
            result = store.client.scroll(
                collection_name=store.collection_name,
                limit=1000,
                offset=offset,
                with_payload=["source_file"],
            )
            points, next_offset = result
            for point in points:
                src = (point.payload or {}).get("source_file", "unknown")
                file_counts[src] = file_counts.get(src, 0) + 1

            if next_offset is None:
                break
            offset = next_offset

        docs = [DocumentInfo(filename=f, chunk_count=c) for f, c in sorted(file_counts.items())]
        return DocumentListResponse(documents=docs, total=len(docs))

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.delete("/api/v1/documents")
@app.delete("/documents", include_in_schema=False)
async def clear_documents():
    """Clear all indexed documents from Qdrant, BM25, and hash registry."""
    try:
        components = get_components()
        store: QdrantStore = components["qdrant_store"]
        dim = components["embedder"].embedding_dim

        # 1. Scroll and delete all points explicitly (safe on Windows SQLite)
        try:
            offset = None
            all_ids = []
            while True:
                points, next_offset = store.client.scroll(
                    collection_name=store.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                all_ids.extend([p.id for p in points])
                if next_offset is None:
                    break
                offset = next_offset

            if all_ids:
                store.client.delete(
                    collection_name=store.collection_name,
                    points_selector=models.PointIdsList(points=all_ids),
                )
                logger.info(f"Deleted {len(all_ids)} points from Qdrant collection '{store.collection_name}'")
        except Exception as e:
            logger.warning(f"Could not delete points via selector: {e}")
            try:
                store.client.delete_collection(store.collection_name)
                store.ensure_collection(embedding_dim=dim)
            except Exception:
                pass

        # 2. Clear BM25 index and remove pickle file
        components["bm25_index"].clear()
        if BM25_INDEX_PATH.exists():
            try:
                BM25_INDEX_PATH.unlink()
            except Exception:
                pass

        # 3. Clear content hash registry
        _save_hash_registry({})

        # 4. Remove uploaded files from upload_dir
        if settings.upload_dir.exists():
            for f in settings.upload_dir.glob("*.pdf"):
                try:
                    f.unlink()
                except Exception:
                    pass

        _invalidate_cache()
        return {"status": "success", "message": "All documents cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")


@app.delete("/api/v1/documents/{filename:path}")
@app.delete("/documents/{filename:path}", include_in_schema=False)
async def delete_document(filename: str):
    """Delete a single document by filename from Qdrant, BM25, and disk."""
    try:
        components = get_components()
        store: QdrantStore = components["qdrant_store"]

        # 1. Delete points in Qdrant matching source_file == filename
        try:
            store.client.delete(
                collection_name=store.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_file",
                                match=models.MatchValue(value=filename),
                            )
                        ]
                    )
                ),
            )
            logger.info(f"Deleted Qdrant points for file '{filename}'")
        except Exception as e:
            logger.warning(f"Failed to delete Qdrant points for '{filename}': {e}")

        # 2. Remove chunks from BM25 and rebuild
        bm25: BM25Index = components["bm25_index"]
        original_count = len(bm25._chunks)
        bm25._chunks = [c for c in bm25._chunks if getattr(c, "source_file", "") != filename]
        if len(bm25._chunks) != original_count:
            bm25._tokenized_corpus = [bm25._tokenize(c.text) for c in bm25._chunks]
            bm25._rebuild_index()
            bm25.save(BM25_INDEX_PATH)
            logger.info(f"Updated BM25 index after deleting '{filename}'")

        # 3. Remove from hash registry
        registry = _load_hash_registry()
        updated_registry = {h: fn for h, fn in registry.items() if fn != filename}
        _save_hash_registry(updated_registry)

        # 4. Remove file from upload_dir if present
        target_file = settings.upload_dir / filename
        if target_file.exists():
            try:
                target_file.unlink()
            except Exception:
                pass

        # Invalidate semantic cache specifically for this deleted document
        _invalidate_cache(filename=filename)
        return {"status": "success", "message": f"Document '{filename}' deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete document '{filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.get("/health", response_model=HealthResponse)
@app.get("/api/v1/health", response_model=HealthResponse, include_in_schema=False)
async def health_check():
    """Check the health of all system components."""
    ollama_ok = False
    gemini_ok = False
    qdrant_ok = False
    docs_indexed = 0
    collection_info = {}

    if settings.using_gemini:
        gemini_ok = bool(settings.gemini_api_key)
    else:
        # Check Ollama
        try:
            from ollama import Client
            client = Client(host=settings.ollama_base_url)
            client.list()
            ollama_ok = True
        except Exception:
            pass

    # Check Qdrant — reachable = healthy; collection may not exist yet if no docs uploaded
    try:
        components = get_components()
        store: QdrantStore = components["qdrant_store"]
        info = store.get_collection_info()
        # 'error' key means a genuine connection failure; 'not_created' is normal pre-upload
        if "error" not in info:
            qdrant_ok = True
            docs_indexed = info.get("points_count", 0) or 0
            collection_info = info
        else:
            # Try a raw ping to distinguish "Qdrant down" from "collection missing"
            store.client.get_collections()
            qdrant_ok = True  # Qdrant reachable, collection just not created yet
            collection_info = {"status": "not_created"}
    except Exception:
        pass

    provider_ok = gemini_ok if settings.using_gemini else ollama_ok
    status = "healthy" if (provider_ok and qdrant_ok) else "degraded"
    if not provider_ok and not qdrant_ok:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        provider=settings.provider,
        provider_connected=provider_ok,
        ollama_connected=ollama_ok,
        qdrant_connected=qdrant_ok,
        documents_indexed=docs_indexed,
        collection_info=collection_info,
    )
