# DocuMind

Agentic multimodal RAG for visually rich enterprise documents (PDFs with text, tables, charts, scans, and complex layout).

## Problem

Naive text-only RAG loses critical signal from tables/charts/layout. DocuMind keeps both textual and visual evidence during ingestion and retrieval, then answers with page-level citations and grounding checks.

## Why this project matters (2026)

Multimodal document QA is a core enterprise/research frontier. This project demonstrates retrieval quality, grounding, and orchestration design beyond plain `.similarity_search()`.

## Target architecture

```text
User (Web UI: upload PDF / ask question)
   |
FastAPI Backend
   - POST /upload
   - POST /query
   - GET  /health
   |
LangGraph Orchestrator
   classify_query_type
      ├─ retrieve_text
      └─ retrieve_visual (VLM-assisted table/chart understanding)
          -> rerank (cross-encoder)
          -> generate_answer (LLM)
          -> verify_citations (NLI/grounding check)
             ├─ grounded: return answer + citations + confidence
             └─ not grounded: reformulate + retry retrieve (max 2), else "I don't know"
   |
Storage / Inference Layer
   - Vector DB (Qdrant/Chroma): text + visual embeddings, BM25/sparse index
   - Ingestion: layout parsing, OCR, chunking, table/figure extraction
   - Models: embedding model + VLM + generator LLM + reranker
```

## Core pipeline

1. **Ingest PDF**
   - Parse layout blocks (headers, paragraphs, tables, figures).
   - OCR scanned pages.
   - Extract table/figure regions with page + bounding box metadata.
2. **Represent content**
   - Text chunks: recursive/layout-aware splitting.
   - Visual/table/chart regions: VLM to structured text/captions.
   - Embed both text and visual-derived text.
   - Build sparse index (BM25) for exact term/number matching.
3. **Retrieve**
   - Query classification: text-heavy vs table/chart-heavy.
   - Hybrid dense+sparse retrieval per route.
   - Fuse candidates and rerank with cross-encoder.
4. **Generate**
   - LLM answer constrained to retrieved evidence.
   - Return structured output with citations (page, snippet).
5. **Verify & retry**
   - Entailment/grounding check on claims vs cited evidence.
   - If weak grounding, reformulate and re-retrieve (max 2 loops).
   - If still ungrounded, return **"I don't know"**.

## Why each component exists

- **Hybrid retrieval:** catches exact values/keywords missed by dense vectors.
- **Reranker:** improves top-k precision after cheap first-pass retrieval.
- **LangGraph loop:** enables verify → re-retrieve recovery path.
- **Visual path:** preserves chart/table signal absent in text-only RAG.

## Recommended model stack

- **Embeddings:** `bge-base-en-v1.5`, `nomic-embed-text`, or `text-embedding-3-small`
- **VLM (charts/tables):** `Qwen2.5-VL-7B`, `GPT-4o-mini`, or `LLaVA-NeXT`
- **Generator LLM:** `Llama-3.1-8B-Instruct` (local) or `GPT-4o-mini` / `Claude Haiku`
- **Reranker:** `bge-reranker-base` or `Cohere Rerank`

## Datasets

- Public benchmarks: **DocVQA**, **FinQA / FinanceBench**, **SlideVQA**
- Project corpus: 50–100 real documents (e.g., SEC EDGAR 10-K reports, manuals, papers)

## Tech stack

- Python, PyTorch, Hugging Face Transformers
- LangChain (loaders/splitters/vector-store abstraction only)
- LangGraph (query orchestration loop)
- Qdrant or ChromaDB
- FastAPI backend
- Streamlit or React frontend
- Docker (local reproducibility)

## Quantitative evaluation

- **Retrieval:** Recall@k, MRR, nDCG
- **Generation:** RAGAS (faithfulness, relevancy, context precision/recall), EM/F1 on QA pairs
- **Safety/grounding:** hallucination rate, citation accuracy
- **Systems:** p50/p95 latency

## Deployment plan

- Backend + Qdrant via Docker Compose
- Host API on low-cost cloud VM or Hugging Face Spaces
- Frontend on Streamlit Cloud/Vercel
- Prefer quantized local models (GGUF via llama.cpp/Ollama) with optional API fallback

## 10–12 week roadmap (part-time)

1. **Weeks 1–2:** ingestion + OCR + chunking baseline
2. **Weeks 3–4:** dense retrieval + sparse BM25 + hybrid fusion
3. **Weeks 5–6:** visual/table/chart path with VLM extraction
4. **Weeks 7–8:** LangGraph orchestration + verify/retry loop
5. **Weeks 9–10:** reranking + evaluation harness + hallucination/citation metrics
6. **Weeks 11–12:** optimization, deployment, and demo hardening

## Deliverable behavior

For each answer, the system should return:

- Final answer text
- Page-level citations (and optional bounding boxes)
- Confidence/grounding signal
- Explicit abstention (`"I don't know"`) when evidence is insufficient