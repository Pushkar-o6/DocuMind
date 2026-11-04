"""Advanced Text Chunking Strategies for Enterprise RAG.

Strategies supported:
1. Document-Structure-Aware Chunking: Splits along visual/semantic boundaries
   (headers, titles, bold text, page breaks, and markdown sections).
2. Recursive Character Splitting (Adaptive):
   - Factoid mode: Target 256 to 512 tokens (~1000-2000 chars) with 10-20% overlap.
   - Analytical mode: Target 512 to 1024 tokens (~2000-4000 chars) with 10-20% overlap.
3. Semantic Chunking with Size Floor:
   - Splits on topic/paragraph transitions while enforcing a strict minimum size
     floor (200-400 tokens / 800-1600 chars) to prevent degenerate 40-token chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from documind.config import settings
from documind.ingestion.parser import ContentBlock, ParsedDocument

ChunkingStrategyType = Literal["structure_aware", "recursive", "semantic"]


@dataclass
class Chunk:
    """A text chunk ready for embedding and indexing.

    Carries rich provenance metadata (source doc, page, section heading,
    block type, bbox, timestamp) for hard pre-filtering and precise citation.
    """

    chunk_id: str
    text: str
    page_number: int
    block_type: str  # "text", "table", "header", "image", "ocr"
    bbox: tuple[float, float, float, float] | None = None
    source_file: str = ""
    section_heading: str = ""
    strategy: str = "structure_aware"
    token_count_est: int = 0
    parent_text: str = ""  # Small-to-big parent context
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict = field(default_factory=dict)


class DocumentChunker:
    """Multi-strategy enterprise document chunker."""

    def __init__(
        self,
        strategy: ChunkingStrategyType | None = None,
        query_type: Literal["factoid", "analytical"] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_tokens: int | None = None,
    ):
        self.strategy = strategy or getattr(settings, "chunking_strategy", "structure_aware")
        self.query_type = query_type or getattr(settings, "chunk_query_type", "factoid")
        self.min_tokens = min_tokens or getattr(settings, "chunk_min_tokens", 150)

        # Token targeting: approx 4 chars per token in English
        # Factoid: 256-512 tokens (~1024-2048 chars), overlap 10-20% (~50-100 tokens / 200-400 chars)
        # Analytical: 512-1024 tokens (~2048-4096 chars), overlap 10-20% (~100-200 tokens / 400-800 chars)
        if chunk_size:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap if chunk_overlap is not None else 64
        elif self.query_type == "analytical":
            self.chunk_size = 2500  # ~600 tokens
            self.chunk_overlap = 350 # ~14% overlap
        else:  # factoid (default)
            self.chunk_size = 1200  # ~300 tokens
            self.chunk_overlap = 180 # ~15% overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=len,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Heuristic token count estimation (~4 characters per token)."""
        return max(1, len(text) // 4)

    def chunk_document(
        self,
        parsed_doc: ParsedDocument,
        ocr_texts: dict[int, str] | None = None,
    ) -> list[Chunk]:
        """Chunk a parsed document according to the configured strategy."""
        if self.strategy == "structure_aware":
            return self._chunk_structure_aware(parsed_doc, ocr_texts)
        elif self.strategy == "semantic":
            return self._chunk_semantic_with_floor(parsed_doc, ocr_texts)
        else:
            return self._chunk_recursive(parsed_doc, ocr_texts)

    # ── Strategy 1: Document Structure-Aware Chunking ──

    def _chunk_structure_aware(
        self,
        parsed_doc: ParsedDocument,
        ocr_texts: dict[int, str] | None = None,
    ) -> list[Chunk]:
        """Split document respecting section headings, tables, and page boundaries."""
        chunks: list[Chunk] = []
        chunk_counter = 0

        current_heading = "Introduction"

        for page in parsed_doc.pages:
            # Handle scanned pages
            if page.is_scanned and ocr_texts and page.page_number in ocr_texts:
                ocr_text = ocr_texts[page.page_number].strip()
                if ocr_text:
                    sub_texts = self._split_text_with_floor(ocr_text)
                    for st in sub_texts:
                        chunks.append(
                            Chunk(
                                chunk_id=f"{parsed_doc.filename}_c{chunk_counter}",
                                text=st,
                                page_number=page.page_number,
                                block_type="ocr",
                                source_file=parsed_doc.filename,
                                section_heading=current_heading,
                                strategy="structure_aware",
                                token_count_est=self._estimate_tokens(st),
                                metadata={"ocr": True},
                            )
                        )
                        chunk_counter += 1
                continue

            current_section_blocks: list[ContentBlock] = []

            for block in page.blocks:
                # Standalone block: Tables
                if block.block_type == "table":
                    # Flush current section text before table
                    if current_section_blocks:
                        chunks.extend(
                            self._create_chunks_from_blocks(
                                current_section_blocks,
                                parsed_doc.filename,
                                page.page_number,
                                current_heading,
                                chunk_counter,
                            )
                        )
                        chunk_counter = len(chunks)
                        current_section_blocks = []

                    chunks.append(
                        Chunk(
                            chunk_id=f"{parsed_doc.filename}_c{chunk_counter}",
                            text=block.content,
                            page_number=page.page_number,
                            block_type="table",
                            bbox=block.bbox,
                            source_file=parsed_doc.filename,
                            section_heading=current_heading,
                            strategy="structure_aware",
                            token_count_est=self._estimate_tokens(block.content),
                            metadata=block.metadata,
                        )
                    )
                    chunk_counter += 1
                    continue

                # Header detection: bold or larger font or Markdown heading pattern
                is_heading = self._is_heading_block(block)
                if is_heading:
                    # Flush previous section
                    if current_section_blocks:
                        chunks.extend(
                            self._create_chunks_from_blocks(
                                current_section_blocks,
                                parsed_doc.filename,
                                page.page_number,
                                current_heading,
                                chunk_counter,
                            )
                        )
                        chunk_counter = len(chunks)
                        current_section_blocks = []

                    current_heading = block.content.strip().split("\n")[0][:120]

                current_section_blocks.append(block)

            # Flush end of page blocks
            if current_section_blocks:
                chunks.extend(
                    self._create_chunks_from_blocks(
                        current_section_blocks,
                        parsed_doc.filename,
                        page.page_number,
                        current_heading,
                        chunk_counter,
                    )
                )
                chunk_counter = len(chunks)

        logger.info(
            f"Structure-aware chunking for {parsed_doc.filename}: generated {len(chunks)} chunks"
        )
        return chunks

    def _is_heading_block(self, block: ContentBlock) -> bool:
        """Detect whether a block represents a heading/section title."""
        if block.block_type != "text":
            return False

        text = block.content.strip()
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines or len(lines) > 3:
            return False

        first_line = lines[0]
        # Check markdown header prefix
        if re.match(r"^#{1,4}\s+\w+", first_line):
            return True

        # Check font metadata if present
        is_bold = block.metadata.get("is_bold", False)
        font_sizes = block.metadata.get("font_sizes", [])
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 11.0

        if (is_bold and len(text) < 120) or (avg_font_size > 13.5 and len(text) < 150):
            return True

        return False

    def _create_chunks_from_blocks(
        self,
        blocks: list[ContentBlock],
        source_file: str,
        page_number: int,
        heading: str,
        start_counter: int,
    ) -> list[Chunk]:
        """Aggregate text blocks, calculate bounding box, and split with floor."""
        if not blocks:
            return []

        text_content = "\n\n".join(b.content for b in blocks if b.content.strip()).strip()
        if not text_content:
            return []

        bbox = (
            min(b.bbox[0] for b in blocks),
            min(b.bbox[1] for b in blocks),
            max(b.bbox[2] for b in blocks),
            max(b.bbox[3] for b in blocks),
        )

        segments = self._split_text_with_floor(text_content)
        result = []
        for i, seg in enumerate(segments):
            # Contextual chunk enrichment: prepend title, page, and section context
            enriched_text = f"[Document: {source_file}] [Section: {heading}] [Page: {page_number}]\n{seg}"
            result.append(
                Chunk(
                    chunk_id=f"{source_file}_c{start_counter + i}",
                    text=enriched_text,
                    page_number=page_number,
                    block_type="text",
                    bbox=bbox,
                    source_file=source_file,
                    section_heading=heading,
                    strategy="structure_aware",
                    token_count_est=self._estimate_tokens(enriched_text),
                    parent_text=text_content, # Surrounding section context for small-to-big retrieval
                    metadata={"section": heading, "raw_snippet": seg[:250]},
                )
            )
        return result

    # ── Strategy 2: Recursive Character Splitting (Adaptive) ──

    def _chunk_recursive(
        self,
        parsed_doc: ParsedDocument,
        ocr_texts: dict[int, str] | None = None,
    ) -> list[Chunk]:
        """Adaptive recursive character splitting tuned for factoid or analytical queries."""
        all_chunks: list[Chunk] = []
        chunk_counter = 0

        for page in parsed_doc.pages:
            page_text_blocks = [b for b in page.blocks if b.block_type == "text"]
            special_blocks = [b for b in page.blocks if b.block_type != "text"]

            # OCR fallback for scanned pages
            if page.is_scanned and ocr_texts and page.page_number in ocr_texts:
                ocr_text = ocr_texts[page.page_number]
                if ocr_text.strip():
                    for seg in self._split_text_with_floor(ocr_text):
                        all_chunks.append(
                            Chunk(
                                chunk_id=f"{parsed_doc.filename}_c{chunk_counter}",
                                text=seg,
                                page_number=page.page_number,
                                block_type="ocr",
                                source_file=parsed_doc.filename,
                                strategy="recursive",
                                token_count_est=self._estimate_tokens(seg),
                                metadata={"ocr": True},
                            )
                        )
                        chunk_counter += 1
                continue

            # Text blocks
            if page_text_blocks:
                merged = "\n\n".join(b.content for b in page_text_blocks)
                bbox = (
                    min(b.bbox[0] for b in page_text_blocks),
                    min(b.bbox[1] for b in page_text_blocks),
                    max(b.bbox[2] for b in page_text_blocks),
                    max(b.bbox[3] for b in page_text_blocks),
                )
                for seg in self._split_text_with_floor(merged):
                    all_chunks.append(
                        Chunk(
                            chunk_id=f"{parsed_doc.filename}_c{chunk_counter}",
                            text=seg,
                            page_number=page.page_number,
                            block_type="text",
                            bbox=bbox,
                            source_file=parsed_doc.filename,
                            strategy="recursive",
                            token_count_est=self._estimate_tokens(seg),
                        )
                    )
                    chunk_counter += 1

            # Tables & image placeholders
            for sb in special_blocks:
                all_chunks.append(
                    Chunk(
                        chunk_id=f"{parsed_doc.filename}_c{chunk_counter}",
                        text=sb.content,
                        page_number=page.page_number,
                        block_type=sb.block_type,
                        bbox=sb.bbox,
                        source_file=parsed_doc.filename,
                        strategy="recursive",
                        token_count_est=self._estimate_tokens(sb.content),
                        metadata=sb.metadata,
                    )
                )
                chunk_counter += 1

        return all_chunks

    # ── Strategy 3: Semantic Chunking with Size Floor ──

    def _chunk_semantic_with_floor(
        self,
        parsed_doc: ParsedDocument,
        ocr_texts: dict[int, str] | None = None,
    ) -> list[Chunk]:
        """Split text at topical paragraph transitions with a strict minimum token floor.

        Prevents small 40-token fragments by merging sub-threshold segments into the
        preceding or succeeding chunk.
        """
        all_chunks: list[Chunk] = []
        chunk_counter = 0

        # Min character floor = min_tokens * 4
        char_floor = self.min_tokens * 4

        for page in parsed_doc.pages:
            page_content = "\n\n".join(b.content for b in page.blocks if b.block_type == "text")
            if not page_content.strip():
                continue

            # Split on paragraph boundaries
            raw_paragraphs = [p.strip() for p in page_content.split("\n\n") if p.strip()]
            semantic_groups: list[str] = []
            current_group = ""

            for p in raw_paragraphs:
                if not current_group:
                    current_group = p
                elif len(current_group) + len(p) < self.chunk_size:
                    current_group += "\n\n" + p
                else:
                    # Enforce floor check before finalizing chunk
                    if len(current_group) < char_floor:
                        current_group += "\n\n" + p
                    else:
                        semantic_groups.append(current_group)
                        current_group = p

            if current_group:
                # Merge small trailing fragment with prior chunk if below floor
                if semantic_groups and len(current_group) < char_floor:
                    semantic_groups[-1] += "\n\n" + current_group
                else:
                    semantic_groups.append(current_group)

            for seg in semantic_groups:
                all_chunks.append(
                    Chunk(
                        chunk_id=f"{parsed_doc.filename}_c{chunk_counter}",
                        text=seg,
                        page_number=page.page_number,
                        block_type="text",
                        source_file=parsed_doc.filename,
                        strategy="semantic",
                        token_count_est=self._estimate_tokens(seg),
                    )
                )
                chunk_counter += 1

        return all_chunks

    # ── Helper: Splitting with size floor enforcement ──

    def _split_text_with_floor(self, text: str) -> list[str]:
        """Split text recursively and ensure no fragment falls below the token size floor."""
        raw_splits = [doc.page_content for doc in self.text_splitter.create_documents([text])]
        if not raw_splits:
            return []

        char_floor = self.min_tokens * 4
        consolidated: list[str] = []
        buffer = ""

        for s in raw_splits:
            if not buffer:
                buffer = s
            elif len(buffer) < char_floor:
                buffer += " " + s
            elif len(s) < char_floor:
                # Append tiny chunk to existing buffer
                buffer += " " + s
            else:
                consolidated.append(buffer)
                buffer = s

        if buffer:
            if consolidated and len(buffer) < char_floor:
                consolidated[-1] += " " + buffer
            else:
                consolidated.append(buffer)

        return consolidated
