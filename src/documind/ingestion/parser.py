"""Layout-aware PDF parser using PyMuPDF.

Extracts text blocks, images, tables, and page-level metadata from PDFs.
Each page is represented as a `ParsedPage` with categorized content blocks.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pymupdf as fitz  # PyMuPDF
from loguru import logger
from PIL import Image

from documind.config import settings

BlockType = Literal["text", "image", "table"]


@dataclass
class ContentBlock:
    """A single content block extracted from a PDF page."""

    block_type: BlockType
    content: str  # Text content or base64 image placeholder path
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_number: int
    block_index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedPage:
    """All extracted content from a single PDF page."""

    page_number: int
    width: float
    height: float
    blocks: list[ContentBlock] = field(default_factory=list)
    has_images: bool = False
    is_scanned: bool = False  # True if page appears to be a scan (very little text)


@dataclass
class ParsedDocument:
    """Complete parsed representation of a PDF document."""

    filename: str
    total_pages: int
    pages: list[ParsedPage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class PDFParser:
    """Layout-aware PDF parser.

    Uses PyMuPDF to extract text blocks with bounding boxes,
    detect images, and identify scanned pages that need OCR.
    """

    # If a page has fewer characters than this, it's likely scanned
    SCANNED_PAGE_CHAR_THRESHOLD = 50

    # Minimum image area (relative to page) to be worth extracting
    MIN_IMAGE_AREA_RATIO = 0.02

    def __init__(self, image_output_dir: Path | None = None):
        """Initialize parser.

        Args:
            image_output_dir: Directory to save extracted images.
                              If None, images are kept in memory only.
        """
        self.image_output_dir = image_output_dir
        if image_output_dir:
            image_output_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """Parse a PDF file and extract all content blocks.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ParsedDocument with all pages and their content blocks.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Parsing PDF: {pdf_path.name}")

        doc = fitz.open(str(pdf_path))
        total_pages_in_file = len(doc)

        # Respect the configured max_pages limit
        max_pages = getattr(settings, "max_pages", 500)
        pages_to_parse = min(total_pages_in_file, max_pages)
        if pages_to_parse < total_pages_in_file:
            logger.warning(
                f"PDF has {total_pages_in_file} pages but max_pages={max_pages}. "
                f"Only parsing first {pages_to_parse} pages."
            )

        parsed_doc = ParsedDocument(
            filename=pdf_path.name,
            total_pages=pages_to_parse,
            metadata={
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "page_count": total_pages_in_file,
                "pages_parsed": pages_to_parse,
                "truncated": pages_to_parse < total_pages_in_file,
            },
        )

        for page_num in range(pages_to_parse):
            page = doc[page_num]
            parsed_page = self._parse_page(page, page_num + 1, pdf_path.stem)
            parsed_doc.pages.append(parsed_page)

        doc.close()

        total_blocks = sum(len(p.blocks) for p in parsed_doc.pages)
        scanned_pages = sum(1 for p in parsed_doc.pages if p.is_scanned)
        logger.info(
            f"Parsed {pdf_path.name}: {parsed_doc.total_pages} pages, "
            f"{total_blocks} blocks, {scanned_pages} scanned pages"
        )

        return parsed_doc

    def _parse_page(
        self,
        page: fitz.Page,
        page_number: int,
        doc_stem: str,
    ) -> ParsedPage:
        """Parse a single PDF page.

        Extracts text blocks and images, detects if page is scanned.
        """
        rect = page.rect
        parsed_page = ParsedPage(
            page_number=page_number,
            width=rect.width,
            height=rect.height,
        )

        page_area = rect.width * rect.height
        block_index = 0

        # ── Extract text and image blocks via get_text("dict") ──
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        total_chars = 0
        for block in page_dict.get("blocks", []):
            if block["type"] == 0:  # Text block
                text = self._extract_text_from_block(block)
                if text.strip():
                    total_chars += len(text.strip())
                    content_block = ContentBlock(
                        block_type="text",
                        content=text.strip(),
                        bbox=(
                            block["bbox"][0],
                            block["bbox"][1],
                            block["bbox"][2],
                            block["bbox"][3],
                        ),
                        page_number=page_number,
                        block_index=block_index,
                        metadata={
                            "font_sizes": self._get_font_sizes(block),
                            "is_bold": self._detect_bold(block),
                        },
                    )
                    parsed_page.blocks.append(content_block)
                    block_index += 1

            elif block["type"] == 1:  # Image block
                bbox = block["bbox"]
                img_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                if img_area / page_area >= self.MIN_IMAGE_AREA_RATIO:
                    parsed_page.has_images = True
                    image_path = self._save_image(
                        block, page_number, block_index, doc_stem
                    )
                    content_block = ContentBlock(
                        block_type="image",
                        content=image_path or "[image]",
                        bbox=tuple(bbox),
                        page_number=page_number,
                        block_index=block_index,
                        metadata={
                            "width": block.get("width", 0),
                            "height": block.get("height", 0),
                            "image_path": image_path,
                        },
                    )
                    parsed_page.blocks.append(content_block)
                    block_index += 1

        # ── Detect tables via heuristic (lines of structured data) ──
        tables = self._detect_tables(page, page_number, block_index)
        parsed_page.blocks.extend(tables)

        # ── Mark as scanned if very little text ──
        if total_chars < self.SCANNED_PAGE_CHAR_THRESHOLD:
            parsed_page.is_scanned = True

        return parsed_page

    def _extract_text_from_block(self, block: dict) -> str:
        """Extract all text from a text block's lines and spans."""
        lines = []
        for line in block.get("lines", []):
            spans_text = "".join(span["text"] for span in line.get("spans", []))
            if spans_text.strip():
                lines.append(spans_text)
        return "\n".join(lines)

    def _get_font_sizes(self, block: dict) -> list[float]:
        """Get unique font sizes in a text block."""
        sizes = set()
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sizes.add(round(span.get("size", 0), 1))
        return sorted(sizes)

    def _detect_bold(self, block: dict) -> bool:
        """Check if any span in the block uses a bold font."""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                flags = span.get("flags", 0)
                # Bit 4 (value 16) = bold in PyMuPDF font flags
                if flags & 16:
                    return True
                font_name = span.get("font", "").lower()
                if "bold" in font_name:
                    return True
        return False

    def _save_image(
        self,
        block: dict,
        page_number: int,
        block_index: int,
        doc_stem: str,
    ) -> str | None:
        """Save an extracted image to disk if output dir is set."""
        if not self.image_output_dir:
            return None

        try:
            image_bytes = block.get("image")
            if not image_bytes:
                return None

            img = Image.open(io.BytesIO(image_bytes))
            filename = f"{doc_stem}_p{page_number}_b{block_index}.png"
            save_path = self.image_output_dir / filename
            img.save(str(save_path))
            return str(save_path)
        except Exception as e:
            logger.warning(f"Failed to save image (p{page_number}, b{block_index}): {e}")
            return None

    def _detect_tables(
        self,
        page: fitz.Page,
        page_number: int,
        start_block_index: int,
    ) -> list[ContentBlock]:
        """Detect tables on a page using PyMuPDF's built-in table finder.

        Falls back to empty list if no tables or on error.
        """
        table_blocks = []
        try:
            tables = page.find_tables()
            for i, table in enumerate(tables):
                # Extract table as a list of lists
                table_data = table.extract()
                if not table_data:
                    continue

                # Convert to markdown-style text for embedding
                markdown_table = self._table_to_markdown(table_data)
                if markdown_table.strip():
                    block = ContentBlock(
                        block_type="table",
                        content=markdown_table,
                        bbox=tuple(table.bbox),
                        page_number=page_number,
                        block_index=start_block_index + i,
                        metadata={
                            "rows": len(table_data),
                            "cols": len(table_data[0]) if table_data else 0,
                        },
                    )
                    table_blocks.append(block)
        except Exception as e:
            logger.debug(f"Table detection failed on page {page_number}: {e}")

        return table_blocks

    def _table_to_markdown(self, table_data: list[list[str | None]]) -> str:
        """Convert a 2D table to markdown format."""
        if not table_data:
            return ""

        # Clean cells
        cleaned = []
        for row in table_data:
            cleaned.append([str(cell).strip() if cell else "" for cell in row])

        # Build markdown
        lines = []
        # Header row
        lines.append("| " + " | ".join(cleaned[0]) + " |")
        lines.append("| " + " | ".join("---" for _ in cleaned[0]) + " |")
        # Data rows
        for row in cleaned[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
