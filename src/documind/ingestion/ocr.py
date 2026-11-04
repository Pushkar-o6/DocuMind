"""OCR module for scanned PDF pages.

Uses pytesseract to extract text from page images when
the layout parser detects a scanned (image-only) page.
"""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf as fitz  # PyMuPDF
from loguru import logger
from PIL import Image

from documind.config import settings

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning(
        "pytesseract not available. OCR will use Gemini Cloud or be skipped. "
        "Install Tesseract OCR: https://github.com/tesseract-ocr/tesseract"
    )


def check_tesseract_installed() -> bool:
    """Check if Tesseract OCR binary is installed and accessible."""
    if not TESSERACT_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_page_gemini(page: fitz.Page, dpi: int = 150) -> str:
    """Run OCR on a scanned page using Gemini Multimodal Vision API in the cloud."""
    import google.generativeai as genai

    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")

    model = genai.GenerativeModel(settings.gemini_llm_model)
    prompt = (
        "Extract all readable text, tables, and structured data from this scanned document page. "
        "Format tables in markdown. Output only the extracted content without preamble."
    )
    response = model.generate_content([
        prompt,
        {"mime_type": "image/png", "data": img_bytes},
    ])
    return response.text.strip()


def ocr_page(page: fitz.Page, dpi: int = 150, lang: str = "eng") -> str:
    """Run OCR on a single PDF page.

    Prefers Gemini Cloud Vision when configured (fast cloud extraction, 0% CPU).
    Falls back to local Tesseract OCR if configured for Ollama / local.

    Args:
        page: A PyMuPDF page object.
        dpi: Resolution for rendering.
        lang: Tesseract language code.

    Returns:
        Extracted text string. Empty string if OCR fails or is unavailable.
    """
    if settings.using_gemini and settings.gemini_api_key:
        try:
            logger.info(f"Running Gemini Cloud OCR on page {page.number + 1}...")
            text = _ocr_page_gemini(page, dpi=dpi)
            if text:
                logger.info(f"Gemini Cloud OCR p{page.number + 1}: extracted {len(text)} chars")
                return text
        except Exception as e:
            logger.warning(
                f"Gemini Cloud OCR failed on p{page.number + 1}: {e}. Falling back to local OCR..."
            )

    if not TESSERACT_AVAILABLE:
        logger.warning("Skipping local OCR — pytesseract not installed")
        return ""

    if not check_tesseract_installed():
        logger.warning("Skipping local OCR — Tesseract binary not found")
        return ""

    try:
        # Render page to image at specified DPI
        zoom = dpi / 72  # 72 DPI is PyMuPDF's default
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))

        # Run OCR
        text = pytesseract.image_to_string(image, lang=lang)

        logger.debug(
            f"OCR page {page.number + 1}: extracted {len(text)} chars"
        )
        return text.strip()

    except Exception as e:
        logger.error(f"OCR failed on page {page.number + 1}: {e}")
        return ""


def ocr_pdf_pages(
    pdf_path: str | Path,
    page_numbers: list[int] | None = None,
    dpi: int = 300,
    lang: str = "eng",
) -> dict[int, str]:
    """Run OCR on specific pages (or all pages) of a PDF.

    Args:
        pdf_path: Path to the PDF file.
        page_numbers: 1-indexed page numbers to OCR. None = all pages.
        dpi: OCR rendering resolution.
        lang: Tesseract language code.

    Returns:
        Dict mapping page number (1-indexed) to extracted text.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    if page_numbers is None:
        page_numbers = list(range(1, len(doc) + 1))

    results = {}
    for page_num in page_numbers:
        if 1 <= page_num <= len(doc):
            page = doc[page_num - 1]
            text = ocr_page(page, dpi=dpi, lang=lang)
            if text:
                results[page_num] = text
            logger.info(f"OCR page {page_num}/{len(doc)}: {len(text)} chars")
        else:
            logger.warning(f"Page {page_num} out of range (1-{len(doc)})")

    doc.close()
    return results
