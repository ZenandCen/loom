"""OCR + Vision file processing pipeline.

Handles image and scanned PDF files that the standard text loaders
cannot process. Uses Tesseract for text extraction and an optional
Vision LLM for semantic understanding.
"""

import io
import logging
import os
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Image extensions that need OCR/Vision
IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}

# Max image size for OCR (10MB per image)
MAX_IMAGE_SIZE = 10 * 1024 * 1024
# Max PDF size for full OCR processing (50MB)
MAX_PDF_SIZE = 50 * 1024 * 1024
# Max pages to process in a PDF
MAX_PDF_PAGES = 200


def _get_vision_llm():
    """Get vision LLM instance, or None if not configured."""
    try:
        from utils.models import LLM, get_llm
        return get_llm(LLM.VISION)
    except Exception:
        return None


def ocr_image(image_bytes: bytes, lang: str = "eng+vie") -> str:
    """Run Tesseract OCR on image bytes. Returns extracted text."""
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        # Convert to grayscale for better OCR
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        text = pytesseract.image_to_string(img, lang=lang)
        return text.strip()
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""


def vision_describe(image_bytes: bytes) -> str:
    """Use Vision LLM to describe/understand image content. Returns description."""
    try:
        llm = _get_vision_llm()
        if llm is None:
            return ""

        import base64
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail. If it contains text, transcribe all visible text. "
                            "If it's a diagram, chart, or table, describe its structure and content. "
                            "Be comprehensive and factual. Respond in the same language as the text in the image."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ],
            }
        ]

        result = llm.invoke(message)
        return result.content.strip()
    except Exception as e:
        logger.warning(f"Vision LLM failed: {e}")
        return ""


def _pdf_has_text(pdf_bytes: bytes) -> bool:
    """Check if a PDF has extractable text (not scanned)."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        # Check first 5 pages for any text
        for i in range(min(5, len(reader.pages))):
            text = reader.pages[i].extract_text()
            if text and len(text.strip()) > 20:
                return True
        return False
    except Exception:
        return False


def _ocr_pdf_pages(pdf_bytes: bytes, lang: str = "eng+vie") -> list[str]:
    """Convert PDF pages to images and run OCR on each. Returns list of page texts."""
    try:
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(pdf_bytes, dpi=200)
    except Exception as e:
        logger.warning(f"pdf2image failed: {e}")
        return []

    page_texts = []
    for i, page in enumerate(pages[:MAX_PDF_PAGES]):
        if i % 10 == 0:
            logger.info(f"OCR page {i + 1}/{len(pages)}")
        img_buffer = io.BytesIO()
        page.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        text = ocr_image(img_buffer.read(), lang=lang)
        page_texts.append(text)
    return page_texts


def process_file(file_bytes: bytes, filename: str, mimetype: str = "") -> list[Document]:
    """
    Process a file and return LangChain Documents.

    Handles:
    - Images (png/jpg/jpeg/webp/bmp/tiff/gif): Tesseract OCR + Vision LLM
    - PDF with text: PyPDFLoader (standard)
    - PDF scanned (no text): pdf2image → per-page OCR + Vision
    - Other formats (docx, csv, xlsx, md, txt, html): delegated to standard loaders

    Args:
        file_bytes: Raw file content
        filename: Original filename (used for metadata)
        mimetype: MIME type from Slack (e.g., "image/png", "application/pdf")

    Returns:
        List of LangChain Documents ready for chunking and indexing.
    """
    ext = Path(filename).suffix.lower()
    docs: list[Document] = []

    # ── Image files: OCR + Vision ──
    if ext in IMAGE_EXTENSIONS or mimetype.startswith("image/"):
        if len(file_bytes) > MAX_IMAGE_SIZE:
            logger.warning(f"Image too large for OCR: {filename} ({len(file_bytes)} bytes)")
            return []

        logger.info(f"Processing image: {filename} ({len(file_bytes)} bytes)")
        ocr_text = ocr_image(file_bytes)
        vision_text = vision_describe(file_bytes)

        content_parts = []
        if ocr_text:
            content_parts.append(f"[OCR Text]\n{ocr_text}")
        if vision_text:
            content_parts.append(f"[Vision Description]\n{vision_text}")

        if not content_parts:
            logger.warning(f"No extractable content from image: {filename}")
            return [Document(
                page_content=f"[Image: {filename}] (no text extracted)",
                metadata={"source": filename, "type": "image", "size": len(file_bytes)},
            )]

        content = "\n\n".join(content_parts)
        docs.append(Document(
            page_content=content,
            metadata={
                "source": filename,
                "type": "image",
                "mimetype": mimetype,
                "size": len(file_bytes),
                "has_ocr": bool(ocr_text),
                "has_vision": bool(vision_text),
            },
        ))
        return docs

    # ── PDF files ──
    if ext == ".pdf" or mimetype == "application/pdf":
        if len(file_bytes) > MAX_PDF_SIZE:
            logger.warning(f"PDF too large for OCR: {filename} ({len(file_bytes)} bytes)")
            # Fall through to standard PyPDFLoader (may still work for text PDFs)

        if _pdf_has_text(file_bytes):
            # Standard text PDF — use PyPDFLoader
            logger.info(f"Processing text PDF: {filename}")
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(io.BytesIO(file_bytes))
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = filename
                doc.metadata["type"] = "pdf_text"
            return documents
        else:
            # Scanned PDF — use OCR
            logger.info(f"Processing scanned PDF: {filename} ({len(file_bytes)} bytes)")
            page_texts = _ocr_pdf_pages(file_bytes)

            if not page_texts or all(not t.strip() for t in page_texts):
                logger.warning(f"No text extracted from scanned PDF: {filename}")
                return [Document(
                    page_content=f"[Scanned PDF: {filename}] (no text extracted)",
                    metadata={"source": filename, "type": "pdf_scanned", "size": len(file_bytes)},
                )]

            for i, text in enumerate(page_texts):
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "type": "pdf_scanned",
                            "page": i + 1,
                            "size": len(file_bytes),
                        },
                    ))
            return docs

    # ── Other text-based files: delegate to standard loader ──
    # These are handled by the existing load_document() pipeline
    # We create a temp file and use the standard loader
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        from rag.indexing import load_document
        file_path = Path(tmp_path)
        documents = load_document(file_path)
        if documents:
            for doc in documents:
                doc.metadata["source"] = filename
                if "type" not in doc.metadata:
                    doc.metadata["type"] = ext.lstrip(".")
            return documents
        return []
    except Exception as e:
        logger.warning(f"Failed to load {filename}: {e}")
        return []
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
