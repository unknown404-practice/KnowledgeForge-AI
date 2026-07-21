"""
PDF Loader extracting text and page numbers using PyMuPDF (fitz) with EasyOCR fallback for scanned pages.
"""

from typing import List, Dict, Any
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

# Lazy load OCR reader singleton to avoid high memory overhead when not needed
_OCR_READER = None


def get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        try:
            import easyocr

            logger.info(
                "Initializing EasyOCR reader (`en`) for scanned PDF extraction..."
            )
            _OCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR: {e}")
            _OCR_READER = False
    return _OCR_READER


class PDFLoader:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)

    def load(self) -> List[Dict[str, Any]]:
        """Load text from PDF and return list of page documents with metadata."""
        documents = []
        try:
            import fitz  # PyMuPDF

            with fitz.open(str(self.file_path)) as doc:
                for i, page in enumerate(doc):
                    text_dict = page.get_text("dict")

                    # 1. Calculate the median font size to represent "body text"
                    sizes = []
                    for block in text_dict.get("blocks", []):
                        if block.get("type") == 0:  # text block
                            for line in block.get("lines", []):
                                for span in line.get("spans", []):
                                    if span.get("text", "").strip():
                                        sizes.append(span.get("size", 10.0))

                    body_size = 10.0
                    if sizes:
                        sizes.sort()
                        body_size = sizes[len(sizes) // 2]

                    # 2. Reconstruct page text and inject Markdown headings for larger fonts
                    full_text = []
                    for block in text_dict.get("blocks", []):
                        if block.get("type") == 0:
                            for line in block.get("lines", []):
                                line_text = ""
                                max_size = 0
                                for span in line.get("spans", []):
                                    t = span.get("text", "")
                                    if t.strip():
                                        line_text += t
                                        max_size = max(max_size, span.get("size", 0))

                                line_text = line_text.strip()
                                if line_text:
                                    # If font is significantly larger than body, classify as a heading
                                    if max_size > body_size * 1.15:
                                        if max_size > body_size * 1.5:
                                            line_text = f"# {line_text}"
                                        elif max_size > body_size * 1.3:
                                            line_text = f"## {line_text}"
                                        else:
                                            line_text = f"### {line_text}"
                                    full_text.append(line_text)

                    text = "\n".join(full_text).strip()

                    # If page text is very short or missing, trigger EasyOCR on rendered page image
                    if len(text) < 20:
                        logger.info(
                            f"Page {i + 1} of {self.file_path.name} has little/no selectable text. Attempting EasyOCR fallback..."
                        )
                        reader = get_ocr_reader()
                        if reader:
                            try:
                                pix = page.get_pixmap(dpi=150)
                                img_bytes = pix.tobytes("png")
                                ocr_results = reader.readtext(img_bytes, detail=0)
                                ocr_text = "\n".join(ocr_results).strip()
                                if ocr_text:
                                    text = f"[OCR Extracted Text]\n{ocr_text}"
                            except Exception as ocr_e:
                                logger.error(f"EasyOCR failed on page {i + 1}: {ocr_e}")

                    if text.strip():
                        documents.append(
                            {
                                "content": text.strip(),
                                "metadata": {
                                    "source": self.file_path.name,
                                    "page": i + 1,
                                    "file_type": "pdf",
                                },
                            }
                        )
        except Exception as e:
            logger.error(
                f"Error reading PDF {self.file_path.name} with PyMuPDF: {e}. Attempting pypdf fallback..."
            )
            try:
                import pypdf

                reader = pypdf.PdfReader(str(self.file_path))
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        documents.append(
                            {
                                "content": text.strip(),
                                "metadata": {
                                    "source": self.file_path.name,
                                    "page": i + 1,
                                    "file_type": "pdf",
                                },
                            }
                        )
            except Exception as e2:
                logger.error(
                    f"Fallback pypdf also failed for {self.file_path.name}: {e2}"
                )
        return documents
