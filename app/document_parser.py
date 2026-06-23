# Document Parser Module (Milestone 1 & 2)
# Day 2: PDF Text Parsing with PyMuPDF
# Day 3: Recursive Text Chunking with LangChain

import fitz  # PyMuPDF
import os
import re
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentParser:
    """
    Handles PDF loading, text extraction, and recursive text chunking.

    This class provides a complete document ingestion pipeline:
    1. Parse raw text from PDF files using PyMuPDF (fitz)
    2. Extract layout properties (fonts, tables, coordinates)
    3. Clean and normalize extracted text (noise handling)
    4. Split text into overlapping chunks while retaining page metadata

    Attributes:
        chunk_size (int): Maximum character count per chunk (default: 1000).
        chunk_overlap (int): Number of overlapping characters between
            consecutive chunks for context continuity (default: 200).
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize DocumentParser with chunking configuration.

        Args:
            chunk_size: Maximum number of characters per text chunk.
            chunk_overlap: Number of characters to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize the LangChain recursive text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False,
        )

    # ──────────────────────────────────────────────
    # Day 2: PDF Text Parsing
    # ──────────────────────────────────────────────

    def parse_pdf(self, pdf_path: str) -> dict:
        """
        Parse a PDF file and extract text content with metadata.

        Uses PyMuPDF (fitz) to extract raw text from each page along with
        document-level metadata such as page count, file name, and file size.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Returns:
            A dictionary containing:
                - 'file_name' (str): Base name of the PDF file.
                - 'file_path' (str): Absolute path to the PDF file.
                - 'file_size_kb' (float): File size in kilobytes.
                - 'total_pages' (int): Number of pages in the document.
                - 'metadata' (dict): PDF document metadata (title, author, etc.).
                - 'pages' (list[dict]): List of page data dicts, each containing:
                    - 'page_number' (int): 1-indexed page number.
                    - 'raw_text' (str): Raw extracted text from the page.
                    - 'clean_text' (str): Cleaned/normalized text.
                    - 'char_count' (int): Character count of clean text.
                    - 'word_count' (int): Word count of clean text.
                    - 'has_images' (bool): Whether the page contains images.
                    - 'image_count' (int): Number of images on the page.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a valid PDF.
            RuntimeError: If PyMuPDF fails to open/read the file.
        """
        # Validate file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Validate file extension
        if not pdf_path.lower().endswith(".pdf"):
            raise ValueError(f"File is not a PDF: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF '{pdf_path}': {e}")

        abs_path = os.path.abspath(pdf_path)
        file_size_kb = round(os.path.getsize(abs_path) / 1024, 2)

        # Extract document-level metadata
        doc_metadata = doc.metadata or {}

        pages_data = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            raw_text = page.get_text("text")
            clean_text = self._clean_text(raw_text)

            # Detect images on the page
            image_list = page.get_images(full=True)

            page_info = {
                "page_number": page_idx + 1,  # 1-indexed for user display
                "raw_text": raw_text,
                "clean_text": clean_text,
                "char_count": len(clean_text),
                "word_count": len(clean_text.split()) if clean_text.strip() else 0,
                "has_images": len(image_list) > 0,
                "image_count": len(image_list),
            }
            pages_data.append(page_info)

        result = {
            "file_name": os.path.basename(pdf_path),
            "file_path": abs_path,
            "file_size_kb": file_size_kb,
            "total_pages": len(doc),
            "metadata": {
                "title": doc_metadata.get("title", ""),
                "author": doc_metadata.get("author", ""),
                "subject": doc_metadata.get("subject", ""),
                "creator": doc_metadata.get("creator", ""),
                "producer": doc_metadata.get("producer", ""),
                "creation_date": doc_metadata.get("creationDate", ""),
                "modification_date": doc_metadata.get("modDate", ""),
            },
            "pages": pages_data,
        }

        doc.close()
        return result

    def extract_layout_properties(self, pdf_path: str) -> list[dict]:
        """
        Extract detailed layout properties from each page including
        font information and text block coordinates.

        This provides deeper structural analysis useful for understanding
        document formatting — headings, body text, footers, etc.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A list of dicts per page, each containing:
                - 'page_number' (int): 1-indexed page number.
                - 'width' (float): Page width in points.
                - 'height' (float): Page height in points.
                - 'text_blocks' (list[dict]): Positioned text blocks with
                    x0, y0, x1, y1 coordinates, text content, and block_type.
                - 'fonts_used' (list[str]): Unique font names found on the page.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        layout_data = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            rect = page.rect

            # Extract text blocks with coordinates
            # Each block: (x0, y0, x1, y1, "text", block_no, block_type)
            blocks = page.get_text("blocks")
            text_blocks = []
            for block in blocks:
                if block[6] == 0:  # block_type 0 = text block
                    text_blocks.append({
                        "x0": round(block[0], 2),
                        "y0": round(block[1], 2),
                        "x1": round(block[2], 2),
                        "y1": round(block[3], 2),
                        "text": block[4].strip(),
                        "block_number": block[5],
                        "block_type": "text",
                    })

            # Extract unique font names from the page
            fonts_used = set()
            text_dict = page.get_text("dict")
            for b in text_dict.get("blocks", []):
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        fonts_used.add(span.get("font", "unknown"))

            layout_data.append({
                "page_number": page_idx + 1,
                "width": round(rect.width, 2),
                "height": round(rect.height, 2),
                "text_blocks": text_blocks,
                "fonts_used": sorted(fonts_used),
            })

        doc.close()
        return layout_data

    def extract_tables_text(self, pdf_path: str) -> list[dict]:
        """
        Attempt to extract tabular data from PDF pages.

        Uses PyMuPDF's table finder to detect and extract tables.
        Falls back to coordinate-based text block grouping if the
        built-in table finder is not available.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A list of dicts per page containing detected tables:
                - 'page_number' (int): 1-indexed page number.
                - 'tables' (list[list[list[str]]]): Nested lists representing
                    rows and cells of each detected table.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        tables_data = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_tables = []

            # Try PyMuPDF's built-in table finder (available in recent versions)
            try:
                tabs = page.find_tables()
                for table in tabs:
                    extracted = table.extract()
                    # Clean None values in cells
                    cleaned_table = [
                        [cell if cell is not None else "" for cell in row]
                        for row in extracted
                    ]
                    page_tables.append(cleaned_table)
            except AttributeError:
                # Fallback: table finder not available in this PyMuPDF version
                pass

            tables_data.append({
                "page_number": page_idx + 1,
                "tables": page_tables,
            })

        doc.close()
        return tables_data

    # ──────────────────────────────────────────────
    # Day 3: Text Chunking
    # ──────────────────────────────────────────────

    def chunk_text(
        self,
        parsed_document: dict,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> list[dict]:
        """
        Split parsed document text into overlapping chunks while retaining
        page metadata for source citation.

        Uses LangChain's RecursiveCharacterTextSplitter to intelligently
        split text at natural boundaries (paragraphs, sentences, words)
        while maintaining context via overlapping character windows.

        Args:
            parsed_document: Output dict from parse_pdf() containing
                'pages' with 'clean_text' and 'page_number' fields.
            chunk_size: Override the default chunk size (optional).
            chunk_overlap: Override the default chunk overlap (optional).

        Returns:
            A list of chunk dicts, each containing:
                - 'chunk_id' (int): Sequential chunk index (0-based).
                - 'text' (str): The chunk text content.
                - 'char_count' (int): Character count of the chunk.
                - 'word_count' (int): Word count of the chunk.
                - 'page_number' (int): Source page number (1-indexed).
                - 'file_name' (str): Source PDF file name.
                - 'metadata' (dict): Combined metadata for LangChain compatibility.
        """
        # Allow per-call override of chunk parameters
        if chunk_size is not None or chunk_overlap is not None:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size or self.chunk_size,
                chunk_overlap=chunk_overlap or self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
                is_separator_regex=False,
            )
        else:
            splitter = self.text_splitter

        file_name = parsed_document.get("file_name", "unknown")
        chunks = []
        chunk_id = 0

        for page_data in parsed_document.get("pages", []):
            page_text = page_data.get("clean_text", "")
            page_number = page_data.get("page_number", 0)

            # Skip pages with no meaningful text
            if not page_text.strip():
                continue

            # Split this page's text into chunks
            page_chunks = splitter.split_text(page_text)

            for chunk_text in page_chunks:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "page_number": page_number,
                    "file_name": file_name,
                    "metadata": {
                        "source": file_name,
                        "page": page_number,
                        "chunk_id": chunk_id,
                    },
                })
                chunk_id += 1

        return chunks

    def get_chunk_statistics(self, chunks: list[dict]) -> dict:
        """
        Compute summary statistics about the generated chunks.

        Args:
            chunks: List of chunk dicts from chunk_text().

        Returns:
            A dict with statistics:
                - 'total_chunks' (int): Total number of chunks generated.
                - 'avg_chunk_size' (float): Average character count per chunk.
                - 'min_chunk_size' (int): Smallest chunk character count.
                - 'max_chunk_size' (int): Largest chunk character count.
                - 'total_characters' (int): Sum of all chunk characters.
                - 'pages_covered' (list[int]): Unique page numbers represented.
                - 'chunks_per_page' (dict): Count of chunks per page number.
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "total_characters": 0,
                "pages_covered": [],
                "chunks_per_page": {},
            }

        char_counts = [c["char_count"] for c in chunks]
        pages = [c["page_number"] for c in chunks]

        # Count chunks per page
        chunks_per_page = {}
        for p in pages:
            chunks_per_page[p] = chunks_per_page.get(p, 0) + 1

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": round(sum(char_counts) / len(char_counts), 1),
            "min_chunk_size": min(char_counts),
            "max_chunk_size": max(char_counts),
            "total_characters": sum(char_counts),
            "pages_covered": sorted(set(pages)),
            "chunks_per_page": dict(sorted(chunks_per_page.items())),
        }

    # ──────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _clean_text(raw_text: str) -> str:
        """
        Clean and normalize raw extracted PDF text.

        Handles common PDF extraction noise:
        - Excessive whitespace and blank lines
        - Page header/footer artifacts
        - Special Unicode characters
        - Hyphenated line breaks (word-\nwrap)

        Args:
            raw_text: Raw text string from PyMuPDF extraction.

        Returns:
            Cleaned and normalized text string.
        """
        if not raw_text:
            return ""

        text = raw_text

        # Fix hyphenated line breaks (e.g., "docu-\nment" -> "document")
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # Replace multiple consecutive newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Replace multiple spaces with single space
        text = re.sub(r"[ \t]{2,}", " ", text)

        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]

        # Filter out very short lines that are likely noise
        # (e.g., lone page numbers, single characters)
        cleaned_lines = []
        for line in lines:
            # Keep blank lines (paragraph separators) and meaningful lines
            if line == "" or len(line) > 2:
                cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)

        # Remove leading/trailing whitespace from the entire text
        text = text.strip()

        return text
