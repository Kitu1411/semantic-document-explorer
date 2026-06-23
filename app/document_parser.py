# Document Parser Module (Milestone 1 & 2)

class DocumentParser:
    """
    Handles PDF loading, text extraction, and recursive text chunking.
    """
    def __init__(self):
        pass

    def parse_pdf(self, pdf_path: str):
        """
        Parses raw text from a PDF file using PyMuPDF.
        """
        raise NotImplementedError("parse_pdf will be implemented in Day 2.")

    def chunk_text(self, document_text: dict, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Splits text recursively while retaining metadata.
        """
        raise NotImplementedError("chunk_text will be implemented in Day 3.")
