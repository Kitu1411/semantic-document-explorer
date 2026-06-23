# Unit Tests for DocumentParser (Day 2 & Day 3)

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.document_parser import DocumentParser
from tests.create_sample_pdf import create_sample_pdf


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a sample PDF in a temp directory for testing."""
    pdf_path = str(tmp_path / "test_report.pdf")
    create_sample_pdf(pdf_path)
    return pdf_path


@pytest.fixture
def parser():
    """Create a DocumentParser instance with default settings."""
    return DocumentParser(chunk_size=1000, chunk_overlap=200)


# ── Day 2: PDF Parsing Tests ──

class TestParsePdf:
    """Tests for parse_pdf method."""

    def test_parse_returns_correct_structure(self, parser, sample_pdf):
        result = parser.parse_pdf(sample_pdf)
        assert "file_name" in result
        assert "file_path" in result
        assert "file_size_kb" in result
        assert "total_pages" in result
        assert "metadata" in result
        assert "pages" in result

    def test_parse_correct_page_count(self, parser, sample_pdf):
        result = parser.parse_pdf(sample_pdf)
        assert result["total_pages"] == 3
        assert len(result["pages"]) == 3

    def test_parse_page_numbers_are_1_indexed(self, parser, sample_pdf):
        result = parser.parse_pdf(sample_pdf)
        page_numbers = [p["page_number"] for p in result["pages"]]
        assert page_numbers == [1, 2, 3]

    def test_parse_extracts_text(self, parser, sample_pdf):
        result = parser.parse_pdf(sample_pdf)
        for page in result["pages"]:
            assert len(page["clean_text"]) > 0
            assert page["char_count"] > 0
            assert page["word_count"] > 0

    def test_parse_extracts_metadata(self, parser, sample_pdf):
        result = parser.parse_pdf(sample_pdf)
        meta = result["metadata"]
        assert meta["title"] == "Quarterly Financial Report Q4 2025"
        assert meta["author"] == "Analytics Division"

    def test_parse_file_not_found_raises(self, parser):
        with pytest.raises(FileNotFoundError):
            parser.parse_pdf("nonexistent_file.pdf")

    def test_parse_invalid_extension_raises(self, parser, tmp_path):
        fake_file = tmp_path / "test.txt"
        fake_file.write_text("not a pdf")
        with pytest.raises(ValueError):
            parser.parse_pdf(str(fake_file))

    def test_clean_text_removes_noise(self, parser):
        raw = "Hello-\nworld\n\n\n\nMultiple   spaces  here"
        cleaned = parser._clean_text(raw)
        assert "Helloworld" in cleaned  # hyphenated line break fixed
        assert "Multiple spaces here" in cleaned  # multiple spaces collapsed
        assert "\n\n\n" not in cleaned  # excessive newlines removed


class TestLayoutProperties:
    """Tests for extract_layout_properties method."""

    def test_layout_returns_all_pages(self, parser, sample_pdf):
        layout = parser.extract_layout_properties(sample_pdf)
        assert len(layout) == 3

    def test_layout_has_dimensions(self, parser, sample_pdf):
        layout = parser.extract_layout_properties(sample_pdf)
        for page in layout:
            assert page["width"] > 0
            assert page["height"] > 0

    def test_layout_has_text_blocks(self, parser, sample_pdf):
        layout = parser.extract_layout_properties(sample_pdf)
        for page in layout:
            assert len(page["text_blocks"]) > 0

    def test_layout_has_fonts(self, parser, sample_pdf):
        layout = parser.extract_layout_properties(sample_pdf)
        for page in layout:
            assert len(page["fonts_used"]) > 0


# ── Day 3: Text Chunking Tests ──

class TestChunkText:
    """Tests for chunk_text method."""

    def test_chunks_are_generated(self, parser, sample_pdf):
        parsed = parser.parse_pdf(sample_pdf)
        chunks = parser.chunk_text(parsed)
        assert len(chunks) > 0

    def test_chunks_have_required_fields(self, parser, sample_pdf):
        parsed = parser.parse_pdf(sample_pdf)
        chunks = parser.chunk_text(parsed)
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "char_count" in chunk
            assert "word_count" in chunk
            assert "page_number" in chunk
            assert "file_name" in chunk
            assert "metadata" in chunk

    def test_chunk_ids_are_sequential(self, parser, sample_pdf):
        parsed = parser.parse_pdf(sample_pdf)
        chunks = parser.chunk_text(parsed)
        ids = [c["chunk_id"] for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_chunks_respect_max_size(self, parser, sample_pdf):
        parsed = parser.parse_pdf(sample_pdf)
        chunks = parser.chunk_text(parsed)
        for chunk in chunks:
            assert chunk["char_count"] <= parser.chunk_size

    def test_smaller_chunk_size_produces_more_chunks(self, parser, sample_pdf):
        parsed = parser.parse_pdf(sample_pdf)
        chunks_default = parser.chunk_text(parsed)
        chunks_small = parser.chunk_text(parsed, chunk_size=500, chunk_overlap=100)
        assert len(chunks_small) > len(chunks_default)

    def test_chunks_preserve_page_metadata(self, parser, sample_pdf):
        parsed = parser.parse_pdf(sample_pdf)
        chunks = parser.chunk_text(parsed)
        pages_in_chunks = set(c["page_number"] for c in chunks)
        assert pages_in_chunks == {1, 2, 3}

    def test_chunk_metadata_has_source_info(self, parser, sample_pdf):
        parsed = parser.parse_pdf(sample_pdf)
        chunks = parser.chunk_text(parsed)
        for chunk in chunks:
            assert chunk["metadata"]["source"] == parsed["file_name"]
            assert chunk["metadata"]["page"] == chunk["page_number"]

    def test_empty_document_returns_no_chunks(self, parser):
        empty_doc = {"file_name": "empty.pdf", "pages": []}
        chunks = parser.chunk_text(empty_doc)
        assert len(chunks) == 0


class TestChunkStatistics:
    """Tests for get_chunk_statistics method."""

    def test_statistics_computed_correctly(self, parser, sample_pdf):
        parsed = parser.parse_pdf(sample_pdf)
        chunks = parser.chunk_text(parsed)
        stats = parser.get_chunk_statistics(chunks)

        assert stats["total_chunks"] == len(chunks)
        assert stats["min_chunk_size"] <= stats["avg_chunk_size"]
        assert stats["avg_chunk_size"] <= stats["max_chunk_size"]
        assert stats["total_characters"] > 0
        assert len(stats["pages_covered"]) == 3

    def test_empty_chunks_statistics(self, parser):
        stats = parser.get_chunk_statistics([])
        assert stats["total_chunks"] == 0
        assert stats["avg_chunk_size"] == 0
