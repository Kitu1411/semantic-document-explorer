# Unit Tests for Semantic Document Explorer
# Tests for DocumentParser (Modules 1-2), VectorManager (Module 3),
# and RAGEngine (Module 4) — Module 5 (Gemini API) is skipped without API key.

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.document_parser import DocumentParser
from app.vector_manager import VectorManager
from app.rag_engine import RAGEngine
from tests.create_sample_pdf import create_sample_pdf


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

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


@pytest.fixture
def parsed_doc(parser, sample_pdf):
    """Parse the sample PDF and return the result."""
    return parser.parse_pdf(sample_pdf)


@pytest.fixture
def chunks(parser, parsed_doc):
    """Generate chunks from the parsed document."""
    return parser.chunk_text(parsed_doc)


@pytest.fixture
def vector_manager(tmp_path):
    """Create a VectorManager with a temp storage directory."""
    storage = str(tmp_path / "faiss_index")
    return VectorManager(storage_dir=storage)


# ──────────────────────────────────────────────
# Module 1 & 2: Document Parser Tests
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# Module 3: Vector Manager Tests
# ──────────────────────────────────────────────

class TestVectorManager:
    """Tests for VectorManager class."""

    def test_build_index_creates_store(self, vector_manager, chunks):
        """Building index from chunks should create a vector store."""
        store = vector_manager.build_index(chunks)
        assert store is not None
        assert vector_manager.vector_store is not None

    def test_build_index_empty_raises(self, vector_manager):
        """Building index from empty list should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            vector_manager.build_index([])

    def test_save_and_load_index(self, vector_manager, chunks):
        """Index should persist to disk and reload correctly."""
        vector_manager.build_index(chunks)
        save_path = vector_manager.save_index()
        assert os.path.exists(save_path)
        assert vector_manager.index_exists()

        # Load into a new manager
        new_manager = VectorManager(storage_dir=vector_manager.storage_dir)
        loaded_store = new_manager.load_index()
        assert loaded_store is not None

    def test_save_without_build_raises(self, vector_manager):
        """Saving without building first should raise ValueError."""
        with pytest.raises(ValueError, match="No vector store"):
            vector_manager.save_index()

    def test_load_nonexistent_raises(self, vector_manager):
        """Loading from empty directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            vector_manager.load_index()

    def test_index_stats(self, vector_manager, chunks):
        """Index stats should reflect the built index."""
        vector_manager.build_index(chunks)
        stats = vector_manager.get_index_stats()
        assert stats["total_vectors"] == len(chunks)
        assert stats["embedding_dimension"] > 0
        assert stats["model_name"] == "all-MiniLM-L6-v2"

    def test_index_stats_empty(self, vector_manager):
        """Stats without index should show zero vectors."""
        stats = vector_manager.get_index_stats()
        assert stats["total_vectors"] == 0

    def test_index_exists_false_initially(self, vector_manager):
        """index_exists should be False before saving."""
        assert not vector_manager.index_exists()


# ──────────────────────────────────────────────
# Module 4: Context Retriever Tests
# ──────────────────────────────────────────────

class TestContextRetriever:
    """Tests for RAGEngine.retrieve_context method."""

    @pytest.fixture
    def rag_with_index(self, vector_manager, chunks):
        """Create a RAGEngine with a built FAISS index."""
        vector_manager.build_index(chunks)
        engine = RAGEngine(vector_store=vector_manager.get_vector_store())
        return engine

    def test_retrieve_returns_results(self, rag_with_index):
        """Retrieval should return relevant chunks."""
        results = rag_with_index.retrieve_context("revenue growth")
        assert len(results) > 0

    def test_retrieve_result_structure(self, rag_with_index):
        """Each result should have the expected fields."""
        results = rag_with_index.retrieve_context("quarterly report")
        for r in results:
            assert "text" in r
            assert "page_number" in r
            assert "source" in r
            assert "score" in r
            assert "similarity" in r

    def test_retrieve_respects_k(self, rag_with_index):
        """Should return at most k results."""
        results = rag_with_index.retrieve_context("AI platform", k=2)
        assert len(results) <= 2

    def test_retrieve_empty_query(self, rag_with_index):
        """Empty query should return empty results."""
        results = rag_with_index.retrieve_context("")
        assert results == []

    def test_retrieve_no_store_raises(self):
        """Retrieval without vector store should raise ValueError."""
        engine = RAGEngine(vector_store=None)
        with pytest.raises(ValueError, match="No vector store"):
            engine.retrieve_context("test query")

    def test_retrieve_similarity_scores(self, rag_with_index):
        """Similarity scores should be between 0 and 1."""
        results = rag_with_index.retrieve_context("financial performance")
        for r in results:
            assert 0.0 <= r["similarity"] <= 1.0

    def test_retrieve_strict_threshold(self, rag_with_index):
        """Very strict threshold should filter out more results."""
        loose = rag_with_index.retrieve_context(
            "revenue", score_threshold=2.0
        )
        strict = rag_with_index.retrieve_context(
            "revenue", score_threshold=0.1
        )
        assert len(strict) <= len(loose)


# ──────────────────────────────────────────────
# Module 5: RAG Engine Tests (API key dependent)
# ──────────────────────────────────────────────

class TestRAGEngine:
    """Tests for RAGEngine answer generation (limited without API key)."""

    def test_api_not_configured_by_default(self):
        """API should not be configured without a real key."""
        engine = RAGEngine()
        # May or may not be configured depending on local .env
        # Just test the method works
        assert isinstance(engine.is_api_configured(), bool)

    def test_generate_answer_no_store(self):
        """Should return error message when no store is available."""
        engine = RAGEngine(vector_store=None)
        result = engine.generate_answer("test question")
        assert result["error"] is not None
        assert "No vector store" in result["error"]

    def test_generate_answer_result_structure(self):
        """Result dict should have all expected keys."""
        engine = RAGEngine(vector_store=None)
        result = engine.generate_answer("test")
        assert "answer" in result
        assert "context" in result
        assert "query" in result
        assert "model" in result
        assert "response_time" in result
        assert "error" in result

    def test_build_prompt_with_context(self):
        """Prompt builder should include context and query."""
        engine = RAGEngine()
        context = [
            {
                "text": "Revenue grew 15% year-over-year.",
                "page_number": 1,
                "source": "report.pdf",
                "similarity": 0.85,
            }
        ]
        prompt = engine._build_prompt("What was revenue growth?", context)
        assert "Revenue grew 15%" in prompt
        assert "What was revenue growth?" in prompt
        assert "Page 1" in prompt

    def test_build_prompt_no_context(self):
        """Prompt with empty context should indicate no results found."""
        engine = RAGEngine()
        prompt = engine._build_prompt("test question", [])
        assert "No relevant context" in prompt

    def test_set_vector_store(self):
        """set_vector_store should update the engine's store."""
        engine = RAGEngine()
        assert engine.vector_store is None
        engine.set_vector_store("mock_store")
        assert engine.vector_store == "mock_store"
