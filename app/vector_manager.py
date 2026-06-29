# Vector Indexing Manager Module (Module 3)
# FAISS vector store with Sentence-Transformers embeddings

import os
import logging
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class VectorManager:
    """
    Manages vector embeddings and FAISS index for semantic document search.

    This class provides a complete vector indexing pipeline:
    1. Generate embeddings from text chunks using Sentence-Transformers
    2. Build a FAISS vector store for efficient similarity search
    3. Persist and load the index to/from disk
    4. Report index statistics

    Attributes:
        storage_dir (str): Directory path for persisting FAISS index files.
        model_name (str): HuggingFace model identifier for embedding generation.
        embeddings (HuggingFaceEmbeddings): The embedding model instance.
        vector_store (FAISS | None): The in-memory FAISS vector store.
    """

    def __init__(
        self,
        storage_dir: str = "storage/faiss_index",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize VectorManager with storage path and embedding model.

        Args:
            storage_dir: Directory where FAISS index files (index.faiss,
                index.pkl) will be saved and loaded from.
            model_name: HuggingFace sentence-transformers model name.
                Default is 'all-MiniLM-L6-v2' (384-dim, ~80MB, fast).
        """
        self.storage_dir = storage_dir
        self.model_name = model_name
        self.vector_store: Optional[FAISS] = None

        # Initialize the embedding model
        logger.info("Loading embedding model: %s", model_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully.")

    # ──────────────────────────────────────────────
    # Index Creation
    # ──────────────────────────────────────────────

    def build_index(self, chunks: list[dict]) -> "FAISS":
        """
        Build a FAISS vector store from document chunk dictionaries.

        Converts chunk dicts (from DocumentParser.chunk_text()) into
        LangChain Document objects, generates embeddings, and creates
        an in-memory FAISS index.

        Args:
            chunks: List of chunk dicts, each containing:
                - 'text' (str): The chunk text content.
                - 'page_number' (int): Source page number.
                - 'file_name' (str): Source file name.
                - 'chunk_id' (int): Sequential chunk identifier.
                - 'metadata' (dict): Additional metadata.

        Returns:
            The created FAISS vector store instance.

        Raises:
            ValueError: If chunks list is empty.
            RuntimeError: If embedding generation or index creation fails.
        """
        if not chunks:
            raise ValueError("Cannot build index from an empty chunks list.")

        logger.info("Building FAISS index from %d chunks...", len(chunks))

        # Convert chunk dicts to LangChain Document objects
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata={
                    "source": chunk.get("file_name", "unknown"),
                    "page": chunk.get("page_number", 0),
                    "chunk_id": chunk.get("chunk_id", 0),
                },
            )
            documents.append(doc)

        try:
            # Create FAISS vector store from documents
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings,
            )
            logger.info(
                "FAISS index built successfully with %d vectors.", len(documents)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build FAISS index: {e}") from e

        return self.vector_store

    # ──────────────────────────────────────────────
    # Index Persistence
    # ──────────────────────────────────────────────

    def save_index(self) -> str:
        """
        Save the current FAISS vector store to disk.

        Persists two files to the storage directory:
        - index.faiss: The FAISS index binary
        - index.pkl: Pickled document store (text + metadata)

        Returns:
            The absolute path to the storage directory.

        Raises:
            ValueError: If no vector store has been built yet.
            RuntimeError: If saving to disk fails.
        """
        if self.vector_store is None:
            raise ValueError(
                "No vector store to save. Call build_index() first."
            )

        os.makedirs(self.storage_dir, exist_ok=True)

        try:
            self.vector_store.save_local(self.storage_dir)
            abs_path = os.path.abspath(self.storage_dir)
            logger.info("FAISS index saved to: %s", abs_path)
            return abs_path
        except Exception as e:
            raise RuntimeError(f"Failed to save FAISS index: {e}") from e

    def load_index(self) -> "FAISS":
        """
        Load a previously saved FAISS vector store from disk.

        Reads index.faiss and index.pkl from the storage directory
        and reconstructs the in-memory vector store.

        Returns:
            The loaded FAISS vector store instance.

        Raises:
            FileNotFoundError: If the index files do not exist.
            RuntimeError: If loading from disk fails.
        """
        if not self.index_exists():
            raise FileNotFoundError(
                f"No FAISS index found at: {self.storage_dir}. "
                "Build and save an index first."
            )

        try:
            self.vector_store = FAISS.load_local(
                self.storage_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS index loaded from: %s", self.storage_dir)
            return self.vector_store
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}") from e

    # ──────────────────────────────────────────────
    # Index Utilities
    # ──────────────────────────────────────────────

    def index_exists(self) -> bool:
        """
        Check whether saved FAISS index files exist on disk.

        Returns:
            True if both index.faiss and index.pkl are present.
        """
        faiss_path = os.path.join(self.storage_dir, "index.faiss")
        pkl_path = os.path.join(self.storage_dir, "index.pkl")
        return os.path.exists(faiss_path) and os.path.exists(pkl_path)

    def get_index_stats(self) -> dict:
        """
        Return statistics about the current in-memory FAISS index.

        Returns:
            A dict with:
                - 'total_vectors' (int): Number of indexed document vectors.
                - 'embedding_dimension' (int): Dimensionality of each vector.
                - 'model_name' (str): The embedding model identifier.
                - 'storage_dir' (str): Path to the index storage directory.
                - 'index_on_disk' (bool): Whether a saved index exists.
        """
        stats = {
            "total_vectors": 0,
            "embedding_dimension": 0,
            "model_name": self.model_name,
            "storage_dir": os.path.abspath(self.storage_dir),
            "index_on_disk": self.index_exists(),
        }

        if self.vector_store is not None:
            # Access the underlying FAISS index for stats
            faiss_index = self.vector_store.index
            stats["total_vectors"] = faiss_index.ntotal
            stats["embedding_dimension"] = faiss_index.d

        return stats

    def get_vector_store(self) -> Optional["FAISS"]:
        """
        Return the current in-memory vector store, or None if not loaded.

        Returns:
            The FAISS vector store instance or None.
        """
        return self.vector_store
