# RAG Orchestration Engine (Module 4 & 5)
# Module 4: Context Retriever — FAISS similarity search with score filtering
# Module 5: RAG Prompt Orchestrator — Gemini 2.0 Flash API integration

import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Coordinates context retrieval from FAISS and generates answers via Gemini.

    This class provides two core capabilities:
    1. Context Retrieval (Module 4): Vectorize user queries, search the
       FAISS index, filter results by similarity threshold, and return
       matching text passages with page citations.
    2. Answer Generation (Module 5): Construct RAG prompts from user
       queries and retrieved context, call the Google Gemini API, and
       return conversational answers with source attribution.

    Attributes:
        vector_store: The FAISS vector store instance for similarity search.
        model: The Gemini generative model instance (initialized lazily).
        api_key (str | None): The Gemini API key loaded from environment.
    """

    # Default RAG system prompt
    SYSTEM_PROMPT = (
        "You are an expert document analysis assistant. Your role is to answer "
        "questions based ONLY on the provided context passages extracted from "
        "uploaded PDF documents.\n\n"
        "Rules:\n"
        "1. Answer the question using ONLY the information in the context below.\n"
        "2. If the context does not contain enough information to answer, say so "
        "clearly — do NOT make up information.\n"
        "3. Cite the source page number(s) for each claim using the format "
        "[Page X].\n"
        "4. Be concise but thorough. Use bullet points for multi-part answers.\n"
        "5. If numbers or data are mentioned in the context, include them in "
        "your answer.\n"
    )

    def __init__(self, vector_store=None):
        """
        Initialize RAGEngine with an optional FAISS vector store.

        Args:
            vector_store: A FAISS vector store instance from VectorManager.
                Can be set later via set_vector_store().
        """
        self.vector_store = vector_store
        self.model = None
        self.api_key = None

        # Load environment variables from config/.env
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            ".env",
        )
        load_dotenv(env_path)
        self.api_key = os.getenv("GEMINI_API_KEY")

    def set_vector_store(self, vector_store) -> None:
        """
        Set or update the FAISS vector store used for retrieval.

        Args:
            vector_store: A FAISS vector store instance.
        """
        self.vector_store = vector_store

    # ──────────────────────────────────────────────
    # Module 4: Context Retrieval
    # ──────────────────────────────────────────────

    def retrieve_context(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 1.5,
    ) -> list[dict]:
        """
        Retrieve the most relevant document chunks for a given query.

        Vectorizes the query string, searches the FAISS index for the
        top-k most similar document chunks, and filters results that
        exceed the distance threshold (lower distance = more similar
        in L2 space).

        Args:
            query: The user's search query string.
            k: Number of top results to retrieve (default: 4).
            score_threshold: Maximum L2 distance to accept. Chunks with
                distance above this value are filtered out. Lower values
                are more restrictive. Default is 1.5 (fairly permissive
                for normalized embeddings where max distance ≈ 2.0).

        Returns:
            A list of result dicts sorted by relevance (best first), each
            containing:
                - 'text' (str): The chunk text content.
                - 'page_number' (int): Source page number.
                - 'source' (str): Source file name.
                - 'chunk_id' (int): The chunk identifier.
                - 'score' (float): L2 distance score (lower is better).
                - 'similarity' (float): Normalized similarity (0-1, higher
                    is better), computed as max(0, 1 - distance/2).

        Raises:
            ValueError: If no vector store is available.
        """
        if self.vector_store is None:
            raise ValueError(
                "No vector store available. Build or load an index first."
            )

        if not query.strip():
            return []

        logger.info("Retrieving context for query: '%s' (k=%d)", query, k)

        try:
            # FAISS similarity_search_with_score returns (Document, score) pairs
            # where score is L2 distance (lower = more similar)
            results_with_scores = (
                self.vector_store.similarity_search_with_score(query, k=k)
            )
        except Exception as e:
            logger.error("FAISS search failed: %s", e)
            return []

        # Filter and format results
        filtered_results = []
        for doc, distance in results_with_scores:
            # Filter out results above the distance threshold
            if distance > score_threshold:
                continue

            # Convert L2 distance to a normalized similarity score (0-1)
            # For normalized embeddings, max L2 distance is 2.0
            similarity = max(0.0, 1.0 - distance / 2.0)

            filtered_results.append({
                "text": doc.page_content,
                "page_number": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", 0),
                "score": round(float(distance), 4),
                "similarity": round(similarity, 4),
            })

        logger.info(
            "Retrieved %d results (%d passed threshold)",
            len(results_with_scores),
            len(filtered_results),
        )

        return filtered_results

    # ──────────────────────────────────────────────
    # Module 5: RAG Prompt Orchestration
    # ──────────────────────────────────────────────

    def _initialize_model(self) -> None:
        """
        Lazily initialize the Gemini generative model.

        Raises:
            ValueError: If GEMINI_API_KEY is not set.
            RuntimeError: If model initialization fails.
        """
        if self.model is not None:
            return

        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Set it in config/.env or as an environment variable."
            )

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction=self.SYSTEM_PROMPT,
            )
            logger.info("Gemini model initialized (gemini-2.0-flash).")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini model: {e}") from e

    def _build_prompt(self, query: str, context_chunks: list[dict]) -> str:
        """
        Build a structured RAG prompt from query and context passages.

        Args:
            query: The user's question.
            context_chunks: List of context dicts from retrieve_context().

        Returns:
            The formatted prompt string ready to send to the LLM.
        """
        if not context_chunks:
            return (
                f"Question: {query}\n\n"
                "No relevant context was found in the uploaded documents. "
                "Please inform the user that their question could not be "
                "answered from the available documents."
            )

        # Build context section with page citations
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            page = chunk.get("page_number", "?")
            source = chunk.get("source", "unknown")
            similarity = chunk.get("similarity", 0)
            context_parts.append(
                f"--- Passage {i} [Source: {source}, Page {page}, "
                f"Relevance: {similarity:.0%}] ---\n"
                f"{chunk['text']}\n"
            )

        context_text = "\n".join(context_parts)

        prompt = (
            f"CONTEXT PASSAGES:\n"
            f"{context_text}\n"
            f"---\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Please answer the question based on the context passages above. "
            f"Cite page numbers using [Page X] format."
        )

        return prompt

    def generate_answer(
        self,
        query: str,
        context_chunks: Optional[list[dict]] = None,
        k: int = 4,
        score_threshold: float = 1.5,
    ) -> dict:
        """
        Generate an AI answer for the user's query using RAG.

        If context_chunks are not provided, retrieves them automatically
        from the FAISS vector store. Then constructs a RAG prompt and
        calls the Gemini API to generate an answer.

        Args:
            query: The user's question string.
            context_chunks: Pre-retrieved context chunks (optional).
                If None, retrieve_context() is called automatically.
            k: Number of context chunks to retrieve (default: 4).
            score_threshold: Maximum L2 distance threshold for retrieval.

        Returns:
            A dict containing:
                - 'answer' (str): The generated answer text.
                - 'context' (list[dict]): The context chunks used.
                - 'query' (str): The original query.
                - 'model' (str): The model name used.
                - 'response_time' (float): Time taken in seconds.
                - 'error' (str | None): Error message if generation failed.

        Raises:
            ValueError: If GEMINI_API_KEY is not configured.
        """
        start_time = time.time()

        result = {
            "answer": "",
            "context": [],
            "query": query,
            "model": "gemini-2.0-flash",
            "response_time": 0.0,
            "error": None,
        }

        # Step 1: Retrieve context if not provided
        if context_chunks is None:
            if self.vector_store is None:
                result["error"] = "No vector store available."
                result["answer"] = (
                    "⚠️ No document index is loaded. Please upload a PDF "
                    "and build the index first."
                )
                result["response_time"] = time.time() - start_time
                return result

            context_chunks = self.retrieve_context(
                query, k=k, score_threshold=score_threshold
            )

        result["context"] = context_chunks

        # Step 2: Initialize Gemini model
        try:
            self._initialize_model()
        except (ValueError, RuntimeError) as e:
            result["error"] = str(e)
            result["answer"] = f"⚠️ Model initialization failed: {e}"
            result["response_time"] = time.time() - start_time
            return result

        # Step 3: Build prompt and generate answer
        prompt = self._build_prompt(query, context_chunks)

        try:
            logger.info("Sending query to Gemini API...")
            response = self.model.generate_content(prompt)
            result["answer"] = response.text
            logger.info("Gemini API response received.")
        except Exception as e:
            logger.error("Gemini API call failed: %s", e)
            result["error"] = str(e)
            result["answer"] = (
                f"⚠️ Failed to generate answer: {e}\n\n"
                "Please check your API key and internet connection."
            )

        result["response_time"] = round(time.time() - start_time, 2)
        return result

    def is_api_configured(self) -> bool:
        """
        Check if the Gemini API key is configured and valid.

        Returns:
            True if an API key is set and not the placeholder value.
        """
        return bool(
            self.api_key and self.api_key != "your_gemini_api_key_here"
        )
