# RAG Orchestration Engine (Milestone 3 & 4)

class RAGEngine:
    """
    Coordinates query retrieval from the FAISS database and builds contextual LLM prompts.
    """
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve_context(self, query: str, k: int = 3):
        """
        Retrieves the top k most similar document chunks.
        """
        raise NotImplementedError("retrieve_context will be implemented in Day 6.")

    def generate_answer(self, query: str, context: str):
        """
        Calls Gemini API with prompt instructions and retrieved context.
        """
        raise NotImplementedError("generate_answer will be implemented in Day 7.")
