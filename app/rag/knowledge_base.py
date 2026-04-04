"""RAG knowledge base using ChromaDB for vector search."""

import logging

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.rag.loader import load_documents

logger = logging.getLogger(__name__)

# Multilingual embedding model — works well for kk/ru/en
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class KnowledgeBase:
    """Vector-based knowledge base for RAG retrieval."""

    def __init__(self):
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Load documents and build the vector index."""
        if self._initialized:
            return

        logger.info("Initializing knowledge base...")

        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
        )

        self._client = chromadb.Client()  # In-memory for simplicity
        self._collection = self._client.get_or_create_collection(
            name="kbtu_knowledge",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        documents = load_documents()

        if not documents:
            logger.warning("No documents found in knowledge base!")
            self._initialized = True
            return

        # Add documents to ChromaDB
        self._collection.add(
            ids=[doc["chunk_id"] for doc in documents],
            documents=[doc["text"] for doc in documents],
            metadatas=[{"source": doc["source"]} for doc in documents],
        )

        logger.info("Knowledge base initialized with %d chunks", len(documents))
        self._initialized = True

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context for a query.

        Args:
            query: User's question text.
            top_k: Number of top results to return.

        Returns:
            Concatenated relevant text chunks.
        """
        if not self._initialized:
            self.initialize()

        if self._collection is None or self._collection.count() == 0:
            return ""

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        # Join retrieved chunks with separator
        chunks = results["documents"][0]
        context = "\n---\n".join(chunks)

        logger.info("RAG retrieved %d chunks for query: '%s'", len(chunks), query[:60])
        return context


# Singleton
knowledge_base = KnowledgeBase()
