"""RAG system components."""

from src.rag.document_loader import (
    chunk_documents,
    load_and_chunk_policies,
    load_policy_documents,
)
from src.rag.retriever import (
    format_retrieved_context,
    get_policy_by_type,
    query_policies,
    retrieve_relevant_policies,
    retrieve_with_scores,
)
from src.rag.vector_store import (
    add_documents_to_store,
    get_embeddings,
    get_vector_store,
    initialize_vector_store,
    reset_vector_store,
)

__all__ = [
    # document loading
    "load_policy_documents",
    "chunk_documents",
    "load_and_chunk_policies",
    # vector store
    "get_embeddings",
    "initialize_vector_store",
    "add_documents_to_store",
    "get_vector_store",
    "reset_vector_store",
    # retrieval
    "retrieve_relevant_policies",
    "retrieve_with_scores",
    "format_retrieved_context",
    "query_policies",
    "get_policy_by_type",
]
