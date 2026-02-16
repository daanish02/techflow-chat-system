"""Retrieval interface for querying policy documents."""

from typing import List, Tuple, Union

from langchain_core.documents import Document

from src.config import settings
from src.rag.vector_store import get_vector_store
from src.logger import get_logger

logger = get_logger(__name__)


def retrieve_relevant_policies(
    query: str, k: int | None = None, score_threshold: float = 0.0
) -> List[Document]:
    """
    Retrieve relevant policy documents for a query.

    Uses semantic search to find the most relevant policy document chunks
    that can help answer the customer's question or address their concern.

    Args:
        query: The question or topic to search for
        k: Number of results to return (defaults to settings.TOP_K_RESULTS)
        score_threshold: Minimum relevance score (0.0 to 1.0)

    Returns:
        List of relevant Document objects, sorted by relevance

    Example:
        >>> docs = retrieve_relevant_policies("What does Care+ cover?")
        >>> len(docs) <= 3
        True
        >>> "care_plus_benefits" in docs[0].metadata['source']
        True
    """
    if k is None:
        k = settings.TOP_K_RESULTS

    try:
        vector_store = get_vector_store()

        # similarity search with scores
        results_with_scores = vector_store.similarity_search_with_score(query, k=k)

        # filter by score threshold and extract documents
        filtered_docs = [
            doc for doc, score in results_with_scores if score >= score_threshold
        ]

        logger.info(
            f"Retrieved {len(filtered_docs)} relevant documents for query: '{query[:50]}...'"
        )

        return filtered_docs

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []


def retrieve_with_scores(
    query: str, k: int | None = None
) -> List[Tuple[Document, float]]:
    """
    Retrieve documents with their relevance scores.

    Useful for debugging or when you need to see confidence scores.

    Args:
        query: The question or topic to search for
        k: Number of results to return

    Returns:
        List of (Document, score) tuples

    Example:
        >>> results = retrieve_with_scores("return policy")
        >>> doc, score = results[0]
        >>> score > 0.5
        True
    """
    if k is None:
        k = settings.TOP_K_RESULTS

    try:
        vector_store = get_vector_store()
        results = vector_store.similarity_search_with_score(query, k=k)

        logger.info(
            f"Retrieved {len(results)} documents with scores for: '{query[:50]}...'"
        )

        return results

    except Exception as e:
        logger.error(f"Error retrieving documents with scores: {e}")
        return []


def format_retrieved_context(documents: List[Document]) -> str:
    """
    Format retrieved documents into a context string for the LLM.

    Combines multiple document chunks into a readable context with
    source attribution for the agent to use in responses.

    Args:
        documents: List of retrieved Document objects

    Returns:
        Formatted string with policy information

    Example:
        >>> docs = retrieve_relevant_policies("screen repair")
        >>> context = format_retrieved_context(docs)
        >>> "Care+" in context
        True
    """
    if not documents:
        return "No relevant policy information found."

    formatted_parts = []

    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content.strip()

        formatted_parts.append(f"[Source: {source}]\n{content}")

    context = "\n\n---\n\n".join(formatted_parts)

    logger.debug(f"Formatted {len(documents)} documents into context")

    return context


def query_policies(
    query: str, format_context: bool = True
) -> Union[str, List[Document]]:
    """
    High-level function to query policies and get formatted context.

    This is the main function agents should use to get policy information.
    Combines retrieval and formatting in one convenient call.

    Args:
        query: The question or topic to search for
        format_context: Whether to format results into context string

    Returns:
        Formatted context string ready for LLM consumption, or list of documents if format_context is False

    Example:
        >>> context = query_policies("What are the return windows?")
        >>> "30 days" in context or "14 days" in context
        True
    """
    documents = retrieve_relevant_policies(query)

    if not documents:
        logger.warning(f"No documents found for query: {query}")
        return "I don't have specific policy information about that."

    if format_context:
        return format_retrieved_context(documents)

    return documents


def get_policy_by_type(policy_type: str) -> List[Document]:
    """
    Retrieve all chunks from a specific policy document.

    Useful when you need comprehensive information from one policy.

    Args:
        policy_type: Policy type (e.g., 'return_policy', 'care_plus_benefits')

    Returns:
        List of all document chunks from that policy

    Example:
        >>> docs = get_policy_by_type('care_plus_benefits')
        >>> all(doc.metadata['policy_type'] == 'care_plus_benefits' for doc in docs)
        True
    """
    try:
        vector_store = get_vector_store()

        # filter by metadata
        results = vector_store.get(where={"policy_type": policy_type})

        if not results or not results.get("documents"):
            logger.warning(f"No documents found for policy type: {policy_type}")
            return []

        # convert to document objects
        documents = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(results["documents"], results["metadatas"])
        ]

        logger.info(f"Retrieved {len(documents)} chunks for policy: {policy_type}")
        return documents

    except Exception as e:
        logger.error(f"Error retrieving policy by type: {e}")
        return []
