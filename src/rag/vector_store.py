"""Vector store implementation using ChromaDB."""

from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Get the embeddings model for document vectorization.

    Uses Google's Gemini embedding model for converting text to vectors.

    Returns:
        GoogleGenerativeAIEmbeddings instance
    """
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=SecretStr(settings.GEMINI_API_KEY),
    )


def initialize_vector_store(
    documents: Optional[List[Document]] = None, persist_directory: Optional[Path] = None
) -> Chroma:
    """
    Initialize or load the Chroma vector store.

    If documents are provided, creates a new vector store and adds the documents.
    If no documents provided but persisted store exists, loads the existing store.

    Args:
        documents: Optional list of documents to add to new vector store
        persist_directory: Directory for persistence (defaults to settings.CHROMA_PERSIST_DIR)

    Returns:
        Chroma vector store instance

    Example:
        >>> from src.rag.document_loader import load_and_chunk_policies
        >>> chunks = load_and_chunk_policies()
        >>> vector_store = initialize_vector_store(documents=chunks)
        >>> vector_store._collection.count()
        18
    """
    if persist_directory is None:
        persist_directory = settings.CHROMA_PERSIST_DIR

    # check directory exists
    persist_directory.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()

    if documents:
        # create new vector store with documents
        logger.info(f"Creating new vector store with {len(documents)} documents")

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(persist_directory),
            collection_name="policy_documents",
        )

        logger.info(
            f"Vector store created with {vector_store._collection.count()} vectors"
        )

    else:
        # load existing vector store
        logger.info(f"Loading existing vector store from {persist_directory}")

        vector_store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embeddings,
            collection_name="policy_documents",
        )

        count = vector_store._collection.count()
        logger.info(f"Loaded vector store with {count} vectors")

        if count == 0:
            logger.warning("Vector store is empty. Run setup_rag.py to populate it.")

    return vector_store


def add_documents_to_store(vector_store: Chroma, documents: List[Document]) -> None:
    """
    Add new documents to an existing vector store.

    Args:
        vector_store: Existing Chroma vector store
        documents: List of documents to add

    Example:
        >>> vector_store = initialize_vector_store()
        >>> new_docs = [Document(page_content="New policy", metadata={"source": "new.md"})]
        >>> add_documents_to_store(vector_store, new_docs)
    """
    logger.info(f"Adding {len(documents)} documents to vector store")

    vector_store.add_documents(documents)

    logger.info(f"Vector store now contains {vector_store._collection.count()} vectors")


def get_vector_store() -> Chroma:
    """
    Get the initialized vector store (loads existing or creates empty).

    This is a convenience function for retrieving the vector store
    in other parts of the application (like agents).

    Returns:
        Chroma vector store instance

    Example:
        >>> vector_store = get_vector_store()
        >>> results = vector_store.similarity_search("return policy", k=3)
    """
    return initialize_vector_store()


def reset_vector_store() -> Chroma:
    """
    Delete existing vector store and create a fresh one.

    Useful for reindexing all documents from scratch.

    Returns:
        Empty Chroma vector store
    """
    import shutil

    persist_dir = settings.CHROMA_PERSIST_DIR

    if persist_dir.exists():
        logger.info(f"Deleting existing vector store at {persist_dir}")
        shutil.rmtree(persist_dir)

    logger.info("Creating fresh vector store")
    return initialize_vector_store()
