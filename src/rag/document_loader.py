"""Document loading and chunking for RAG system."""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


def load_policy_documents() -> List[Document]:
    """
    Load all policy documents from the policies directory.

    Reads markdown files from data/policies/ and converts them into
    LangChain Document objects with metadata.

    Returns:
        List of Document objects with content and metadata

    Example:
        >>> docs = load_policy_documents()
        >>> len(docs)
        3
        >>> docs[0].metadata['source']
        'care_plus_benefits.md'
    """
    policies_dir = settings.DATA_DIR / "policies"

    if not policies_dir.exists():
        logger.error(f"Policies directory not found: {policies_dir}")
        return []

    documents = []

    # load all .md files
    for policy_file in policies_dir.glob("*.md"):
        try:
            content = policy_file.read_text(encoding="utf-8")

            doc = Document(
                page_content=content,
                metadata={
                    "source": policy_file.name,
                    "policy_type": policy_file.stem,
                    "path": str(policy_file),
                },
            )
            documents.append(doc)
            logger.info(f"Loaded policy document: {policy_file.name}")

        except Exception as e:
            logger.error(f"Error loading {policy_file.name}: {e}")

    logger.info(f"Loaded {len(documents)} policy documents")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.

    Uses recursive character splitting to maintain semantic boundaries
    while keeping chunks within the configured size limits.

    Args:
        documents: List of Document objects to split

    Returns:
        List of chunked Document objects with preserved metadata

    Example:
        >>> docs = load_policy_documents()
        >>> chunks = chunk_documents(docs)
        >>> len(chunks) > len(docs)
        True
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n## ",  # headers
            "\n### ",
            "\n#### ",
            "\n\n",  # paragraphs
            "\n",  # lines
            ". ",  # sentences
            " ",  # words
            "",  # characters
        ],
    )

    chunks = text_splitter.split_documents(documents)

    logger.info(
        f"Split {len(documents)} documents into {len(chunks)} chunks "
        f"(avg {len(chunks) // len(documents) if documents else 0} chunks per doc)"
    )

    return chunks


def load_and_chunk_policies() -> List[Document]:
    """
    Load policy documents and split them into chunks in one step.

    This is the main function to use for initializing the RAG system.

    Returns:
        List of chunked Document objects ready for embedding

    Example:
        >>> chunks = load_and_chunk_policies()
        >>> chunks[0].metadata['source']
        'care_plus_benefits.md'
        >>> len(chunks[0].page_content) <= 500
        True
    """
    documents = load_policy_documents()

    if not documents:
        logger.warning("No policy documents loaded")
        return []

    chunks = chunk_documents(documents)

    logger.info(f"Ready to embed {len(chunks)} document chunks")
    return chunks
