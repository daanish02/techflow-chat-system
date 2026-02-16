"""Tests for RAG system components."""

from langchain_core.documents import Document

from src.rag.document_loader import (
    chunk_documents,
    load_and_chunk_policies,
    load_policy_documents,
)
from src.rag.retriever import (
    format_retrieved_context,
    query_policies,
    retrieve_relevant_policies,
    retrieve_with_scores,
)
from src.rag.vector_store import get_vector_store


class TestDocumentLoader:
    """Tests for document loading and chunking."""

    def test_load_policy_documents(self):
        """Test loading policy documents from directory."""
        docs = load_policy_documents()

        assert len(docs) == 3, "Should load 3 policy documents"

        # check all expected documents are loaded
        sources = [doc.metadata["source"] for doc in docs]
        assert "care_plus_benefits.md" in sources
        assert "return_policy.md" in sources
        assert "troubleshooting_guide.md" in sources

    def test_document_metadata(self):
        """Test that documents have correct metadata."""
        docs = load_policy_documents()

        for doc in docs:
            assert "source" in doc.metadata
            assert "policy_type" in doc.metadata
            assert "path" in doc.metadata
            assert doc.metadata["source"].endswith(".md")

    def test_chunk_documents(self):
        """Test document chunking."""
        docs = load_policy_documents()
        chunks = chunk_documents(docs)

        # create more chunks than original documents
        assert len(chunks) > len(docs)

        # all chunks should have metadata
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert len(chunk.page_content) > 0

    def test_chunk_size_limits(self):
        """Test that chunks respect size limits."""
        from src.config import settings

        chunks = load_and_chunk_policies()

        # check chunk sizes
        for chunk in chunks:
            chunk_size = len(chunk.page_content)
            assert chunk_size <= settings.CHUNK_SIZE + 100, (
                f"Chunk size {chunk_size} exceeds limit"
            )

    def test_load_and_chunk_combined(self):
        """Test the combined load and chunk function."""
        chunks = load_and_chunk_policies()

        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)


class TestVectorStore:
    """Tests for vector store operations."""

    def test_get_vector_store(self):
        """Test getting initialized vector store."""
        vector_store = get_vector_store()

        assert vector_store is not None
        assert vector_store._collection.count() > 0

    def test_vector_store_has_all_policies(self):
        """Test that vector store contains all policy documents."""
        vector_store = get_vector_store()

        # chunks from all 3 documents
        count = vector_store._collection.count()
        assert count >= 10, f"Expected at least 10 chunks, got {count}"

    def test_similarity_search(self):
        """Test basic similarity search."""
        vector_store = get_vector_store()

        results = vector_store.similarity_search("Care+ benefits", k=3)

        assert len(results) > 0
        assert len(results) <= 3
        assert all(isinstance(r, Document) for r in results)

    def test_similarity_search_with_scores(self):
        """Test similarity search returns scores."""
        vector_store = get_vector_store()

        results = vector_store.similarity_search_with_score("return policy", k=2)

        assert len(results) > 0
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, (int, float))


class TestRetrieval:
    """Tests for retrieval interface."""

    def test_retrieve_relevant_policies(self):
        """Test retrieving relevant policies."""
        docs = retrieve_relevant_policies("Care+ insurance coverage")

        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_retrieve_with_scores(self):
        """Test retrieval with relevance scores."""
        results = retrieve_with_scores("phone overheating", k=2)

        assert len(results) > 0
        assert len(results) <= 2

        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, (int, float))

    def test_format_retrieved_context(self):
        """Test context formatting."""
        docs = retrieve_relevant_policies("screen repair", k=2)
        context = format_retrieved_context(docs)

        assert isinstance(context, str)
        assert len(context) > 0
        assert "[Source:" in context

    def test_format_empty_documents(self):
        """Test formatting with no documents."""
        context = format_retrieved_context([])

        assert "No relevant" in context or "not found" in context.lower()

    def test_query_policies(self):
        """Test the main query function."""
        context = query_policies("What does Care+ cover?")

        assert isinstance(context, str)
        assert len(context) > 0


class TestRetrievalAccuracy:
    """Tests for retrieval accuracy on assignment scenarios."""

    def test_care_plus_benefits_query(self):
        """Test retrieving Care+ benefits information."""
        docs = retrieve_relevant_policies("Care+ benefits and coverage", k=3)

        # retrieve care_plus_benefits document
        sources = [doc.metadata["source"] for doc in docs]
        assert any("care_plus_benefits" in s for s in sources)

    def test_return_policy_query(self):
        """Test retrieving return policy information."""
        docs = retrieve_relevant_policies("return window for devices", k=3)

        # retrieve return_policy document
        sources = [doc.metadata["source"] for doc in docs]
        assert any("return_policy" in s for s in sources)

    def test_troubleshooting_query(self):
        """Test retrieving troubleshooting information."""
        docs = retrieve_relevant_policies("phone overheating problems", k=3)

        # retrieve troubleshooting_guide document
        sources = [doc.metadata["source"] for doc in docs]
        assert any("troubleshooting" in s for s in sources)

    def test_screen_repair_query(self):
        """Test query for screen repair (Test scenario 2)."""
        context = query_policies("screen repair coverage and cost")
        assert isinstance(context, str), "query_policies should return a string"

        assert "screen" in context.lower() or "repair" in context.lower()
        assert len(context) > 100

    def test_charging_issue_query(self):
        """Test query for charging issues (Test scenario 4)."""
        context = query_policies("phone not charging cable problems")
        assert isinstance(context, str), "query_policies should return a string"

        assert "charg" in context.lower() or "cable" in context.lower()

    def test_value_justification_query(self):
        """Test query for value justification (Test scenario 3)."""
        context = query_policies("why keep Care+ insurance benefits value")
        assert isinstance(context, str), "query_policies should return a string"

        # return benefits information
        assert len(context) > 100
        assert "care" in context.lower() or "benefit" in context.lower()


class TestScenarioQueries:
    """Test queries specific to the 5 assignment scenarios."""

    def test_scenario_1_financial_hardship(self):
        """Test retrieval for financial hardship scenario."""
        # customer can't afford, needs payment options
        context = query_policies("payment pause options financial help")

        assert len(context) > 0
        # find relevant information

    def test_scenario_2_overheating_phone(self):
        """Test retrieval for overheating phone scenario."""
        context = query_policies("phone overheating solutions replacement")
        assert isinstance(context, str), "query_policies should return a string"

        assert len(context) > 0
        assert "overheating" in context.lower() or "heat" in context.lower()

    def test_scenario_3_questioning_value(self):
        """Test retrieval for value questioning scenario."""
        context = query_policies("Care+ benefits never used worth it")
        assert isinstance(context, str), "query_policies should return a string"

        assert len(context) > 0
        # contain benefits information
        assert "care" in context.lower() or "coverage" in context.lower()

    def test_scenario_4_charging_issues(self):
        """Test retrieval for tech support scenario."""
        context = query_policies("phone won't charge tried cables")
        assert isinstance(context, str), "query_policies should return a string"

        assert len(context) > 0
        assert "charg" in context.lower() or "cable" in context.lower()

    def test_scenario_5_billing_question(self):
        """Test retrieval for billing question."""
        # billing questions might not have specific policy docs
        context = query_policies("monthly cost pricing tiers")

        # return something even if not highly relevant
        assert len(context) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_query(self):
        """Test behavior with empty query."""
        docs = retrieve_relevant_policies("")

        # handle gracefully
        assert isinstance(docs, list)

    def test_very_long_query(self):
        """Test with very long query."""
        long_query = "phone problems " * 100
        docs = retrieve_relevant_policies(long_query, k=1)

        assert isinstance(docs, list)

    def test_irrelevant_query(self):
        """Test with completely irrelevant query."""
        docs = retrieve_relevant_policies("quantum physics spacetime", k=1)

        # still return something
        assert isinstance(docs, list)

    def test_special_characters_query(self):
        """Test query with special characters."""
        docs = retrieve_relevant_policies("Care+ @#$ coverage!!!", k=1)

        assert isinstance(docs, list)
