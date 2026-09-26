"""Tests for document ingestion."""

import pytest
import tempfile
import os
from pathlib import Path

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.ingestion import DocumentIngestionPipeline


@pytest.fixture
def temp_documents_dir():
    """Create a temporary directory with test documents."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample markdown file
        md_file = Path(temp_dir) / "test.md"
        md_file.write_text("# Test Document\n\nThis is a test document for ingestion.")
        
        # Create a sample markdown file with more content
        md_file2 = Path(temp_dir) / "test2.md"
        md_file2.write_text("# Another Test\n\nMore content here.")
        
        yield temp_dir


class TestDocumentProcessor:
    """Tests for DocumentProcessor."""

    def test_load_markdown(self, temp_documents_dir):
        """Test loading markdown files."""
        processor = DocumentProcessor()
        md_file = Path(temp_documents_dir) / "test.md"
        
        documents = processor.load_markdown(str(md_file))
        assert len(documents) > 0
        assert "Test Document" in documents[0]

    def test_load_documents_from_directory(self, temp_documents_dir):
        """Test loading all documents from a directory."""
        processor = DocumentProcessor()
        documents = processor.load_documents_from_directory(temp_documents_dir)
        
        assert len(documents) > 0
        assert any("Test Document" in doc for doc in documents)
        assert any("Another Test" in doc for doc in documents)

    def test_chunk_documents(self):
        """Test document chunking."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        documents = ["This is a test document. " * 20]  # Long document
        
        chunks = processor.chunk_documents(documents)
        assert len(chunks) > 1  # Should be split into multiple chunks


class TestVectorStore:
    """Tests for VectorStore."""

    @pytest.fixture
    def temp_vector_store(self):
        """Create a temporary vector store for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = VectorStore(
                persist_directory=temp_dir,
                collection_name="test_collection",
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                use_openai_embeddings=False,
            )
            yield store

    def test_add_and_search(self, temp_vector_store):
        """Test adding documents and searching."""
        texts = ["Machine learning is a subset of AI", "Deep learning uses neural networks"]
        temp_vector_store.add_documents(texts)
        
        results = temp_vector_store.similarity_search("neural networks", k=2)
        assert len(results) > 0
        assert any("neural" in result["content"].lower() for result in results)

    def test_similarity_search_with_score(self, temp_vector_store):
        """Test similarity search with scores."""
        texts = ["Machine learning is a subset of AI", "Deep learning uses neural networks"]
        temp_vector_store.add_documents(texts)
        
        results = temp_vector_store.similarity_search_with_score("AI", k=2)
        assert len(results) > 0
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)  # (doc, score)

    def test_get_collection_stats(self, temp_vector_store):
        """Test getting collection statistics."""
        texts = ["Test document"]
        temp_vector_store.add_documents(texts)
        
        stats = temp_vector_store.get_collection_stats()
        assert stats["count"] >= 1
        assert stats["name"] == "test_collection"


class TestIngestionPipeline:
    """Tests for DocumentIngestionPipeline."""

    @pytest.fixture
    def temp_pipeline(self, temp_documents_dir):
        """Create a temporary ingestion pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = DocumentIngestionPipeline(
                documents_dir=temp_documents_dir,
                persist_directory=temp_dir,
                collection_name="test_collection",
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                use_openai_embeddings=False,
            )
            yield pipeline

    def test_ingest_documents(self, temp_pipeline):
        """Test ingesting documents."""
        result = temp_pipeline.ingest(clear_existing=False)
        
        assert result["status"] == "success"
        assert result["documents_loaded"] > 0
        assert result["chunks_created"] > 0

    def test_create_sample_documents(self, temp_pipeline):
        """Test creating sample documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = DocumentIngestionPipeline(
                documents_dir=temp_dir,
                persist_directory=temp_dir,
                collection_name="test_collection",
            )
            
            result = pipeline.create_sample_documents()
            assert result["status"] == "samples_created"
            assert len(result["files"]) == 2
            
            # Verify files were created
            assert os.path.exists(result["files"][0])
            assert os.path.exists(result["files"][1])
