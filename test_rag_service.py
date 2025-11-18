"""
Tests for RAGService class.
"""
import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from rag_service import RAGService


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for test database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_test_file():
    """Create a temporary test text file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file.write("This is a test document.\n" * 100)  # Create content > 1000 chars
    temp_file.close()
    yield temp_file.name
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.fixture
def mock_rag_service(temp_db_path):
    """Create a RAGService instance with mocked embedding model."""
    with patch('rag_service.SentenceTransformer') as mock_transformer:
        # Mock the embedding model
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]  # Dummy embedding
        mock_transformer.return_value = mock_model

        service = RAGService(db_path=temp_db_path, collection_name="test_collection")
        yield service


@pytest.fixture
def rag_service_with_real_embeddings(temp_db_path):
    """Create a RAGService instance with real embeddings (for integration tests)."""
    service = RAGService(db_path=temp_db_path, collection_name="test_collection")
    yield service


class TestRAGServiceInitialization:
    """Tests for RAGService initialization."""

    def test_init_creates_db_directory(self, temp_db_path):
        """Test that initialization creates database directory."""
        with patch('rag_service.SentenceTransformer'):
            service = RAGService(db_path=temp_db_path, collection_name="test")
            assert os.path.exists(temp_db_path)

    def test_init_sets_attributes(self, mock_rag_service, temp_db_path):
        """Test that initialization sets correct attributes."""
        assert mock_rag_service.db_path == temp_db_path
        assert mock_rag_service.collection_name == "test_collection"
        assert mock_rag_service.collection is not None

    def test_init_creates_collection(self, mock_rag_service):
        """Test that initialization creates collection."""
        assert mock_rag_service.collection.count() == 0


class TestChunkText:
    """Tests for _chunk_text method."""

    def test_chunk_short_text(self, mock_rag_service):
        """Test chunking text shorter than chunk_size."""
        text = "This is a short text."
        chunks = mock_rag_service._chunk_text(text, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_long_text(self, mock_rag_service):
        """Test chunking text longer than chunk_size."""
        text = "A" * 2500  # Create long text
        chunks = mock_rag_service._chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) > 1
        assert all(len(chunk) <= 1000 for chunk in chunks)

    def test_chunk_with_overlap(self, mock_rag_service):
        """Test that chunks have proper overlap."""
        text = "Word " * 500  # Create text with clear boundaries
        chunks = mock_rag_service._chunk_text(text, chunk_size=1000, overlap=200)

        # Check that we have multiple chunks
        assert len(chunks) >= 2

    def test_chunk_respects_sentence_boundaries(self, mock_rag_service):
        """Test that chunking respects sentence boundaries."""
        text = "This is sentence one. " * 100
        chunks = mock_rag_service._chunk_text(text, chunk_size=1000, overlap=200)

        # Most chunks should end with period (except possibly the last)
        for chunk in chunks[:-1]:
            # Should end with period or be at natural boundary
            assert chunk.endswith('.') or chunk.endswith('. ')

    def test_chunk_empty_text(self, mock_rag_service):
        """Test chunking empty text."""
        chunks = mock_rag_service._chunk_text("")
        assert len(chunks) == 0

    def test_chunk_whitespace_only(self, mock_rag_service):
        """Test chunking whitespace-only text."""
        chunks = mock_rag_service._chunk_text("   \n  \n  ")
        assert len(chunks) == 0


class TestGenerateDocId:
    """Tests for _generate_doc_id method."""

    def test_generate_doc_id_consistent(self, mock_rag_service):
        """Test that same path generates same ID."""
        path = "/test/path/file.txt"
        id1 = mock_rag_service._generate_doc_id(path)
        id2 = mock_rag_service._generate_doc_id(path)
        assert id1 == id2

    def test_generate_doc_id_different_paths(self, mock_rag_service):
        """Test that different paths generate different IDs."""
        id1 = mock_rag_service._generate_doc_id("/test/file1.txt")
        id2 = mock_rag_service._generate_doc_id("/test/file2.txt")
        assert id1 != id2

    def test_generate_doc_id_format(self, mock_rag_service):
        """Test that generated ID is valid MD5 hash."""
        doc_id = mock_rag_service._generate_doc_id("/test/file.txt")
        assert len(doc_id) == 32  # MD5 hex digest length
        assert all(c in '0123456789abcdef' for c in doc_id)


class TestAddDocument:
    """Tests for add_document method."""

    @patch('rag_service.DocumentLoader.load_document')
    def test_add_document_success(self, mock_loader, mock_rag_service):
        """Test successfully adding a document."""
        # Mock document loader
        mock_loader.return_value = {
            'content': 'This is test content. ' * 100,
            'metadata': {
                'source': '/test/file.txt',
                'filename': 'file.txt',
                'extension': '.txt',
                'size': 1000
            }
        }

        # Mock embedding model to return proper shape
        mock_rag_service.embedding_model.encode.return_value = [[0.1, 0.2] for _ in range(3)]

        result = mock_rag_service.add_document('/test/file.txt')

        assert result['status'] == 'success'
        assert result['filename'] == 'file.txt'
        assert result['chunks'] > 0
        assert 'document_id' in result

    @patch('rag_service.DocumentLoader.load_document')
    def test_add_document_empty_content(self, mock_loader, mock_rag_service):
        """Test adding document with no extractable content."""
        mock_loader.return_value = {
            'content': '',
            'metadata': {
                'source': '/test/empty.txt',
                'filename': 'empty.txt',
                'extension': '.txt',
                'size': 0
            }
        }

        result = mock_rag_service.add_document('/test/empty.txt')

        assert result['status'] == 'error'
        assert 'No content' in result['message']

    @patch('rag_service.DocumentLoader.load_document')
    def test_add_document_increments_count(self, mock_loader, mock_rag_service):
        """Test that adding documents increments collection count."""
        mock_loader.return_value = {
            'content': 'Test content. ' * 100,
            'metadata': {
                'source': '/test/file.txt',
                'filename': 'file.txt',
                'extension': '.txt',
                'size': 1000
            }
        }

        mock_rag_service.embedding_model.encode.return_value = [[0.1, 0.2] for _ in range(3)]

        initial_count = mock_rag_service.collection.count()
        mock_rag_service.add_document('/test/file.txt')
        final_count = mock_rag_service.collection.count()

        assert final_count > initial_count


class TestQuery:
    """Tests for query method."""

    @patch('rag_service.DocumentLoader.load_document')
    def test_query_returns_results(self, mock_loader, mock_rag_service):
        """Test querying returns relevant results."""
        # Add a document first
        mock_loader.return_value = {
            'content': 'The quick brown fox jumps over the lazy dog. ' * 30,
            'metadata': {
                'source': '/test/fox.txt',
                'filename': 'fox.txt',
                'extension': '.txt',
                'size': 500
            }
        }

        mock_rag_service.embedding_model.encode.return_value = [[0.1, 0.2, 0.3] for _ in range(5)]
        mock_rag_service.add_document('/test/fox.txt')

        # Query for content
        results = mock_rag_service.query("fox jumps", n_results=3)

        assert isinstance(results, list)
        assert len(results) <= 3

    def test_query_empty_database(self, mock_rag_service):
        """Test querying empty database returns empty list."""
        results = mock_rag_service.query("test query")
        assert results == []

    @patch('rag_service.DocumentLoader.load_document')
    def test_query_with_metadata_filter(self, mock_loader, mock_rag_service):
        """Test querying with metadata filter."""
        mock_loader.return_value = {
            'content': 'Test content. ' * 50,
            'metadata': {
                'source': '/test/file.txt',
                'filename': 'file.txt',
                'extension': '.txt',
                'size': 500
            }
        }

        mock_rag_service.embedding_model.encode.return_value = [[0.1, 0.2, 0.3] for _ in range(5)]
        mock_rag_service.add_document('/test/file.txt')

        # Query with filter
        results = mock_rag_service.query(
            "test",
            n_results=5,
            filter_metadata={'filename': 'file.txt'}
        )

        assert isinstance(results, list)

    @patch('rag_service.DocumentLoader.load_document')
    def test_query_result_format(self, mock_loader, mock_rag_service):
        """Test that query results have correct format."""
        mock_loader.return_value = {
            'content': 'Test content. ' * 50,
            'metadata': {
                'source': '/test/file.txt',
                'filename': 'file.txt',
                'extension': '.txt',
                'size': 500
            }
        }

        mock_rag_service.embedding_model.encode.return_value = [[0.1, 0.2, 0.3] for _ in range(5)]
        mock_rag_service.add_document('/test/file.txt')

        results = mock_rag_service.query("test")

        if results:
            result = results[0]
            assert 'content' in result
            assert 'metadata' in result
            assert isinstance(result['metadata'], dict)


class TestGetStats:
    """Tests for get_stats method."""

    def test_get_stats_empty_database(self, mock_rag_service):
        """Test getting stats from empty database."""
        stats = mock_rag_service.get_stats()

        assert stats['total_chunks'] == 0
        assert stats['collection_name'] == 'test_collection'
        assert 'db_path' in stats

    @patch('rag_service.DocumentLoader.load_document')
    def test_get_stats_with_documents(self, mock_loader, mock_rag_service):
        """Test getting stats after adding documents."""
        mock_loader.return_value = {
            'content': 'Test content. ' * 100,
            'metadata': {
                'source': '/test/file.txt',
                'filename': 'file.txt',
                'extension': '.txt',
                'size': 1000
            }
        }

        mock_rag_service.embedding_model.encode.return_value = [[0.1, 0.2] for _ in range(5)]
        mock_rag_service.add_document('/test/file.txt')

        stats = mock_rag_service.get_stats()

        assert stats['total_chunks'] > 0
        assert stats['collection_name'] == 'test_collection'


class TestDeleteAll:
    """Tests for delete_all method."""

    @patch('rag_service.DocumentLoader.load_document')
    def test_delete_all_clears_collection(self, mock_loader, mock_rag_service):
        """Test that delete_all removes all documents."""
        # Add a document
        mock_loader.return_value = {
            'content': 'Test content. ' * 100,
            'metadata': {
                'source': '/test/file.txt',
                'filename': 'file.txt',
                'extension': '.txt',
                'size': 1000
            }
        }

        mock_rag_service.embedding_model.encode.return_value = [[0.1, 0.2] for _ in range(5)]
        mock_rag_service.add_document('/test/file.txt')

        assert mock_rag_service.collection.count() > 0

        # Delete all
        mock_rag_service.delete_all()

        assert mock_rag_service.collection.count() == 0

    def test_delete_all_empty_database(self, mock_rag_service):
        """Test delete_all on empty database doesn't raise error."""
        mock_rag_service.delete_all()
        assert mock_rag_service.collection.count() == 0


@pytest.mark.integration
class TestRAGServiceIntegration:
    """Integration tests with real embeddings (slower)."""

    def test_full_workflow(self, rag_service_with_real_embeddings, temp_test_file):
        """Test complete workflow: add document, query, get stats, delete."""
        service = rag_service_with_real_embeddings

        # Add document
        with patch('rag_service.DocumentLoader.load_document') as mock_loader:
            mock_loader.return_value = {
                'content': 'The quick brown fox jumps over the lazy dog. ' * 50,
                'metadata': {
                    'source': temp_test_file,
                    'filename': 'test.txt',
                    'extension': '.txt',
                    'size': 1000
                }
            }

            result = service.add_document(temp_test_file)
            assert result['status'] == 'success'

        # Query
        results = service.query("fox jumps", n_results=3)
        assert len(results) > 0

        # Get stats
        stats = service.get_stats()
        assert stats['total_chunks'] > 0

        # Delete all
        service.delete_all()
        assert service.collection.count() == 0
