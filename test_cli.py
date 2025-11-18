"""
Tests for CLI commands.
"""
import os
import pytest
import tempfile
import shutil
import csv
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from cli import cli, add_document, query, stats, clear, usage


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
    writer = csv.writer(temp_file)
    writer.writerow(['Name', 'Age', 'City'])
    writer.writerow(['Alice', '30', 'New York'])
    writer.writerow(['Bob', '25', 'San Francisco'])
    temp_file.close()
    yield temp_file.name
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.fixture
def mock_rag_service():
    """Mock RAGService for testing."""
    with patch('cli.RAGService') as mock_service:
        # Create mock instance
        mock_instance = MagicMock()
        mock_service.return_value = mock_instance

        # Configure default return values
        mock_instance.add_document.return_value = {
            'status': 'success',
            'document_id': 'test123',
            'filename': 'test.csv',
            'chunks': 5,
            'total_documents': 10
        }

        mock_instance.get_stats.return_value = {
            'total_chunks': 50,
            'collection_name': 'documents',
            'db_path': './test_db'
        }

        mock_instance.query.return_value = [
            {
                'content': 'Test content 1',
                'metadata': {'filename': 'test.csv', 'chunk_index': 0}
            },
            {
                'content': 'Test content 2',
                'metadata': {'filename': 'test.csv', 'chunk_index': 1}
            }
        ]

        yield mock_instance


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    with patch('cli.LLMFactory.create_provider') as mock_factory:
        mock_provider = MagicMock()
        mock_factory.return_value = mock_provider
        yield mock_provider


@pytest.fixture
def mock_generate_rag_response():
    """Mock generate_rag_response function."""
    with patch('cli.generate_rag_response') as mock_generate:
        mock_generate.return_value = {
            'response': 'This is a test response from the LLM.',
            'sources': [
                {'filename': 'test.csv', 'chunk_index': 0},
                {'filename': 'test.csv', 'chunk_index': 1}
            ],
            'num_contexts_used': 2
        }
        yield mock_generate


class TestCLIBasics:
    """Basic CLI tests."""

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'RAG CLI' in result.output
        assert 'add_document' in result.output or 'add-document' in result.output

    def test_cli_with_db_path_option(self, runner, temp_db_dir):
        """Test CLI with custom db-path option."""
        result = runner.invoke(cli, ['--db-path', temp_db_dir, '--help'])
        assert result.exit_code == 0

    def test_cli_with_collection_option(self, runner):
        """Test CLI with custom collection option."""
        result = runner.invoke(cli, ['--collection', 'test_collection', '--help'])
        assert result.exit_code == 0


class TestAddDocumentCommand:
    """Tests for add_document command."""

    def test_add_document_help(self, runner):
        """Test add_document help."""
        result = runner.invoke(cli, ['add_document', '--help'])
        assert result.exit_code == 0
        assert 'Add a document' in result.output

    def test_add_document_nonexistent_file(self, runner):
        """Test adding nonexistent file."""
        result = runner.invoke(cli, ['add_document', '/nonexistent/file.pdf'])
        assert result.exit_code != 0

    @patch('cli.RAGService')
    def test_add_document_success(self, mock_service_class, runner, temp_csv_file):
        """Test successfully adding a document."""
        # Configure mock
        mock_instance = MagicMock()
        mock_service_class.return_value = mock_instance
        mock_instance.add_document.return_value = {
            'status': 'success',
            'document_id': 'abc123',
            'filename': 'test.csv',
            'chunks': 3,
            'total_documents': 1
        }

        result = runner.invoke(cli, ['add_document', temp_csv_file])

        assert result.exit_code == 0
        assert 'success' in result.output.lower()
        assert 'test.csv' in result.output
        mock_instance.add_document.assert_called_once_with(temp_csv_file)

    @patch('cli.RAGService')
    def test_add_document_error(self, mock_service_class, runner, temp_csv_file):
        """Test adding document with error."""
        # Configure mock to return error
        mock_instance = MagicMock()
        mock_service_class.return_value = mock_instance
        mock_instance.add_document.return_value = {
            'status': 'error',
            'message': 'No content could be extracted'
        }

        result = runner.invoke(cli, ['add_document', temp_csv_file])

        assert result.exit_code != 0
        assert 'error' in result.output.lower()

    @patch('cli.RAGService')
    def test_add_document_with_custom_db_path(self, mock_service_class, runner, temp_csv_file, temp_db_dir):
        """Test adding document with custom db path."""
        mock_instance = MagicMock()
        mock_service_class.return_value = mock_instance
        mock_instance.add_document.return_value = {
            'status': 'success',
            'filename': 'test.csv',
            'chunks': 3,
            'total_documents': 1
        }

        result = runner.invoke(cli, ['--db-path', temp_db_dir, 'add_document', temp_csv_file])

        # Verify RAGService was called with correct db_path
        mock_service_class.assert_called_with(db_path=temp_db_dir, collection_name='documents')

    @patch('cli.RAGService')
    def test_add_document_exception(self, mock_service_class, runner, temp_csv_file):
        """Test handling of exception during add_document."""
        mock_instance = MagicMock()
        mock_service_class.return_value = mock_instance
        mock_instance.add_document.side_effect = Exception("Test error")

        result = runner.invoke(cli, ['add_document', temp_csv_file])

        assert result.exit_code != 0
        assert 'error' in result.output.lower()


class TestQueryCommand:
    """Tests for query command."""

    def test_query_help(self, runner):
        """Test query help."""
        result = runner.invoke(cli, ['query', '--help'])
        assert result.exit_code == 0
        assert 'Query the RAG database' in result.output

    @patch('cli.generate_rag_response')
    @patch('cli.LLMFactory.create_provider')
    @patch('cli.RAGService')
    def test_query_success(self, mock_service_class, mock_llm_factory, mock_generate, runner):
        """Test successful query."""
        # Configure RAG service mock
        mock_rag = MagicMock()
        mock_service_class.return_value = mock_rag
        mock_rag.get_stats.return_value = {'total_chunks': 10}
        mock_rag.query.return_value = [
            {'content': 'Test content', 'metadata': {'filename': 'test.csv', 'chunk_index': 0}}
        ]

        # Configure LLM mock
        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm

        # Configure generate_rag_response mock
        mock_generate.return_value = {
            'response': 'Test response',
            'sources': [{'filename': 'test.csv', 'chunk_index': 0}],
            'num_contexts_used': 1
        }

        result = runner.invoke(cli, ['query', 'What is the test about?'])

        assert result.exit_code == 0
        assert 'Test response' in result.output

    @patch('cli.RAGService')
    def test_query_empty_database(self, mock_service_class, runner):
        """Test querying empty database."""
        mock_rag = MagicMock()
        mock_service_class.return_value = mock_rag
        mock_rag.get_stats.return_value = {'total_chunks': 0}

        result = runner.invoke(cli, ['query', 'test query'])

        assert result.exit_code != 0
        assert 'No documents in database' in result.output

    @patch('cli.LLMFactory.create_provider')
    @patch('cli.RAGService')
    def test_query_no_relevant_contexts(self, mock_service_class, mock_llm_factory, runner):
        """Test query with no relevant contexts found."""
        mock_rag = MagicMock()
        mock_service_class.return_value = mock_rag
        mock_rag.get_stats.return_value = {'total_chunks': 10}
        mock_rag.query.return_value = []

        result = runner.invoke(cli, ['query', 'test query'])

        assert result.exit_code == 0
        assert 'No relevant contexts found' in result.output

    @patch('cli.generate_rag_response')
    @patch('cli.LLMFactory.create_provider')
    @patch('cli.RAGService')
    def test_query_with_show_sources(self, mock_service_class, mock_llm_factory, mock_generate, runner):
        """Test query with --show-sources flag."""
        # Configure mocks
        mock_rag = MagicMock()
        mock_service_class.return_value = mock_rag
        mock_rag.get_stats.return_value = {'total_chunks': 10}
        mock_rag.query.return_value = [
            {'content': 'Test', 'metadata': {'filename': 'test.csv', 'chunk_index': 0}}
        ]

        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm

        mock_generate.return_value = {
            'response': 'Test response',
            'sources': [{'filename': 'test.csv', 'chunk_index': 0}],
            'num_contexts_used': 1
        }

        result = runner.invoke(cli, ['query', 'test', '--show-sources'])

        assert result.exit_code == 0
        assert 'Sources:' in result.output
        assert 'test.csv' in result.output

    @patch('cli.generate_rag_response')
    @patch('cli.LLMFactory.create_provider')
    @patch('cli.RAGService')
    def test_query_with_custom_provider(self, mock_service_class, mock_llm_factory, mock_generate, runner):
        """Test query with custom provider."""
        mock_rag = MagicMock()
        mock_service_class.return_value = mock_rag
        mock_rag.get_stats.return_value = {'total_chunks': 10}
        mock_rag.query.return_value = [
            {'content': 'Test', 'metadata': {'filename': 'test.csv', 'chunk_index': 0}}
        ]

        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm

        mock_generate.return_value = {
            'response': 'Test response',
            'sources': [],
            'num_contexts_used': 1
        }

        result = runner.invoke(cli, ['query', 'test', '--provider', 'anthropic'])

        assert result.exit_code == 0
        mock_llm_factory.assert_called_with('anthropic')

    @patch('cli.generate_rag_response')
    @patch('cli.LLMFactory.create_provider')
    @patch('cli.RAGService')
    def test_query_with_custom_model(self, mock_service_class, mock_llm_factory, mock_generate, runner):
        """Test query with custom model."""
        mock_rag = MagicMock()
        mock_service_class.return_value = mock_rag
        mock_rag.get_stats.return_value = {'total_chunks': 10}
        mock_rag.query.return_value = [
            {'content': 'Test', 'metadata': {'filename': 'test.csv', 'chunk_index': 0}}
        ]

        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm

        mock_generate.return_value = {
            'response': 'Test response',
            'sources': [],
            'num_contexts_used': 1
        }

        result = runner.invoke(cli, ['query', 'test', '--model', 'gpt-4'])

        assert result.exit_code == 0
        mock_llm_factory.assert_called_with('openai', model='gpt-4')

    @patch('cli.LLMFactory.create_provider')
    @patch('cli.RAGService')
    def test_query_llm_error(self, mock_service_class, mock_llm_factory, runner):
        """Test query with LLM provider error."""
        mock_rag = MagicMock()
        mock_service_class.return_value = mock_rag
        mock_rag.get_stats.return_value = {'total_chunks': 10}
        mock_rag.query.return_value = [
            {'content': 'Test', 'metadata': {'filename': 'test.csv', 'chunk_index': 0}}
        ]

        mock_llm_factory.side_effect = ValueError("Invalid API key")

        result = runner.invoke(cli, ['query', 'test'])

        assert result.exit_code != 0
        assert 'Configuration error' in result.output or 'Error' in result.output


class TestStatsCommand:
    """Tests for stats command."""

    def test_stats_help(self, runner):
        """Test stats help."""
        result = runner.invoke(cli, ['stats', '--help'])
        assert result.exit_code == 0
        assert 'statistics' in result.output.lower()

    @patch('cli.RAGService')
    def test_stats_success(self, mock_service_class, runner):
        """Test stats command success."""
        mock_rag = MagicMock()
        mock_service_class.return_value = mock_rag
        mock_rag.get_stats.return_value = {
            'total_chunks': 100,
            'collection_name': 'documents',
            'db_path': './chroma_db'
        }

        result = runner.invoke(cli, ['stats'])

        assert result.exit_code == 0
        assert 'Statistics' in result.output
        assert '100' in result.output
        assert 'documents' in result.output

    @patch('cli.RAGService')
    def test_stats_with_custom_db_path(self, mock_service_class, runner, temp_db_dir):
        """Test stats with custom db path."""
        mock_rag = MagicMock()
        mock_service_class.return_value = mock_rag
        mock_rag.get_stats.return_value = {
            'total_chunks': 50,
            'collection_name': 'test_collection',
            'db_path': temp_db_dir
        }

        result = runner.invoke(cli, ['--db-path', temp_db_dir, 'stats'])

        assert result.exit_code == 0
        mock_service_class.assert_called_with(db_path=temp_db_dir, collection_name='documents')

    @patch('cli.RAGService')
    def test_stats_error(self, mock_service_class, runner):
        """Test stats command with error."""
        mock_rag = MagicMock()
        mock_service_class.return_value = mock_rag
        mock_rag.get_stats.side_effect = Exception("Database error")

        result = runner.invoke(cli, ['stats'])

        assert result.exit_code != 0
        assert 'error' in result.output.lower()


class TestClearCommand:
    """Tests for clear command."""

    def test_clear_help(self, runner):
        """Test clear help."""
        result = runner.invoke(cli, ['clear', '--help'])
        assert result.exit_code == 0
        assert 'Clear all documents' in result.output or 'delete' in result.output.lower()

    @patch('cli.RAGService')
    def test_clear_aborted(self, mock_service_class, runner):
        """Test clear command when user aborts."""
        result = runner.invoke(cli, ['clear'], input='n\n')

        # Should abort without calling delete
        assert result.exit_code != 0 or 'Aborted' in result.output

    @patch('cli.RAGService')
    def test_clear_confirmed(self, mock_service_class, runner):
        """Test clear command when user confirms."""
        mock_rag = MagicMock()
        mock_service_class.return_value = mock_rag

        result = runner.invoke(cli, ['clear'], input='y\n')

        assert result.exit_code == 0
        assert 'cleared' in result.output.lower() or 'success' in result.output.lower()
        mock_rag.delete_all.assert_called_once()

    @patch('cli.RAGService')
    def test_clear_error(self, mock_service_class, runner):
        """Test clear command with error."""
        mock_rag = MagicMock()
        mock_service_class.return_value = mock_rag
        mock_rag.delete_all.side_effect = Exception("Database error")

        result = runner.invoke(cli, ['clear'], input='y\n')

        assert result.exit_code != 0
        assert 'error' in result.output.lower()


class TestUsageCommand:
    """Tests for usage command."""

    def test_usage_help(self, runner):
        """Test usage help."""
        result = runner.invoke(cli, ['usage', '--help'])
        assert result.exit_code == 0
        assert 'token usage' in result.output.lower()

    @patch('cli.get_logger')
    def test_usage_no_logs(self, mock_get_logger, runner):
        """Test usage command with no logs."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_logger.get_all_log_files.return_value = []

        result = runner.invoke(cli, ['usage'])

        assert result.exit_code == 0
        assert 'No usage data found' in result.output

    @patch('cli.get_logger')
    def test_usage_with_logs(self, mock_get_logger, runner):
        """Test usage command with log data."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_logger.get_all_log_files.return_value = [Path('test.log')]
        mock_logger.get_statistics.return_value = {
            'total': {
                'requests': 10,
                'input_tokens': 1000,
                'output_tokens': 500,
                'total_tokens': 1500
            },
            'by_provider': {
                'openai': {
                    'requests': 10,
                    'input_tokens': 1000,
                    'output_tokens': 500,
                    'total_tokens': 1500,
                    'by_model': {
                        'gpt-3.5-turbo': {
                            'requests': 10,
                            'input_tokens': 1000,
                            'output_tokens': 500,
                            'total_tokens': 1500
                        }
                    }
                }
            }
        }

        result = runner.invoke(cli, ['usage'])

        assert result.exit_code == 0
        assert '10' in result.output
        assert '1,000' in result.output or '1000' in result.output

    @patch('cli.get_logger')
    def test_usage_with_date_filter(self, mock_get_logger, runner):
        """Test usage command with date filter."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Mock _get_log_file_path to return a Path that exists
        mock_log_path = MagicMock(spec=Path)
        mock_log_path.exists.return_value = True
        mock_logger._get_log_file_path.return_value = mock_log_path

        mock_logger.get_statistics.return_value = {
            'total': {
                'requests': 5,
                'input_tokens': 500,
                'output_tokens': 250,
                'total_tokens': 750
            },
            'by_provider': {}
        }

        result = runner.invoke(cli, ['usage', '--date', '2025-01-15'])

        assert result.exit_code == 0
        assert '2025-01-15' in result.output

    @patch('cli.get_logger')
    def test_usage_with_show_costs(self, mock_get_logger, runner):
        """Test usage command with --show-costs flag."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_logger.get_all_log_files.return_value = [Path('test.log')]
        mock_logger.get_statistics.return_value = {
            'total': {
                'requests': 10,
                'input_tokens': 1000,
                'output_tokens': 500,
                'total_tokens': 1500
            },
            'by_provider': {
                'openai': {
                    'requests': 10,
                    'input_tokens': 1000,
                    'output_tokens': 500,
                    'total_tokens': 1500,
                    'by_model': {
                        'gpt-3.5-turbo': {
                            'requests': 10,
                            'input_tokens': 1000,
                            'output_tokens': 500,
                            'total_tokens': 1500
                        }
                    }
                }
            }
        }
        mock_logger.get_cost_estimate.return_value = 0.0025

        result = runner.invoke(cli, ['usage', '--show-costs'])

        assert result.exit_code == 0
        assert 'cost' in result.output.lower()
