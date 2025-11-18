# RAG Database and Commands - Test Suite

This document describes the test suite for the RAG (Retrieval-Augmented Generation) system.

## Test Files

### 1. `tests/test_rag_service.py`
Tests for the core RAG service functionality (`src/rag_service.py`).

**Test Coverage:**
- `TestRAGServiceInitialization`: Database initialization and setup
- `TestChunkText`: Text chunking logic with overlap
- `TestGenerateDocId`: Document ID generation (MD5 hashing)
- `TestAddDocument`: Document ingestion and storage
- `TestQuery`: Vector search and retrieval
- `TestGetStats`: Database statistics
- `TestDeleteAll`: Database clearing functionality
- `TestRAGServiceIntegration`: End-to-end integration tests

**Key Features Tested:**
- ChromaDB initialization and collection management
- Text chunking with configurable size and overlap
- Sentence boundary detection in chunks
- Document embedding generation
- Vector similarity search
- Metadata tracking
- Error handling for empty content

### 2. `tests/test_document_loader.py`
Tests for document loading functionality (`src/document_loader.py`).

**Test Coverage:**
- `TestDocumentLoaderBasics`: Core loader functionality
- `TestLoadCSV`: CSV file parsing
- `TestLoadDOCX`: Microsoft Word document parsing
- `TestLoadPDF`: PDF document parsing
- `TestLoadDOC`: Legacy DOC file handling
- `TestLoadODT`: OpenDocument format parsing
- `TestDocumentLoaderEdgeCases`: Edge cases and error handling
- `TestStaticMethods`: Individual loader methods

**Key Features Tested:**
- Support for multiple file formats (.pdf, .docx, .doc, .odt, .csv)
- File existence validation
- Format validation and error messages
- Metadata extraction (source, filename, extension, size)
- Path normalization (absolute paths, case-insensitive extensions)
- Special characters and Unicode handling
- Empty file handling

### 3. `tests/test_cli.py`
Tests for the command-line interface (`src/cli.py`).

**Test Coverage:**
- `TestCLIBasics`: CLI initialization and help
- `TestAddDocumentCommand`: Document addition command
- `TestQueryCommand`: Query and response generation
- `TestStatsCommand`: Database statistics display
- `TestClearCommand`: Database clearing with confirmation
- `TestUsageCommand`: Token usage and cost tracking

**Key Features Tested:**
- Click CLI framework integration
- Command-line argument parsing
- Database path and collection customization
- LLM provider selection (OpenAI, Anthropic, Mistral, Gemini)
- Model selection and API key handling
- Source attribution display
- Confirmation prompts for destructive operations
- Token usage statistics and cost estimation
- Error handling and user-friendly messages

## Running the Tests

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install test dependencies (if not already included):**
   ```bash
   pip install pytest pytest-mock pytest-cov reportlab
   ```

### Running All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=. --cov-report=html
```

### Running Specific Test Files

```bash
# Test RAG service only
pytest tests/test_rag_service.py -v

# Test document loader only
pytest tests/test_document_loader.py -v

# Test CLI commands only
pytest tests/test_cli.py -v
```

### Running Specific Test Classes

```bash
# Test only chunk text functionality
pytest tests/test_rag_service.py::TestChunkText -v

# Test only CSV loading
pytest tests/test_document_loader.py::TestLoadCSV -v

# Test only query command
pytest tests/test_cli.py::TestQueryCommand -v
```

### Running Integration Tests

```bash
# Run only integration tests (slower, uses real embeddings)
pytest -m integration -v

# Skip integration tests
pytest -m "not integration" -v
```

## Test Markers

- `@pytest.mark.unit`: Unit tests (fast, mocked dependencies)
- `@pytest.mark.integration`: Integration tests (slower, real dependencies)
- `@pytest.mark.slow`: Tests that take longer to execute

## Test Fixtures

### Common Fixtures

- `temp_db_path`: Temporary database directory (auto-cleanup)
- `temp_test_file`: Temporary text file for testing
- `temp_csv_file`: Temporary CSV file with sample data
- `temp_docx_file`: Temporary DOCX file
- `temp_pdf_file`: Temporary PDF file
- `mock_rag_service`: Mocked RAG service with embeddings
- `rag_service_with_real_embeddings`: Real RAG service for integration tests
- `runner`: Click CLI test runner
- `mock_llm_provider`: Mocked LLM provider

## Code Coverage

To generate a detailed coverage report:

```bash
# Generate HTML coverage report
pytest --cov=. --cov-report=html --cov-report=term

# View report
open htmlcov/index.html
```

## Continuous Integration

The tests are designed to run in CI/CD environments. Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-mock pytest-cov reportlab
      - run: pytest -v --cov=. --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Test Organization

```
rag-llm-test/
├── src/                     # Source code
│   ├── cli.py
│   ├── document_loader.py
│   ├── llm_integration.py
│   ├── rag_service.py
│   └── token_logger.py
├── tests/                   # Test files
│   ├── test_rag_service.py      # RAG database tests (200+ lines)
│   ├── test_document_loader.py  # Document parsing tests (330+ lines)
│   └── test_cli.py              # CLI command tests (430+ lines)
├── pytest.ini               # Pytest configuration
└── TEST_README.md           # This file
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install all dependencies with `pip install -r requirements.txt`

2. **ChromaDB Permission Issues**: Tests use temporary directories, but ensure write permissions in test directory

3. **Embedding Model Download**: First test run may download sentence-transformers model (~80MB)

4. **PDF Creation Failures**: Install reportlab: `pip install reportlab`

5. **Slow Tests**: Use `-m "not slow"` to skip slow tests during development

## Writing New Tests

When adding new features, follow these guidelines:

1. **Use appropriate fixtures** for test data and mocking
2. **Mock external dependencies** (APIs, file I/O) where appropriate
3. **Test edge cases** (empty inputs, invalid formats, missing files)
4. **Add docstrings** explaining what each test validates
5. **Use descriptive test names** following `test_<functionality>_<scenario>` pattern
6. **Group related tests** in test classes

Example:
```python
class TestNewFeature:
    """Tests for new feature."""

    def test_feature_success_case(self, fixture):
        """Test that feature works correctly with valid input."""
        # Test implementation
        assert result == expected

    def test_feature_error_handling(self, fixture):
        """Test that feature handles errors appropriately."""
        with pytest.raises(ExpectedError):
            # Test implementation
```

## Test Statistics

- **Total Test Files**: 3
- **Total Test Functions**: 60+
- **Lines of Test Code**: ~960 LOC
- **Coverage Target**: >80% for core modules

## Contact

For questions or issues with the test suite, please open an issue in the repository.
