# Local RAG Implementation

A local Retrieval-Augmented Generation (RAG) system built with Python that allows you to index documents and query them using external LLM models.

## Features

- **Multiple Document Formats**: Supports PDF, DOC, DOCX, ODT, and CSV files
- **Local Vector Database**: Uses ChromaDB for efficient document storage and retrieval
- **External LLM Integration**: Supports OpenAI GPT, Anthropic Claude, and Mistral AI models
- **Local Embeddings**: Uses sentence-transformers for local embedding generation
- **Simple CLI Interface**: Easy-to-use command-line interface built with Click
- **Document Chunking**: Intelligent text chunking with overlap for better context retrieval

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-llm-test
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys:

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and add your API key(s):
```bash
# For OpenAI
OPENAI_API_KEY=your-openai-api-key-here

# For Anthropic Claude
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# For Mistral AI
MISTRAL_API_KEY=your-mistral-api-key-here
```

Alternatively, export them directly:
```bash
export OPENAI_API_KEY='your-key-here'
# or
export ANTHROPIC_API_KEY='your-key-here'
# or
export MISTRAL_API_KEY='your-key-here'
```

## Usage

### Basic Commands

The CLI provides the following commands:

```bash
python src/cli.py [OPTIONS] COMMAND [ARGS]...
```

### Add Documents

Add a document to the RAG database:

```bash
python src/cli.py add_document <file_path>
```

**Examples:**
```bash
python src/cli.py add_document research_paper.pdf
python src/cli.py add_document data.csv
python src/cli.py add_document report.docx
```

### Query Documents

Query the database and get AI-generated responses:

```bash
python src/cli.py query "Your question here"
```

**Examples:**
```bash
# Using OpenAI (default)
python src/cli.py query "What are the main findings in the documents?"

# Using Anthropic Claude
python src/cli.py query "Summarize the key points" --provider anthropic

# Using Mistral AI
python src/cli.py query "What are the implications?" --provider mistral

# Show source documents
python src/cli.py query "What is the conclusion?" --show-sources

# Use specific model
python src/cli.py query "Explain the methodology" --provider openai --model gpt-4
```

**Query Options:**
- `--provider`: LLM provider (`openai`, `anthropic`, or `mistral`) - default: `openai`
- `--model`: Specific model name (optional)
- `--api-key`: API key (if not set via environment variable)
- `--num-results`: Number of context chunks to retrieve (default: 5)
- `--max-contexts`: Maximum contexts to send to LLM (default: 3)
- `--show-sources`: Display source documents used

### View Statistics

Check database statistics:

```bash
python src/cli.py stats
```

### Clear Database

Remove all documents from the database:

```bash
python src/cli.py clear
```

### Global Options

- `--db-path`: Path to ChromaDB storage directory (default: `./chroma_db`)
- `--collection`: Collection name (default: `documents`)

**Example:**
```bash
python src/cli.py --db-path ./my_custom_db --collection my_docs add_document paper.pdf
```

## Architecture

### Components

1. **Document Loader** (`src/document_loader.py`)
   - Handles loading and parsing of various document formats
   - Extracts text content from PDF, DOC, DOCX, ODT, and CSV files

2. **RAG Service** (`src/rag_service.py`)
   - Manages the ChromaDB vector database
   - Handles document chunking and embedding generation
   - Provides query functionality for context retrieval

3. **LLM Integration** (`src/llm_integration.py`)
   - Integrates with external LLM providers (OpenAI, Anthropic, Mistral)
   - Generates responses based on retrieved contexts
   - Supports multiple model configurations

4. **CLI Interface** (`src/cli.py`)
   - Command-line interface built with Click
   - User-friendly commands for all operations

### How It Works

1. **Document Ingestion**:
   - Documents are loaded and parsed
   - Text is split into overlapping chunks (default: 1000 chars with 200 char overlap)
   - Local embeddings are generated using sentence-transformers
   - Chunks and embeddings are stored in ChromaDB

2. **Querying**:
   - User query is embedded using the same model
   - Similar document chunks are retrieved from ChromaDB
   - Retrieved contexts are sent to the external LLM
   - LLM generates a response based on the provided context

## Supported File Formats

- **PDF** (.pdf): Extracted using PyPDF2
- **Word Documents** (.docx): Extracted using python-docx
- **Legacy Word** (.doc): Limited support via python-docx
- **OpenDocument** (.odt): Extracted using odfpy
- **CSV** (.csv): Parsed using pandas

## Configuration

### LLM Providers

**OpenAI:**
- Default model: `gpt-3.5-turbo`
- Requires: `OPENAI_API_KEY`
- Supported models: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`, etc.

**Anthropic:**
- Default model: `claude-3-5-sonnet-20241022`
- Requires: `ANTHROPIC_API_KEY`
- Supported models: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`, etc.

**Mistral AI:**
- Default model: `mistral-large-latest`
- Requires: `MISTRAL_API_KEY`
- Supported models: `mistral-large-latest`, `mistral-medium-latest`, `mistral-small-latest`, etc.

### Embedding Model

- Uses `all-MiniLM-L6-v2` from sentence-transformers
- Downloaded automatically on first run
- Runs locally, no API key required

## Example Workflow

```bash
# 1. Add some documents
python src/cli.py add_document research_paper.pdf
python src/cli.py add_document data_analysis.csv
python src/cli.py add_document meeting_notes.docx

# 2. Check statistics
python src/cli.py stats

# 3. Query your documents
python src/cli.py query "What are the main findings?" --show-sources

# 4. Use a different provider
python src/cli.py query "Summarize the data" --provider anthropic

# 5. Clear database when done
python src/cli.py clear
```

## Troubleshooting

### Common Issues

1. **API Key Error**:
   - Ensure your API key is set correctly in `.env` or exported as an environment variable
   - Verify the key has the correct format and permissions

2. **Import Errors**:
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Activate your virtual environment if using one

3. **Document Loading Fails**:
   - Check that the file format is supported
   - For .doc files, try converting to .docx format
   - Ensure the file is not corrupted

4. **No Results Found**:
   - Make sure documents have been added to the database first
   - Check that documents contain relevant content
   - Try adjusting `--num-results` parameter

## Development

### Project Structure

```
rag-llm-test/
├── cli.py                  # Main CLI interface
├── document_loader.py      # Document parsing
├── rag_service.py         # Vector database operations
├── llm_integration.py     # LLM provider integrations
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variable template
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

### Adding New Document Formats

To add support for a new document format:

1. Add the file extension to `SUPPORTED_FORMATS` in `document_loader.py`
2. Implement a new loader method (e.g., `_load_xml`)
3. Add the loader to the `loaders` dictionary in `load_document()`

### Adding New LLM Providers

To add a new LLM provider:

1. Create a new provider class in `llm_integration.py` that inherits from `LLMProvider`
2. Implement the `generate()` method
3. Register the provider in `LLMFactory.PROVIDERS`

## License

This project is provided as-is for educational and development purposes.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
