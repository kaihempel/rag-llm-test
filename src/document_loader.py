"""
Document loader module for various file formats (PDF, DOC, ODT, CSV).
"""
import os
from typing import List, Dict
from pathlib import Path

import PyPDF2
from docx import Document
from odf import text, teletype
from odf.opendocument import load as load_odf
import pandas as pd


class DocumentLoader:
    """Handles loading and parsing of various document formats."""

    SUPPORTED_FORMATS = ['.pdf', '.doc', '.docx', '.odt', '.csv']

    @staticmethod
    def load_document(file_path: str) -> Dict[str, str]:
        """
        Load a document and extract its text content.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary with 'content' and 'metadata' keys

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lower()

        if extension not in DocumentLoader.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {', '.join(DocumentLoader.SUPPORTED_FORMATS)}"
            )

        loaders = {
            '.pdf': DocumentLoader._load_pdf,
            '.doc': DocumentLoader._load_doc,
            '.docx': DocumentLoader._load_docx,
            '.odt': DocumentLoader._load_odt,
            '.csv': DocumentLoader._load_csv,
        }

        loader = loaders.get(extension)
        content = loader(file_path)

        return {
            'content': content,
            'metadata': {
                'source': str(path.absolute()),
                'filename': path.name,
                'extension': extension,
                'size': path.stat().st_size
            }
        }

    @staticmethod
    def _load_pdf(file_path: str) -> str:
        """Extract text from PDF file."""
        text_content = []

        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())

        return '\n'.join(text_content)

    @staticmethod
    def _load_docx(file_path: str) -> str:
        """Extract text from DOCX file."""
        doc = Document(file_path)
        paragraphs = [paragraph.text for paragraph in doc.paragraphs]
        return '\n'.join(paragraphs)

    @staticmethod
    def _load_doc(file_path: str) -> str:
        """
        Extract text from DOC file.
        Note: python-docx can handle some .doc files, but may not work for all.
        """
        try:
            return DocumentLoader._load_docx(file_path)
        except Exception as e:
            raise ValueError(
                f"Unable to process .doc file. Please convert to .docx format. Error: {str(e)}"
            )

    @staticmethod
    def _load_odt(file_path: str) -> str:
        """Extract text from ODT file."""
        doc = load_odf(file_path)
        all_paragraphs = doc.getElementsByType(text.P)
        text_content = [teletype.extractText(p) for p in all_paragraphs]
        return '\n'.join(text_content)

    @staticmethod
    def _load_csv(file_path: str) -> str:
        """Extract text from CSV file."""
        df = pd.read_csv(file_path)
        # Convert DataFrame to a readable text format
        return df.to_string(index=False)
