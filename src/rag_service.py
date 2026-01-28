"""
RAG (Retrieval-Augmented Generation) service for document storage and querying.
"""
from typing import List, Dict, Optional
import hashlib

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from document_loader import DocumentLoader


class RAGService:
    """Manages document storage and retrieval using ChromaDB."""

    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "documents"):
        """
        Initialize the RAG service.

        Args:
            db_path: Path to ChromaDB storage directory
            collection_name: Name of the collection to use
        """
        self.db_path = db_path
        self.collection_name = collection_name

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize embedding model (local)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG document collection"}
        )

        print(f"RAG service initialized. Database: {db_path}")
        print(f"Collection: {collection_name}")
        print(f"Documents in collection: {self.collection.count()}")

    def _generate_doc_id(self, file_path: str) -> str:
        """Generate a unique ID for a document based on its path."""
        return hashlib.md5(file_path.encode()).hexdigest()

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at a sentence or word boundary
            if end < len(text):
                # Look for last period, newline, or space
                for sep in ['. ', '\n', ' ']:
                    last_sep = chunk.rfind(sep)
                    if last_sep > chunk_size * 0.5:  # At least 50% of chunk size
                        chunk = chunk[:last_sep + len(sep)]
                        end = start + len(chunk)
                        break

            chunks.append(chunk.strip())
            start = end - overlap

        return [c for c in chunks if c]  # Filter empty chunks

    def add_document(self, file_path: str) -> Dict[str, any]:
        """
        Load and add a document to the RAG database.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary with status and metadata
        """
        # Load document
        doc_data = DocumentLoader.load_document(file_path)
        content = doc_data['content']
        metadata = doc_data['metadata']

        # Check if document already exists
        doc_id = self._generate_doc_id(metadata['source'])

        # Chunk the document
        chunks = self._chunk_text(content)

        if not chunks:
            return {
                'status': 'error',
                'message': 'No content could be extracted from the document'
            }

        # Generate embeddings
        print(f"Processing {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(chunks).tolist()

        # Prepare data for ChromaDB
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                **metadata,
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            for i in range(len(chunks))
        ]

        # Add to collection
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

        return {
            'status': 'success',
            'document_id': doc_id,
            'filename': metadata['filename'],
            'chunks': len(chunks),
            'total_documents': self.collection.count()
        }

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Query the RAG database for relevant documents.

        Args:
            query_text: Query string
            n_results: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of relevant document chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text]).tolist()

        # Query the collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=filter_metadata
        )

        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })

        return formatted_results

    def get_stats(self) -> Dict:
        """Get statistics about the RAG database."""
        return {
            'total_chunks': self.collection.count(),
            'collection_name': self.collection_name,
            'db_path': self.db_path
        }

    def delete_all(self):
        """Delete all documents from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"}
        )
