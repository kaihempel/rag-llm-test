#!/usr/bin/env python3
"""
RAG CLI - Command-line interface for local RAG implementation.
"""
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from rag_service import RAGService
from llm_integration import LLMFactory, generate_rag_response


# Load environment variables
load_dotenv()


@click.group()
@click.option('--db-path', default='./chroma_db', help='Path to ChromaDB storage directory')
@click.option('--collection', default='documents', help='Collection name')
@click.pass_context
def cli(ctx, db_path, collection):
    """
    RAG CLI - Local Retrieval-Augmented Generation system.

    Supports PDF, DOC, DOCX, ODT, and CSV documents.
    """
    ctx.ensure_object(dict)
    ctx.obj['db_path'] = db_path
    ctx.obj['collection'] = collection


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.pass_context
def add_document(ctx, file_path):
    """
    Add a document to the RAG database.

    FILE_PATH: Path to the document file (PDF, DOC, DOCX, ODT, or CSV)

    Example:
        python cli.py add_document my_document.pdf
    """
    try:
        click.echo(f"\n📄 Loading document: {file_path}")

        # Initialize RAG service
        rag = RAGService(
            db_path=ctx.obj['db_path'],
            collection_name=ctx.obj['collection']
        )

        # Add document
        result = rag.add_document(file_path)

        if result['status'] == 'success':
            click.echo(f"\n✅ Document added successfully!")
            click.echo(f"   Filename: {result['filename']}")
            click.echo(f"   Chunks created: {result['chunks']}")
            click.echo(f"   Total documents in database: {result['total_documents']}")
        else:
            click.echo(f"\n❌ Error: {result['message']}", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('query_text')
@click.option('--provider', default='openai', help='LLM provider (openai, anthropic)')
@click.option('--model', help='Model name (optional)')
@click.option('--api-key', help='API key (or set via environment variable)')
@click.option('--num-results', default=5, help='Number of context chunks to retrieve')
@click.option('--max-contexts', default=3, help='Maximum contexts to send to LLM')
@click.option('--show-sources', is_flag=True, help='Show source documents')
@click.pass_context
def query(ctx, query_text, provider, model, api_key, num_results, max_contexts, show_sources):
    """
    Query the RAG database and generate a response.

    QUERY_TEXT: Your question or query

    Example:
        python cli.py query "What is the main topic of the documents?"

    Environment variables:
        OPENAI_API_KEY: OpenAI API key
        ANTHROPIC_API_KEY: Anthropic API key
    """
    try:
        click.echo(f"\n🔍 Processing query: {query_text}\n")

        # Initialize RAG service
        rag = RAGService(
            db_path=ctx.obj['db_path'],
            collection_name=ctx.obj['collection']
        )

        # Check if database has documents
        stats = rag.get_stats()
        if stats['total_chunks'] == 0:
            click.echo("❌ No documents in database. Add documents first using 'add_document' command.", err=True)
            sys.exit(1)

        # Retrieve relevant contexts
        click.echo(f"📚 Retrieving relevant contexts...")
        contexts = rag.query(query_text, n_results=num_results)

        if not contexts:
            click.echo("⚠️  No relevant contexts found in the database.")
            sys.exit(0)

        click.echo(f"   Found {len(contexts)} relevant chunks\n")

        # Initialize LLM provider
        click.echo(f"🤖 Generating response using {provider}...")

        provider_kwargs = {}
        if api_key:
            provider_kwargs['api_key'] = api_key
        if model:
            provider_kwargs['model'] = model

        llm_provider = LLMFactory.create_provider(provider, **provider_kwargs)

        # Generate response
        result = generate_rag_response(
            query_text,
            contexts,
            llm_provider,
            max_contexts=max_contexts
        )

        # Display response
        click.echo("\n" + "="*60)
        click.echo("RESPONSE:")
        click.echo("="*60)
        click.echo(result['response'])
        click.echo("="*60)

        # Show sources if requested
        if show_sources and result['sources']:
            click.echo("\n📖 Sources:")
            for i, source in enumerate(result['sources'], 1):
                click.echo(f"   {i}. {source['filename']} (chunk {source['chunk_index']})")

        click.echo(f"\n✅ Used {result['num_contexts_used']} context(s) for generation\n")

    except ValueError as e:
        click.echo(f"\n❌ Configuration error: {str(e)}", err=True)
        click.echo("\nMake sure to set your API key:", err=True)
        click.echo("  - OpenAI: export OPENAI_API_KEY='your-key'", err=True)
        click.echo("  - Anthropic: export ANTHROPIC_API_KEY='your-key'", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def stats(ctx):
    """Show statistics about the RAG database."""
    try:
        rag = RAGService(
            db_path=ctx.obj['db_path'],
            collection_name=ctx.obj['collection']
        )

        stats = rag.get_stats()

        click.echo("\n📊 RAG Database Statistics:")
        click.echo(f"   Database path: {stats['db_path']}")
        click.echo(f"   Collection: {stats['collection_name']}")
        click.echo(f"   Total chunks: {stats['total_chunks']}\n")

    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to delete all documents?')
@click.pass_context
def clear(ctx):
    """Clear all documents from the RAG database."""
    try:
        rag = RAGService(
            db_path=ctx.obj['db_path'],
            collection_name=ctx.obj['collection']
        )

        rag.delete_all()
        click.echo("\n✅ All documents cleared from database.\n")

    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli(obj={})
