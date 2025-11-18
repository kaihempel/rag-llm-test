#!/usr/bin/env python3
"""
RAG CLI - Command-line interface for local RAG implementation.
"""
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from src.rag_service import RAGService
from src.llm_integration import LLMFactory, generate_rag_response
from src.token_logger import get_logger


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
@click.option('--provider', default='openai', help='LLM provider (openai, anthropic, mistral, gemini)')
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
        MISTRAL_API_KEY: Mistral API key
        GOOGLE_API_KEY: Google API key (for Gemini)
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
        click.echo("  - Mistral: export MISTRAL_API_KEY='your-key'", err=True)
        click.echo("  - Gemini: export GOOGLE_API_KEY='your-key'", err=True)
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


@cli.command()
@click.option('--date', help='Show usage for specific date (YYYY-MM-DD)')
@click.option('--log-dir', default='./logs', help='Directory containing log files')
@click.option('--show-costs', is_flag=True, help='Show estimated costs')
@click.pass_context
def usage(ctx, date, log_dir, show_costs):
    """
    Display token usage statistics.

    Shows token usage by provider and model. By default, shows all-time statistics.
    Use --date to filter by a specific date.

    Examples:
        python cli.py usage
        python cli.py usage --date 2025-01-15
        python cli.py usage --show-costs
    """
    try:
        logger = get_logger(log_dir)

        # Get log files
        log_files = logger.get_all_log_files()

        if not log_files:
            click.echo("\n📊 No usage data found.")
            click.echo(f"   Log directory: {log_dir}")
            click.echo(f"   Token usage will be logged automatically when you make queries.\n")
            return

        # Filter by date if specified
        if date:
            log_file = logger._get_log_file_path(date)
            if not log_file.exists():
                click.echo(f"\n❌ No usage data found for date: {date}\n", err=True)
                sys.exit(1)
            stats = logger.get_statistics(log_file)
            title = f"Token Usage Statistics for {date}"
        else:
            stats = logger.get_statistics()
            title = "All-Time Token Usage Statistics"

        # Display statistics
        click.echo(f"\n{'='*70}")
        click.echo(f"{title}")
        click.echo(f"{'='*70}\n")

        # Total statistics
        total = stats['total']
        click.echo("📊 TOTAL USAGE:")
        click.echo(f"   Requests:      {total['requests']:,}")
        click.echo(f"   Input tokens:  {total['input_tokens']:,}")
        click.echo(f"   Output tokens: {total['output_tokens']:,}")
        click.echo(f"   Total tokens:  {total['total_tokens']:,}")

        # Calculate total cost if requested
        if show_costs:
            total_cost = 0.0
            for provider, provider_stats in stats['by_provider'].items():
                for model, model_stats in provider_stats['by_model'].items():
                    cost = logger.get_cost_estimate(
                        provider,
                        model,
                        model_stats['input_tokens'],
                        model_stats['output_tokens']
                    )
                    if cost:
                        total_cost += cost

            if total_cost > 0:
                click.echo(f"   Estimated cost: ${total_cost:.4f}")

        click.echo()

        # By provider
        if stats['by_provider']:
            click.echo("📈 BY PROVIDER:\n")

            for provider, provider_stats in sorted(stats['by_provider'].items()):
                click.echo(f"  {provider.upper()}:")
                click.echo(f"    Requests:      {provider_stats['requests']:,}")
                click.echo(f"    Input tokens:  {provider_stats['input_tokens']:,}")
                click.echo(f"    Output tokens: {provider_stats['output_tokens']:,}")
                click.echo(f"    Total tokens:  {provider_stats['total_tokens']:,}")

                # Provider cost
                if show_costs:
                    provider_cost = 0.0
                    for model, model_stats in provider_stats['by_model'].items():
                        cost = logger.get_cost_estimate(
                            provider,
                            model,
                            model_stats['input_tokens'],
                            model_stats['output_tokens']
                        )
                        if cost:
                            provider_cost += cost

                    if provider_cost > 0:
                        click.echo(f"    Estimated cost: ${provider_cost:.4f}")

                # By model
                if provider_stats['by_model']:
                    click.echo(f"\n    Models:")
                    for model, model_stats in sorted(provider_stats['by_model'].items()):
                        click.echo(f"      • {model}:")
                        click.echo(f"          Requests:      {model_stats['requests']:,}")
                        click.echo(f"          Input tokens:  {model_stats['input_tokens']:,}")
                        click.echo(f"          Output tokens: {model_stats['output_tokens']:,}")
                        click.echo(f"          Total tokens:  {model_stats['total_tokens']:,}")

                        # Model cost
                        if show_costs:
                            model_cost = logger.get_cost_estimate(
                                provider,
                                model,
                                model_stats['input_tokens'],
                                model_stats['output_tokens']
                            )
                            if model_cost:
                                click.echo(f"          Estimated cost: ${model_cost:.4f}")
                            else:
                                click.echo(f"          Estimated cost: N/A (pricing unknown)")

                click.echo()

        # Show available log files
        if not date:
            click.echo("📁 AVAILABLE LOG FILES:\n")
            for log_file in log_files[:10]:  # Show up to 10 recent files
                file_stats = logger.get_statistics(log_file)
                file_date = log_file.stem.replace('token_usage_', '')
                click.echo(f"   {file_date}: {file_stats['total']['requests']} requests, "
                          f"{file_stats['total']['total_tokens']:,} tokens")

            if len(log_files) > 10:
                click.echo(f"   ... and {len(log_files) - 10} more")

        click.echo(f"\n{'='*70}\n")

    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    cli(obj={})
