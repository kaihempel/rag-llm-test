"""
LLM integration module for generating responses using external models.
"""
import os
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from src.token_logger import get_logger

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from mistralai.client import MistralClient
except ImportError:
    MistralClient = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, context: List[str]) -> str:
        """Generate a response based on the prompt and context."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (default: gpt-3.5-turbo)
        """
        if openai is None:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str, context: List[str]) -> str:
        """Generate a response using OpenAI."""
        context_text = "\n\n".join([f"Document {i+1}:\n{ctx}" for i, ctx in enumerate(context)])

        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Use the context documents to answer the user's question accurately.
If the answer cannot be found in the context, say so clearly."""

        user_prompt = f"""Context documents:
{context_text}

Question: {prompt}

Please provide a detailed answer based on the context above."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        # Log token usage
        if response.usage:
            logger = get_logger()
            logger.log_usage(
                provider="openai",
                model=self.model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )

        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model to use (default: claude-3-5-sonnet-20241022)
        """
        if anthropic is None:
            raise ImportError("Anthropic library not installed. Install with: pip install anthropic")

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str, context: List[str]) -> str:
        """Generate a response using Anthropic Claude."""
        context_text = "\n\n".join([f"Document {i+1}:\n{ctx}" for i, ctx in enumerate(context)])

        user_prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context documents:
{context_text}

Question: {prompt}

Please provide a detailed answer based on the context above. If the answer cannot be found in the context, say so clearly."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        # Log token usage
        if response.usage:
            logger = get_logger()
            logger.log_usage(
                provider="anthropic",
                model=self.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )

        return response.content[0].text


class MistralProvider(LLMProvider):
    """Mistral AI LLM provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-large-latest"):
        """
        Initialize Mistral provider.

        Args:
            api_key: Mistral API key (or set MISTRAL_API_KEY env var)
            model: Model to use (default: mistral-large-latest)
        """
        if MistralClient is None:
            raise ImportError("Mistral library not installed. Install with: pip install mistralai")

        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key not provided")

        self.client = MistralClient(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str, context: List[str]) -> str:
        """Generate a response using Mistral AI."""
        context_text = "\n\n".join([f"Document {i+1}:\n{ctx}" for i, ctx in enumerate(context)])

        user_prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context documents:
{context_text}

Question: {prompt}

Please provide a detailed answer based on the context above. If the answer cannot be found in the context, say so clearly."""

        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        # Log token usage
        if hasattr(response, 'usage') and response.usage:
            logger = get_logger()
            logger.log_usage(
                provider="mistral",
                model=self.model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )

        return response.choices[0].message.content


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-pro"):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            model: Model to use (default: gemini-pro)
        """
        if genai is None:
            raise ImportError("Google Generative AI library not installed. Install with: pip install google-generativeai")

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not provided")

        genai.configure(api_key=self.api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str, context: List[str]) -> str:
        """Generate a response using Google Gemini."""
        context_text = "\n\n".join([f"Document {i+1}:\n{ctx}" for i, ctx in enumerate(context)])

        user_prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context documents:
{context_text}

Question: {prompt}

Please provide a detailed answer based on the context above. If the answer cannot be found in the context, say so clearly."""

        response = self.model.generate_content(user_prompt)

        # Log token usage
        # Note: Gemini's response metadata includes usage information
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            logger = get_logger()
            logger.log_usage(
                provider="gemini",
                model=self.model_name,
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count
            )

        return response.text


class LLMFactory:
    """Factory for creating LLM providers."""

    PROVIDERS = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'mistral': MistralProvider,
        'gemini': GeminiProvider,
    }

    @staticmethod
    def create_provider(provider_name: str, **kwargs) -> LLMProvider:
        """
        Create an LLM provider.

        Args:
            provider_name: Name of the provider (openai, anthropic, mistral, gemini)
            **kwargs: Additional arguments for the provider

        Returns:
            LLM provider instance

        Raises:
            ValueError: If provider is not supported
        """
        provider_class = LLMFactory.PROVIDERS.get(provider_name.lower())

        if not provider_class:
            raise ValueError(
                f"Unsupported provider: {provider_name}. "
                f"Available providers: {', '.join(LLMFactory.PROVIDERS.keys())}"
            )

        return provider_class(**kwargs)


def generate_rag_response(
    query: str,
    retrieved_contexts: List[Dict],
    provider: LLMProvider,
    max_contexts: int = 3
) -> Dict[str, any]:
    """
    Generate a RAG response using retrieved contexts and an LLM.

    Args:
        query: User query
        retrieved_contexts: List of retrieved context documents
        provider: LLM provider to use
        max_contexts: Maximum number of contexts to use

    Returns:
        Dictionary with response and metadata
    """
    if not retrieved_contexts:
        return {
            'response': "I couldn't find any relevant information in the documents to answer your question.",
            'sources': []
        }

    # Extract top contexts
    contexts = [ctx['content'] for ctx in retrieved_contexts[:max_contexts]]

    # Generate response
    response = provider.generate(query, contexts)

    # Collect source information
    sources = []
    for ctx in retrieved_contexts[:max_contexts]:
        sources.append({
            'filename': ctx['metadata'].get('filename', 'Unknown'),
            'chunk_index': ctx['metadata'].get('chunk_index', 0)
        })

    return {
        'response': response,
        'sources': sources,
        'num_contexts_used': len(contexts)
    }
