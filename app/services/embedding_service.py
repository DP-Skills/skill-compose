"""Embedding service for memory vector search.

Uses the OpenAI-compatible embedding API (works with OpenAI, Azure, or any
compatible endpoint). Falls back gracefully when no API key is configured.
"""

import asyncio
import logging
from typing import Optional

from app.config import get_settings, read_env_value

logger = logging.getLogger(__name__)

# Singleton client and cached config — lazily initialized
_client = None
_client_initialized = False
_cached_model: Optional[str] = None


def _get_client():
    """Get or create the OpenAI client for embeddings. Caches config on init."""
    global _client, _client_initialized, _cached_model
    if _client_initialized:
        return _client

    _client_initialized = True
    settings = get_settings()

    api_key = read_env_value("EMBEDDING_API_KEY") or read_env_value("OPENAI_API_KEY")
    if not api_key:
        logger.info("No embedding API key configured — vector search disabled")
        _client = None
        return None

    # Cache the model name at init time
    _cached_model = read_env_value("EMBEDDING_MODEL") or settings.embedding_model

    try:
        from openai import OpenAI
        kwargs = {"api_key": api_key}
        base_url = read_env_value("EMBEDDING_BASE_URL") or settings.embedding_base_url
        if base_url:
            kwargs["base_url"] = base_url
        _client = OpenAI(**kwargs)
        return _client
    except Exception as e:
        logger.warning(f"Failed to create embedding client: {e}")
        _client = None
        return None


def reset_client():
    """Reset the cached client and config (e.g. after config change)."""
    global _client, _client_initialized, _cached_model
    _client = None
    _client_initialized = False
    _cached_model = None


def get_model() -> str:
    """Get the cached embedding model name."""
    if _cached_model:
        return _cached_model
    settings = get_settings()
    return settings.embedding_model


def embed(texts: list[str]) -> list[Optional[list[float]]]:
    """Generate embeddings synchronously.

    Returns a list of embedding vectors (or None per item if embedding fails).
    """
    client = _get_client()
    if not client or not texts:
        return [None] * len(texts)

    try:
        response = client.embeddings.create(input=texts, model=get_model())
        result: list[Optional[list[float]]] = [None] * len(texts)
        for item in response.data:
            result[item.index] = item.embedding
        return result
    except Exception as e:
        logger.warning(f"Embedding API call failed: {e}")
        return [None] * len(texts)


async def aembed(texts: list[str]) -> list[Optional[list[float]]]:
    """Generate embeddings asynchronously (runs sync call in thread pool)."""
    return await asyncio.to_thread(embed, texts)


def embed_single(text: str) -> Optional[list[float]]:
    """Generate embedding for a single text. Returns None on failure."""
    results = embed([text])
    return results[0] if results else None


async def aembed_single(text: str) -> Optional[list[float]]:
    """Generate embedding for a single text asynchronously."""
    results = await aembed([text])
    return results[0] if results else None
