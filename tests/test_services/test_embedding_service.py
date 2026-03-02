"""
Tests for embedding_service.

Tests client initialization, fallback when no API key is configured,
model name resolution, and embed/aembed functions.
"""

from unittest.mock import patch, MagicMock

import pytest

from app.services import embedding_service


@pytest.fixture(autouse=True)
def _reset_embedding_client():
    """Reset the singleton client before each test."""
    embedding_service.reset_client()
    yield
    embedding_service.reset_client()


class TestClientInitialization:
    """Tests for _get_client() lazy initialization."""

    def test_no_api_key_returns_none(self):
        """When no embedding API key is configured, client should be None."""
        with patch.object(embedding_service, "read_env_value", return_value=None):
            client = embedding_service._get_client()
            assert client is None

    def test_caches_after_first_call(self):
        """Second call should return cached result without re-initializing."""
        with patch.object(embedding_service, "read_env_value", return_value=None):
            embedding_service._get_client()

        # Second call — read_env_value should NOT be called again
        with patch.object(embedding_service, "read_env_value") as mock_read:
            client = embedding_service._get_client()
            assert client is None
            mock_read.assert_not_called()

    def test_with_api_key_creates_client(self):
        """When API key is set, should create an OpenAI client."""
        def fake_read(key):
            if key == "EMBEDDING_API_KEY":
                return "test-key-123"
            return None

        mock_openai = MagicMock()
        with (
            patch.object(embedding_service, "read_env_value", side_effect=fake_read),
            patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai)}),
        ):
            # Re-import to pick up patched module
            embedding_service.reset_client()
            client = embedding_service._get_client()
            # Client was created (mock_openai was called)
            assert mock_openai.called or client is not None


class TestGetModel:
    """Tests for get_model()."""

    def test_returns_setting_default(self):
        """When no cached model, returns the config default."""
        model = embedding_service.get_model()
        assert isinstance(model, str)
        assert len(model) > 0

    def test_returns_cached_model_after_init(self):
        """After client init caches model, get_model() returns cached value."""
        embedding_service._cached_model = "text-embedding-custom"
        assert embedding_service.get_model() == "text-embedding-custom"


class TestEmbed:
    """Tests for embed() and aembed()."""

    def test_no_client_returns_nones(self):
        """Without a configured client, embed returns [None] for each input."""
        with patch.object(embedding_service, "read_env_value", return_value=None):
            results = embedding_service.embed(["hello", "world"])
            assert results == [None, None]

    def test_empty_input_returns_empty(self):
        """Empty text list returns empty list of Nones."""
        with patch.object(embedding_service, "read_env_value", return_value=None):
            results = embedding_service.embed([])
            assert results == []

    def test_embed_single_no_client(self):
        """embed_single returns None when no client is configured."""
        with patch.object(embedding_service, "read_env_value", return_value=None):
            result = embedding_service.embed_single("test text")
            assert result is None

    @pytest.mark.asyncio
    async def test_aembed_single_no_client(self):
        """aembed_single returns None when no client is configured."""
        with patch.object(embedding_service, "read_env_value", return_value=None):
            result = await embedding_service.aembed_single("test text")
            assert result is None

    @pytest.mark.asyncio
    async def test_aembed_no_client(self):
        """aembed returns [None, None] when no client is configured."""
        with patch.object(embedding_service, "read_env_value", return_value=None):
            results = await embedding_service.aembed(["a", "b"])
            assert results == [None, None]

    def test_api_failure_returns_nones(self):
        """If the embedding API call raises, returns [None] for each input."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("API error")

        embedding_service._client = mock_client
        embedding_service._client_initialized = True
        embedding_service._cached_model = "test-model"

        results = embedding_service.embed(["hello"])
        assert results == [None]


class TestResetClient:
    """Tests for reset_client()."""

    def test_reset_clears_state(self):
        """reset_client should clear all cached state."""
        embedding_service._client = "fake"
        embedding_service._client_initialized = True
        embedding_service._cached_model = "test-model"

        embedding_service.reset_client()

        assert embedding_service._client is None
        assert embedding_service._client_initialized is False
        assert embedding_service._cached_model is None
