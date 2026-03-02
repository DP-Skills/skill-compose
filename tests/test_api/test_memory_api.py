"""
Tests for Memory API endpoints.

Endpoints tested:
- GET    /api/v1/memory/files
- GET    /api/v1/memory/files/{scope}/{filename}
- PUT    /api/v1/memory/files/{scope}/{filename}
- DELETE /api/v1/memory/files/{scope}/{filename}
- GET    /api/v1/memory/entries
- POST   /api/v1/memory/entries
- PUT    /api/v1/memory/entries/{id}
- DELETE /api/v1/memory/entries/{id}
- POST   /api/v1/memory/search
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient

API = "/api/v1/memory"


# ─── Bootstrap File Endpoints ─────────────────────────────────


class TestListBootstrapFiles:
    """Tests for GET /api/v1/memory/files."""

    async def test_list_files(self, client: AsyncClient):
        """Should return file list."""
        with patch("app.services.memory_service.list_bootstrap_files", return_value=[
            {"filename": "SOUL.md", "global_exists": False, "agent_exists": False, "effective_scope": None, "size": 0},
        ]):
            response = await client.get(f"{API}/files")
            assert response.status_code == 200
            assert "files" in response.json()

    async def test_list_files_with_agent_id(self, client: AsyncClient):
        """Should pass agent_id to service."""
        agent_id = "12345678-1234-1234-1234-123456789abc"
        with patch("app.services.memory_service.list_bootstrap_files", return_value=[]) as mock:
            response = await client.get(f"{API}/files", params={"agent_id": agent_id})
            assert response.status_code == 200
            mock.assert_called_once_with(agent_id)

    async def test_list_files_invalid_agent_id(self, client: AsyncClient):
        """Non-UUID agent_id should return 400."""
        response = await client.get(f"{API}/files", params={"agent_id": "../../etc"})
        assert response.status_code == 400


class TestReadBootstrapFile:
    """Tests for GET /api/v1/memory/files/{scope}/{filename}."""

    async def test_read_file(self, client: AsyncClient):
        """Should return file content."""
        with patch("app.services.memory_service.read_bootstrap_file", return_value="hello"):
            response = await client.get(f"{API}/files/global/SOUL.md")
            assert response.status_code == 200
            data = response.json()
            assert data["content"] == "hello"
            assert data["scope"] == "global"
            assert data["filename"] == "SOUL.md"

    async def test_read_file_not_found(self, client: AsyncClient):
        """Non-existent file should return 404."""
        with patch("app.services.memory_service.read_bootstrap_file", return_value=None):
            response = await client.get(f"{API}/files/global/SOUL.md")
            assert response.status_code == 404

    async def test_read_file_invalid_scope(self, client: AsyncClient):
        """Invalid scope should return 400."""
        response = await client.get(f"{API}/files/not-a-uuid/SOUL.md")
        assert response.status_code == 400


class TestWriteBootstrapFile:
    """Tests for PUT /api/v1/memory/files/{scope}/{filename}."""

    async def test_write_file(self, client: AsyncClient):
        """Should write file and return success."""
        with patch("app.services.memory_service.write_bootstrap_file", return_value=True):
            response = await client.put(
                f"{API}/files/global/SOUL.md",
                json={"content": "new content"},
            )
            assert response.status_code == 200
            assert response.json()["success"] is True

    async def test_write_file_invalid_filename(self, client: AsyncClient):
        """Invalid filename should return 400."""
        response = await client.put(
            f"{API}/files/global/EVIL.md",
            json={"content": "hack"},
        )
        assert response.status_code == 400

    async def test_write_file_invalid_scope(self, client: AsyncClient):
        """Invalid scope should return 400."""
        response = await client.put(
            f"{API}/files/not-a-uuid/SOUL.md",
            json={"content": "content"},
        )
        assert response.status_code == 400

    async def test_write_file_failure(self, client: AsyncClient):
        """Write failure should return 500."""
        with patch("app.services.memory_service.write_bootstrap_file", return_value=False):
            response = await client.put(
                f"{API}/files/global/SOUL.md",
                json={"content": "content"},
            )
            assert response.status_code == 500


class TestDeleteBootstrapFile:
    """Tests for DELETE /api/v1/memory/files/{scope}/{filename}."""

    async def test_delete_file(self, client: AsyncClient):
        """Should delete file and return success."""
        with patch("app.services.memory_service.delete_bootstrap_file", return_value=True):
            response = await client.delete(f"{API}/files/global/SOUL.md")
            assert response.status_code == 200
            assert response.json()["success"] is True

    async def test_delete_file_not_found(self, client: AsyncClient):
        """Non-existent file should return 404."""
        with patch("app.services.memory_service.delete_bootstrap_file", return_value=False):
            response = await client.delete(f"{API}/files/global/SOUL.md")
            assert response.status_code == 404


# ─── Memory Entry Endpoints ───────────────────────────────────


class TestCreateEntry:
    """Tests for POST /api/v1/memory/entries."""

    async def test_create_entry(self, client: AsyncClient):
        """Should create an entry and return it."""
        mock_entry = {
            "id": "test-id",
            "content": "Test fact",
            "agent_id": None,
            "category": "fact",
            "source": "manual",
            "embedding_model": None,
            "session_id": None,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }
        with patch("app.services.memory_service.create_entry", new_callable=AsyncMock, return_value=mock_entry):
            response = await client.post(
                f"{API}/entries",
                json={"content": "Test fact", "category": "fact"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["content"] == "Test fact"
            assert data["category"] == "fact"

    async def test_create_entry_invalid_category(self, client: AsyncClient):
        """Invalid category should return 422."""
        response = await client.post(
            f"{API}/entries",
            json={"content": "Test", "category": "nonsense"},
        )
        assert response.status_code == 422

    async def test_create_entry_content_too_long(self, client: AsyncClient):
        """Content exceeding max_length should return 422."""
        response = await client.post(
            f"{API}/entries",
            json={"content": "x" * 5000, "category": "fact"},
        )
        assert response.status_code == 422

    async def test_create_entry_empty_content(self, client: AsyncClient):
        """Empty content should still be accepted (Pydantic allows empty string)."""
        mock_entry = {
            "id": "test-id", "content": "", "agent_id": None,
            "category": "fact", "source": "manual", "embedding_model": None,
            "session_id": None, "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }
        with patch("app.services.memory_service.create_entry", new_callable=AsyncMock, return_value=mock_entry):
            response = await client.post(
                f"{API}/entries",
                json={"content": "", "category": "fact"},
            )
            assert response.status_code == 200


class TestListEntries:
    """Tests for GET /api/v1/memory/entries."""

    async def test_list_entries(self, client: AsyncClient):
        """Should return entries with total count."""
        mock_result = {"entries": [], "total": 0}
        with patch("app.services.memory_service.list_entries", new_callable=AsyncMock, return_value=mock_result):
            response = await client.get(f"{API}/entries")
            assert response.status_code == 200
            data = response.json()
            assert "entries" in data
            assert "total" in data

    async def test_list_entries_with_filters(self, client: AsyncClient):
        """Should pass filters to service."""
        mock_result = {"entries": [], "total": 0}
        agent_id = "12345678-1234-1234-1234-123456789abc"
        with patch("app.services.memory_service.list_entries", new_callable=AsyncMock, return_value=mock_result) as mock:
            response = await client.get(
                f"{API}/entries",
                params={"agent_id": agent_id, "category": "fact", "limit": 10, "offset": 5},
            )
            assert response.status_code == 200
            mock.assert_called_once_with(
                agent_id=agent_id, category="fact", limit=10, offset=5
            )

    async def test_list_entries_invalid_agent_id(self, client: AsyncClient):
        """Non-UUID agent_id should return 400."""
        response = await client.get(
            f"{API}/entries",
            params={"agent_id": "not-a-uuid"},
        )
        assert response.status_code == 400


class TestUpdateEntry:
    """Tests for PUT /api/v1/memory/entries/{id}."""

    async def test_update_entry(self, client: AsyncClient):
        """Should update and return entry."""
        mock_entry = {
            "id": "test-id", "content": "Updated", "agent_id": None,
            "category": "preference", "source": "manual", "embedding_model": None,
            "session_id": None, "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }
        with patch("app.services.memory_service.update_entry", new_callable=AsyncMock, return_value=mock_entry):
            response = await client.put(
                f"{API}/entries/test-id",
                json={"content": "Updated", "category": "preference"},
            )
            assert response.status_code == 200
            assert response.json()["content"] == "Updated"

    async def test_update_entry_not_found(self, client: AsyncClient):
        """Non-existent entry should return 404."""
        with patch("app.services.memory_service.update_entry", new_callable=AsyncMock, return_value=None):
            response = await client.put(
                f"{API}/entries/nonexistent",
                json={"content": "Updated"},
            )
            assert response.status_code == 404

    async def test_update_entry_invalid_category(self, client: AsyncClient):
        """Invalid category should return 422."""
        response = await client.put(
            f"{API}/entries/test-id",
            json={"category": "invalid_cat"},
        )
        assert response.status_code == 422


class TestDeleteEntry:
    """Tests for DELETE /api/v1/memory/entries/{id}."""

    async def test_delete_entry(self, client: AsyncClient):
        """Should delete and return success."""
        with patch("app.services.memory_service.delete_entry", new_callable=AsyncMock, return_value=True):
            response = await client.delete(f"{API}/entries/test-id")
            assert response.status_code == 200
            assert response.json()["success"] is True

    async def test_delete_entry_not_found(self, client: AsyncClient):
        """Non-existent entry should return 404."""
        with patch("app.services.memory_service.delete_entry", new_callable=AsyncMock, return_value=False):
            response = await client.delete(f"{API}/entries/nonexistent")
            assert response.status_code == 404


# ─── Search Endpoint ──────────────────────────────────────────


class TestSearchMemory:
    """Tests for POST /api/v1/memory/search."""

    async def test_search(self, client: AsyncClient):
        """Should return search results."""
        mock_results = [
            {"id": "1", "content": "Python is great", "category": "fact",
             "agent_id": None, "source": "manual", "created_at": "2024-01-01T00:00:00",
             "similarity": 0.95},
        ]
        with patch("app.services.memory_service.search_memory", new_callable=AsyncMock, return_value=mock_results):
            response = await client.post(
                f"{API}/search",
                json={"query": "Python", "top_k": 5},
            )
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert data["query"] == "Python"
            assert len(data["results"]) == 1

    async def test_search_empty_results(self, client: AsyncClient):
        """No matches should return empty results."""
        with patch("app.services.memory_service.search_memory", new_callable=AsyncMock, return_value=[]):
            response = await client.post(
                f"{API}/search",
                json={"query": "nonexistent"},
            )
            assert response.status_code == 200
            assert response.json()["results"] == []

    async def test_search_top_k_validation(self, client: AsyncClient):
        """top_k out of range should return 422."""
        response = await client.post(
            f"{API}/search",
            json={"query": "test", "top_k": 0},
        )
        assert response.status_code == 422

        response = await client.post(
            f"{API}/search",
            json={"query": "test", "top_k": 200},
        )
        assert response.status_code == 422
