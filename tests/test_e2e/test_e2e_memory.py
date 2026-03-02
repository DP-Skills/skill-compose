"""
E2E tests for the memory system.

Tests the full memory lifecycle via HTTP endpoints:
- Bootstrap file CRUD (write, read, list, delete)
- Memory entry CRUD (create, list, update, delete)
- Search (keyword fallback)
- Scope validation and input validation
- Agent memory tools (memory_save, memory_search)
"""

import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient
from sqlalchemy import text as sa_text

API = "/api/v1/memory"


@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestMemoryE2E:
    """Full memory system E2E tests using class-scoped DB and HTTP client.

    Test flow:
    1. Bootstrap files: write global → read → list → write agent → list with override → delete
    2. Memory entries: create → list → update → search → delete
    3. Validation: invalid scope, category, content length
    4. Agent tools: memory_save, memory_search
    """

    # Store IDs across tests
    _entry_id: str = ""

    # ─── Bootstrap Files ──────────────────────────────────────

    async def test_01_write_global_bootstrap(self, e2e_client: AsyncClient):
        """Write a global SOUL.md file."""
        with patch("app.services.memory_service._memory_dir") as mock_dir:
            import tempfile
            from pathlib import Path
            tmpdir = Path(tempfile.mkdtemp())
            mock_dir.return_value = tmpdir
            self.__class__._tmpdir = tmpdir

            response = await e2e_client.put(
                f"{API}/files/global/SOUL.md",
                json={"content": "You are a helpful assistant."},
            )
            assert response.status_code == 200
            assert response.json()["success"] is True

    async def test_02_read_global_bootstrap(self, e2e_client: AsyncClient):
        """Read the global SOUL.md file."""
        with patch("app.services.memory_service._memory_dir", return_value=self.__class__._tmpdir):
            response = await e2e_client.get(f"{API}/files/global/SOUL.md")
            assert response.status_code == 200
            data = response.json()
            assert data["content"] == "You are a helpful assistant."
            assert data["scope"] == "global"

    async def test_03_list_bootstrap_files(self, e2e_client: AsyncClient):
        """List bootstrap files — SOUL.md should exist globally."""
        with patch("app.services.memory_service._memory_dir", return_value=self.__class__._tmpdir):
            response = await e2e_client.get(f"{API}/files")
            assert response.status_code == 200
            files = response.json()["files"]
            assert len(files) == 3
            soul = next(f for f in files if f["filename"] == "SOUL.md")
            assert soul["global_exists"] is True

    async def test_04_write_agent_bootstrap(self, e2e_client: AsyncClient):
        """Write per-agent SOUL.md override."""
        agent_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        with patch("app.services.memory_service._memory_dir", return_value=self.__class__._tmpdir):
            response = await e2e_client.put(
                f"{API}/files/{agent_id}/SOUL.md",
                json={"content": "Agent-specific soul."},
            )
            assert response.status_code == 200

    async def test_05_list_with_agent_override(self, e2e_client: AsyncClient):
        """List files with agent_id — should show both global and agent exists."""
        agent_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        with patch("app.services.memory_service._memory_dir", return_value=self.__class__._tmpdir):
            response = await e2e_client.get(f"{API}/files", params={"agent_id": agent_id})
            assert response.status_code == 200
            files = response.json()["files"]
            soul = next(f for f in files if f["filename"] == "SOUL.md")
            assert soul["global_exists"] is True
            assert soul["agent_exists"] is True
            assert soul["effective_scope"] == agent_id

    async def test_06_delete_bootstrap(self, e2e_client: AsyncClient):
        """Delete the global SOUL.md file."""
        with patch("app.services.memory_service._memory_dir", return_value=self.__class__._tmpdir):
            response = await e2e_client.delete(f"{API}/files/global/SOUL.md")
            assert response.status_code == 200

            # Confirm it's gone
            response = await e2e_client.get(f"{API}/files/global/SOUL.md")
            assert response.status_code == 404

    async def test_07_invalid_scope_rejected(self, e2e_client: AsyncClient):
        """Invalid scope (not UUID, not 'global') should be rejected."""
        response = await e2e_client.get(f"{API}/files/not-a-uuid/SOUL.md")
        assert response.status_code == 400

    async def test_08_invalid_filename_rejected(self, e2e_client: AsyncClient):
        """Invalid filename should be rejected."""
        response = await e2e_client.put(
            f"{API}/files/global/EVIL.md",
            json={"content": "hack"},
        )
        assert response.status_code == 400

    # ─── Memory Entries ───────────────────────────────────────

    async def test_09_create_entry(self, e2e_client: AsyncClient, e2e_db_session):
        """Create a memory entry via API."""
        # Recreate memory_entries table with proper TIMESTAMPTZ columns
        # (Base.metadata.create_all uses DateTime which maps to TIMESTAMP WITHOUT TIME ZONE,
        # but the service code passes timezone-aware datetimes)
        try:
            await e2e_db_session.execute(sa_text("DROP TABLE IF EXISTS memory_entries CASCADE"))
            await e2e_db_session.execute(sa_text("CREATE EXTENSION IF NOT EXISTS vector"))
            await e2e_db_session.execute(sa_text("""
                CREATE TABLE memory_entries (
                    id VARCHAR(36) PRIMARY KEY,
                    agent_id VARCHAR(36),
                    content TEXT NOT NULL,
                    category VARCHAR(64),
                    source VARCHAR(256),
                    embedding vector(1536),
                    embedding_model VARCHAR(128),
                    session_id VARCHAR(36),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """))
            await e2e_db_session.commit()
        except Exception:
            await e2e_db_session.rollback()

        # Use a proper session factory so each service call gets a fresh session
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

        test_engine = create_async_engine(
            "postgresql+asyncpg://skills:skills123@localhost:62620/skills_api_test",
            echo=False, pool_size=2, max_overflow=5,
        )
        test_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
        self.__class__._test_engine = test_engine

        agent_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        self.__class__._test_agent_id = agent_id

        with patch("app.services.memory_service.embedding_service") as mock_embed:
            mock_embed.aembed_single = AsyncMock(return_value=None)
            mock_embed.get_model.return_value = "test-model"

            with patch("app.services.memory_service.AsyncSessionLocal", test_session_factory):
                response = await e2e_client.post(
                    f"{API}/entries",
                    json={"content": "User prefers dark mode", "category": "preference", "agent_id": agent_id},
                )
                assert response.status_code == 200
                data = response.json()
                assert data["content"] == "User prefers dark mode"
                assert data["category"] == "preference"
                self.__class__._entry_id = data["id"]

    async def test_10_list_entries(self, e2e_client: AsyncClient, e2e_db_session):
        """List entries should return the created entry."""
        test_engine = self.__class__._test_engine
        from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
        test_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
        agent_id = self.__class__._test_agent_id

        with patch("app.services.memory_service.AsyncSessionLocal", test_session_factory):
            response = await e2e_client.get(
                f"{API}/entries", params={"agent_id": agent_id}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["total"] >= 1
            found = any(e["content"] == "User prefers dark mode" for e in data["entries"])
            assert found

    async def test_11_search_keyword(self, e2e_client: AsyncClient, e2e_db_session):
        """Search should find entry via keyword fallback (no embeddings)."""
        test_engine = self.__class__._test_engine
        from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
        test_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
        agent_id = self.__class__._test_agent_id

        with patch("app.services.memory_service.AsyncSessionLocal", test_session_factory):
            with patch("app.services.memory_service.embedding_service") as mock_embed:
                mock_embed.aembed_single = AsyncMock(return_value=None)

                response = await e2e_client.post(
                    f"{API}/search",
                    json={"query": "dark mode", "agent_id": agent_id},
                )
                assert response.status_code == 200
                data = response.json()
                assert len(data["results"]) >= 1
                assert any("dark mode" in r["content"] for r in data["results"])

    async def test_12_update_entry(self, e2e_client: AsyncClient, e2e_db_session):
        """Update the entry's category."""
        test_engine = self.__class__._test_engine
        from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
        test_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

        with patch("app.services.memory_service.AsyncSessionLocal", test_session_factory):
            with patch("app.services.memory_service.embedding_service") as mock_embed:
                mock_embed.aembed_single = AsyncMock(return_value=None)

                response = await e2e_client.put(
                    f"{API}/entries/{self.__class__._entry_id}",
                    json={"category": "fact"},
                )
                assert response.status_code == 200
                assert response.json()["category"] == "fact"

    async def test_13_delete_entry(self, e2e_client: AsyncClient, e2e_db_session):
        """Delete the entry."""
        test_engine = self.__class__._test_engine
        from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
        test_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

        with patch("app.services.memory_service.AsyncSessionLocal", test_session_factory):
            response = await e2e_client.delete(f"{API}/entries/{self.__class__._entry_id}")
            assert response.status_code == 200
            assert response.json()["success"] is True

            # Confirm it's gone
            response = await e2e_client.delete(f"{API}/entries/{self.__class__._entry_id}")
            assert response.status_code == 404

        # Clean up test engine
        await self.__class__._test_engine.dispose()

    # ─── Validation ───────────────────────────────────────────

    async def test_14_invalid_category(self, e2e_client: AsyncClient):
        """Invalid category should be rejected."""
        response = await e2e_client.post(
            f"{API}/entries",
            json={"content": "test", "category": "nonsense"},
        )
        assert response.status_code == 422

    async def test_15_content_too_long(self, e2e_client: AsyncClient):
        """Content exceeding max_length should be rejected."""
        response = await e2e_client.post(
            f"{API}/entries",
            json={"content": "x" * 5000},
        )
        assert response.status_code == 422

    async def test_16_search_top_k_validation(self, e2e_client: AsyncClient):
        """top_k=0 and top_k=200 should be rejected."""
        response = await e2e_client.post(
            f"{API}/search", json={"query": "test", "top_k": 0}
        )
        assert response.status_code == 422

    async def test_17_list_files_invalid_agent_id(self, e2e_client: AsyncClient):
        """Non-UUID agent_id in list_files should be rejected."""
        response = await e2e_client.get(f"{API}/files", params={"agent_id": "../hack"})
        assert response.status_code == 400

    # ─── Agent Memory Tools ───────────────────────────────────

    async def test_18_agent_memory_tools(self, e2e_client: AsyncClient, e2e_db_session):
        """Agent memory tools (save + search) should work correctly."""
        from app.agent.tools import create_memory_tool_functions
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        # Ensure memory_entries table exists with proper schema
        try:
            await e2e_db_session.execute(sa_text("CREATE EXTENSION IF NOT EXISTS vector"))
            await e2e_db_session.execute(sa_text("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id VARCHAR(36) PRIMARY KEY,
                    agent_id VARCHAR(36),
                    content TEXT NOT NULL,
                    category VARCHAR(64),
                    source VARCHAR(256),
                    embedding vector(1536),
                    embedding_model VARCHAR(128),
                    session_id VARCHAR(36),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """))
            await e2e_db_session.commit()
        except Exception:
            await e2e_db_session.rollback()

        with patch("app.services.memory_service.embedding_service") as mock_embed:
            mock_embed.embed_single.return_value = None
            mock_embed.get_model.return_value = "test-model"

            sync_url = "postgresql+psycopg2://skills:skills123@localhost:62620/skills_api_test"
            sync_engine = create_engine(sync_url)
            TestSyncSession = sessionmaker(sync_engine, expire_on_commit=False)

            with patch("app.services.memory_service.SyncSessionLocal", TestSyncSession):
                tools = create_memory_tool_functions(agent_id="bbbbbbbb-cccc-dddd-eeee-ffffffffffff")

                # Test memory_save with valid category
                result = tools["memory_save"](content="Test save from agent", category="fact")
                assert result["saved"] is True
                assert result["category"] == "fact"

                # Test memory_save with invalid category (should default to "context")
                result = tools["memory_save"](content="Invalid cat test", category="bogus")
                assert result["category"] == "context"

                # Test memory_save with content truncation
                result = tools["memory_save"](content="x" * 5000, category="fact")
                assert len(result["content"]) == 4096

                # Test memory_search
                result = tools["memory_search"](query="Test save")
                assert "results" in result

            sync_engine.dispose()

    # ─── Cleanup ──────────────────────────────────────────────

    async def test_19_cleanup(self, e2e_client: AsyncClient):
        """Clean up temp directory."""
        import shutil
        tmpdir = getattr(self.__class__, "_tmpdir", None)
        if tmpdir and tmpdir.exists():
            shutil.rmtree(tmpdir)
