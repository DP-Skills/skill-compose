"""
Tests for memory_service.

Tests bootstrap file operations (CRUD, truncation, override logic),
memory entry CRUD, keyword search fallback, and flush deduplication.
"""

import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from sqlalchemy import text, Column, String, Text, DateTime
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.database import Base
from app.services import memory_service

TEST_DATABASE_URL = "postgresql+asyncpg://skills:skills123@localhost:62620/skills_api_test"


@pytest_asyncio.fixture
async def memory_db_session():
    """DB session that creates the memory_entries table without the vector column.

    The test PostgreSQL instance doesn't have pgvector installed, so we create
    the table via raw SQL (matching the production migration but without embedding).
    """
    engine = create_async_engine(TEST_DATABASE_URL, echo=False, pool_size=2, max_overflow=5)

    # Create memory_entries table with pgvector column type.
    # The test PostgreSQL has the pgvector extension installed.
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("""
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

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    session = session_factory()

    try:
        yield session
    finally:
        await session.close()
        async with engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS memory_entries CASCADE"))
        await engine.dispose()


# ─── Bootstrap File Tests ─────────────────────────────────────


class TestTruncateContent:
    """Tests for _truncate_content()."""

    def test_short_content_unchanged(self):
        """Content within limit should not be truncated."""
        content = "Hello world"
        assert memory_service._truncate_content(content, 100) == content

    def test_exact_limit_unchanged(self):
        """Content exactly at limit should not be truncated."""
        content = "x" * 100
        assert memory_service._truncate_content(content, 100) == content

    def test_long_content_truncated(self):
        """Content exceeding limit should be truncated with marker."""
        content = "A" * 500 + "B" * 500
        result = memory_service._truncate_content(content, 200)
        assert len(result) < len(content)
        assert "[content truncated]" in result
        assert result.startswith("A")
        assert result.endswith("B")

    def test_head_tail_ratio(self):
        """Truncation preserves 70% head and 20% tail."""
        content = "H" * 700 + "T" * 300
        result = memory_service._truncate_content(content, 100)
        # Marker includes surrounding newlines
        assert "H" * 70 in result
        assert result.endswith("T" * 20)
        assert "[content truncated]" in result


class TestBootstrapPath:
    """Tests for _bootstrap_path()."""

    def test_global_scope(self, tmp_path: Path):
        """Global scope should resolve to global/ directory."""
        with patch.object(memory_service, "_memory_dir", return_value=tmp_path):
            path = memory_service._bootstrap_path("global", "SOUL.md")
            assert path == tmp_path / "global" / "SOUL.md"

    def test_agent_scope(self, tmp_path: Path):
        """Agent scope should resolve to agents/{agent_id}/ directory."""
        agent_id = "12345678-1234-1234-1234-123456789abc"
        with patch.object(memory_service, "_memory_dir", return_value=tmp_path):
            path = memory_service._bootstrap_path(agent_id, "USER.md")
            assert path == tmp_path / "agents" / agent_id / "USER.md"

    def test_invalid_filename_raises(self, tmp_path: Path):
        """Invalid filename should raise ValueError."""
        with patch.object(memory_service, "_memory_dir", return_value=tmp_path):
            with pytest.raises(ValueError, match="Invalid filename"):
                memory_service._bootstrap_path("global", "EVIL.md")

    def test_path_traversal_raises(self, tmp_path: Path):
        """Path traversal in scope should raise ValueError."""
        with patch.object(memory_service, "_memory_dir", return_value=tmp_path):
            with pytest.raises(ValueError, match="path traversal"):
                memory_service._bootstrap_path("../../etc", "SOUL.md")


class TestBootstrapFilesCRUD:
    """Tests for bootstrap file read/write/delete/list."""

    def test_write_and_read(self, tmp_path: Path):
        """Writing and reading a file should round-trip correctly."""
        with patch.object(memory_service, "_memory_dir", return_value=tmp_path):
            ok = memory_service.write_bootstrap_file("global", "SOUL.md", "test content")
            assert ok is True

            content = memory_service.read_bootstrap_file("global", "SOUL.md")
            assert content == "test content"

    def test_read_nonexistent(self, tmp_path: Path):
        """Reading a non-existent file should return None."""
        with patch.object(memory_service, "_memory_dir", return_value=tmp_path):
            content = memory_service.read_bootstrap_file("global", "SOUL.md")
            assert content is None

    def test_delete_existing(self, tmp_path: Path):
        """Deleting an existing file should return True."""
        with patch.object(memory_service, "_memory_dir", return_value=tmp_path):
            memory_service.write_bootstrap_file("global", "SOUL.md", "content")
            ok = memory_service.delete_bootstrap_file("global", "SOUL.md")
            assert ok is True
            assert memory_service.read_bootstrap_file("global", "SOUL.md") is None

    def test_delete_nonexistent(self, tmp_path: Path):
        """Deleting a non-existent file should return False."""
        with patch.object(memory_service, "_memory_dir", return_value=tmp_path):
            ok = memory_service.delete_bootstrap_file("global", "SOUL.md")
            assert ok is False

    def test_list_empty(self, tmp_path: Path):
        """Listing files with nothing on disk should show all files as non-existent."""
        with patch.object(memory_service, "_memory_dir", return_value=tmp_path):
            files = memory_service.list_bootstrap_files()
            assert len(files) == 3
            for f in files:
                assert f["global_exists"] is False
                assert f["agent_exists"] is False
                assert f["effective_scope"] is None

    def test_list_with_global_file(self, tmp_path: Path):
        """Listing files with a global file present should report it."""
        with patch.object(memory_service, "_memory_dir", return_value=tmp_path):
            memory_service.write_bootstrap_file("global", "SOUL.md", "soul content")
            files = memory_service.list_bootstrap_files()
            soul = next(f for f in files if f["filename"] == "SOUL.md")
            assert soul["global_exists"] is True
            assert soul["effective_scope"] == "global"
            assert soul["size"] > 0

    def test_list_with_agent_override(self, tmp_path: Path):
        """Per-agent file should override global in effective_scope."""
        agent_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        with patch.object(memory_service, "_memory_dir", return_value=tmp_path):
            memory_service.write_bootstrap_file("global", "SOUL.md", "global soul")
            memory_service.write_bootstrap_file(agent_id, "SOUL.md", "agent soul")
            files = memory_service.list_bootstrap_files(agent_id)
            soul = next(f for f in files if f["filename"] == "SOUL.md")
            assert soul["global_exists"] is True
            assert soul["agent_exists"] is True
            assert soul["effective_scope"] == agent_id


class TestLoadBootstrapFiles:
    """Tests for load_bootstrap_files() with override logic."""

    def test_global_only(self, tmp_path: Path):
        """When only global files exist, load them."""
        with patch.object(memory_service, "_memory_dir", return_value=tmp_path):
            memory_service.write_bootstrap_file("global", "SOUL.md", "I am a soul")
            result = memory_service.load_bootstrap_files()
            assert "SOUL.md" in result
            assert result["SOUL.md"] == "I am a soul"

    def test_agent_overrides_global(self, tmp_path: Path):
        """Per-agent file should override global file."""
        agent_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        with patch.object(memory_service, "_memory_dir", return_value=tmp_path):
            memory_service.write_bootstrap_file("global", "SOUL.md", "global")
            memory_service.write_bootstrap_file(agent_id, "SOUL.md", "agent")
            result = memory_service.load_bootstrap_files(agent_id)
            assert result["SOUL.md"] == "agent"

    def test_per_file_limit(self, tmp_path: Path):
        """Content exceeding per-file limit should be truncated."""
        with patch.object(memory_service, "_memory_dir", return_value=tmp_path):
            big_content = "x" * (memory_service.PER_FILE_CHAR_LIMIT + 1000)
            memory_service.write_bootstrap_file("global", "SOUL.md", big_content)
            result = memory_service.load_bootstrap_files()
            assert len(result["SOUL.md"]) < len(big_content)
            assert "[content truncated]" in result["SOUL.md"]

    def test_total_limit(self, tmp_path: Path):
        """Total content across files should not exceed TOTAL_CHAR_LIMIT."""
        with patch.object(memory_service, "_memory_dir", return_value=tmp_path):
            # Write three large files that together exceed the total limit
            chunk = "x" * memory_service.PER_FILE_CHAR_LIMIT
            for f in memory_service.BOOTSTRAP_FILES:
                memory_service.write_bootstrap_file("global", f, chunk)
            result = memory_service.load_bootstrap_files()
            total = sum(len(v) for v in result.values())
            assert total <= memory_service.TOTAL_CHAR_LIMIT


# ─── Memory Entry Tests (DB) ─────────────────────────────────


class TestMemoryEntryCRUD:
    """Tests for memory entry create/list/update/delete via service functions.

    Uses memory_db_session (table created without vector column) and
    patches embedding_service to return None.
    """

    @pytest.mark.asyncio
    async def test_create_and_list(self, memory_db_session):
        """Creating entries and listing them should work."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_session():
            yield memory_db_session

        with patch.object(memory_service.embedding_service, "aembed_single", new_callable=AsyncMock, return_value=None):
            with patch.object(memory_service, "AsyncSessionLocal", mock_session):
                entry = await memory_service.create_entry(
                    content="Test fact",
                    agent_id="test-agent-id",
                    category="fact",
                    source="manual",
                )
                assert entry["content"] == "Test fact"
                assert entry["category"] == "fact"
                assert "id" in entry

                result = await memory_service.list_entries(agent_id="test-agent-id")
                assert result["total"] >= 1
                found = any(e["content"] == "Test fact" for e in result["entries"])
                assert found

    @pytest.mark.asyncio
    async def test_delete_entry(self, memory_db_session):
        """Deleting an entry should remove it."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_session():
            yield memory_db_session

        with patch.object(memory_service.embedding_service, "aembed_single", new_callable=AsyncMock, return_value=None):
            with patch.object(memory_service, "AsyncSessionLocal", mock_session):
                entry = await memory_service.create_entry(
                    content="To be deleted",
                    category="fact",
                )
                ok = await memory_service.delete_entry(entry["id"])
                assert ok is True

                # Verify it's gone
                ok2 = await memory_service.delete_entry(entry["id"])
                assert ok2 is False

    @pytest.mark.asyncio
    async def test_update_entry(self, memory_db_session):
        """Updating an entry should change content and category."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_session():
            yield memory_db_session

        with patch.object(memory_service.embedding_service, "aembed_single", new_callable=AsyncMock, return_value=None):
            with patch.object(memory_service, "AsyncSessionLocal", mock_session):
                entry = await memory_service.create_entry(
                    content="Original",
                    category="fact",
                )
                updated = await memory_service.update_entry(
                    entry["id"], content="Updated", category="preference"
                )
                assert updated is not None
                assert updated["content"] == "Updated"
                assert updated["category"] == "preference"

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, memory_db_session):
        """Updating a non-existent entry should return None."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_session():
            yield memory_db_session

        with patch.object(memory_service.embedding_service, "aembed_single", new_callable=AsyncMock, return_value=None):
            with patch.object(memory_service, "AsyncSessionLocal", mock_session):
                result = await memory_service.update_entry(
                    "nonexistent-id", content="Updated"
                )
                assert result is None

    @pytest.mark.asyncio
    async def test_list_with_category_filter(self, memory_db_session):
        """Listing entries with category filter should only return matching entries."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_session():
            yield memory_db_session

        with patch.object(memory_service.embedding_service, "aembed_single", new_callable=AsyncMock, return_value=None):
            with patch.object(memory_service, "AsyncSessionLocal", mock_session):
                await memory_service.create_entry(content="Fact 1", category="fact")
                await memory_service.create_entry(content="Pref 1", category="preference")

                result = await memory_service.list_entries(category="fact")
                for e in result["entries"]:
                    assert e["category"] == "fact"

    @pytest.mark.asyncio
    async def test_list_pagination(self, memory_db_session):
        """Listing entries with limit/offset should paginate correctly."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_session():
            yield memory_db_session

        with patch.object(memory_service.embedding_service, "aembed_single", new_callable=AsyncMock, return_value=None):
            with patch.object(memory_service, "AsyncSessionLocal", mock_session):
                for i in range(5):
                    await memory_service.create_entry(content=f"Entry {i}", category="fact")

                result = await memory_service.list_entries(limit=2, offset=0)
                assert len(result["entries"]) == 2
                assert result["total"] == 5


class TestKeywordSearch:
    """Tests for keyword fallback search (no embeddings)."""

    @pytest.mark.asyncio
    async def test_keyword_search(self, memory_db_session):
        """Search without embeddings should fall back to keyword ILIKE."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_session():
            yield memory_db_session

        with patch.object(memory_service.embedding_service, "aembed_single", new_callable=AsyncMock, return_value=None):
            with patch.object(memory_service, "AsyncSessionLocal", mock_session):
                await memory_service.create_entry(content="Python is great", category="fact")
                await memory_service.create_entry(content="Java is fine", category="fact")

                results = await memory_service.search_memory("Python")
                assert len(results) >= 1
                assert any("Python" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_keyword_search_no_match(self, memory_db_session):
        """Search for non-matching query should return empty."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_session():
            yield memory_db_session

        with patch.object(memory_service.embedding_service, "aembed_single", new_callable=AsyncMock, return_value=None):
            with patch.object(memory_service, "AsyncSessionLocal", mock_session):
                await memory_service.create_entry(content="Hello world", category="fact")
                results = await memory_service.search_memory("nonexistent_xyz_query")
                assert len(results) == 0

    @pytest.mark.asyncio
    async def test_keyword_search_escapes_wildcards(self, memory_db_session):
        """LIKE special characters should be escaped properly."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_session():
            yield memory_db_session

        with patch.object(memory_service.embedding_service, "aembed_single", new_callable=AsyncMock, return_value=None):
            with patch.object(memory_service, "AsyncSessionLocal", mock_session):
                await memory_service.create_entry(content="100% complete", category="fact")
                # Searching for "100%" should match (% should be escaped)
                results = await memory_service.search_memory("100%")
                assert len(results) >= 1


# ─── Flush Memory Tests ──────────────────────────────────────


class TestFlushMemory:
    """Tests for flush_memory() — LLM extraction and dedup logic."""

    @pytest.mark.asyncio
    async def test_empty_messages_returns_zero(self):
        """Flush with no messages should return 0 immediately."""
        count = await memory_service.flush_memory(messages=[])
        assert count == 0

    @pytest.mark.asyncio
    async def test_empty_content_messages_returns_zero(self):
        """Flush with messages that have no text content should return 0."""
        count = await memory_service.flush_memory(messages=[{"role": "user", "content": ""}])
        assert count == 0

    @pytest.mark.asyncio
    async def test_llm_parse_and_save(self):
        """Flush should parse LLM response and save entries."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = '[{"content": "User likes Python", "category": "preference"}]'
        mock_response.content = [mock_block]

        mock_llm = AsyncMock()
        mock_llm.acreate = AsyncMock(return_value=mock_response)

        mock_create = AsyncMock(return_value={"id": "new-id", "content": "User likes Python"})
        mock_search = AsyncMock(return_value=[])

        with patch("app.llm.LLMClient", return_value=mock_llm):
            with patch.object(memory_service, "search_memory", mock_search):
                with patch.object(memory_service, "create_entry", mock_create):
                    count = await memory_service.flush_memory(
                        messages=[{"role": "user", "content": "I really like Python programming"}],
                        model_provider="anthropic",
                        model_name="test-model",
                    )
                    assert count == 1
                    mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_dedup_exact_match_keyword_mode(self, memory_db_session):
        """In keyword-only mode (similarity=None), exact content match should dedup."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_session():
            yield memory_db_session

        # Mock LLM response that tries to save an existing entry
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = '[{"content": "User likes Python", "category": "fact"}]'
        mock_response.content = [mock_block]

        mock_llm = AsyncMock()
        mock_llm.acreate = AsyncMock(return_value=mock_response)

        with patch.object(memory_service.embedding_service, "aembed_single", new_callable=AsyncMock, return_value=None):
            with patch.object(memory_service, "AsyncSessionLocal", mock_session):
                # Pre-create the same entry
                await memory_service.create_entry(content="User likes Python", category="fact")

                with patch("app.llm.LLMClient", return_value=mock_llm):
                    count = await memory_service.flush_memory(
                        messages=[{"role": "user", "content": "I like Python"}],
                    )
                    # Should skip the duplicate
                    assert count == 0

    @pytest.mark.asyncio
    async def test_dedup_vector_mode(self, memory_db_session):
        """In vector mode (similarity present), high similarity should dedup."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_session():
            yield memory_db_session

        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = '[{"content": "New fact", "category": "fact"}]'
        mock_response.content = [mock_block]

        mock_llm = AsyncMock()
        mock_llm.acreate = AsyncMock(return_value=mock_response)

        # Mock search_memory to return a high-similarity match
        mock_search = AsyncMock(return_value=[
            {"content": "Similar fact", "similarity": 0.98, "id": "existing-id"}
        ])

        with patch.object(memory_service.embedding_service, "aembed_single", new_callable=AsyncMock, return_value=None):
            with patch.object(memory_service, "AsyncSessionLocal", mock_session):
                with patch("app.llm.LLMClient", return_value=mock_llm):
                    with patch.object(memory_service, "search_memory", mock_search):
                        count = await memory_service.flush_memory(
                            messages=[{"role": "user", "content": "test"}],
                        )
                        assert count == 0

    @pytest.mark.asyncio
    async def test_flush_max_10_entries(self):
        """Flush should save at most 10 entries even if LLM returns more."""
        import json as _json

        # LLM returns 15 entries
        entries = [{"content": f"Fact {i}", "category": "fact"} for i in range(15)]
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = _json.dumps(entries)
        mock_response.content = [mock_block]

        mock_llm = AsyncMock()
        mock_llm.acreate = AsyncMock(return_value=mock_response)

        mock_create = AsyncMock(return_value={"id": "new-id", "content": "x"})
        mock_search = AsyncMock(return_value=[])

        with patch("app.llm.LLMClient", return_value=mock_llm):
            with patch.object(memory_service, "search_memory", mock_search):
                with patch.object(memory_service, "create_entry", mock_create):
                    count = await memory_service.flush_memory(
                        messages=[{"role": "user", "content": "lots of facts"}],
                    )
                    assert count == 10
                    assert mock_create.call_count == 10

    @pytest.mark.asyncio
    async def test_flush_invalid_json_returns_zero(self):
        """Flush with LLM returning invalid JSON should return 0."""
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "This is not JSON at all"
        mock_response.content = [mock_block]

        mock_llm = AsyncMock()
        mock_llm.acreate = AsyncMock(return_value=mock_response)

        with patch("app.llm.LLMClient", return_value=mock_llm):
            count = await memory_service.flush_memory(
                messages=[{"role": "user", "content": "test"}],
            )
            assert count == 0

    @pytest.mark.asyncio
    async def test_flush_code_fenced_json(self):
        """Flush should handle JSON wrapped in markdown code fences."""
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = '```json\n[{"content": "Fenced fact", "category": "fact"}]\n```'
        mock_response.content = [mock_block]

        mock_llm = AsyncMock()
        mock_llm.acreate = AsyncMock(return_value=mock_response)

        mock_create = AsyncMock(return_value={"id": "new-id", "content": "Fenced fact"})
        mock_search = AsyncMock(return_value=[])

        with patch("app.llm.LLMClient", return_value=mock_llm):
            with patch.object(memory_service, "search_memory", mock_search):
                with patch.object(memory_service, "create_entry", mock_create):
                    count = await memory_service.flush_memory(
                        messages=[{"role": "user", "content": "test"}],
                    )
                    assert count == 1
                    mock_create.assert_called_once()


# ─── Save Memory Sync Tests ─────────────────────────────────


class TestSaveMemorySync:
    """Tests for save_memory_sync() content truncation and behavior."""

    def test_content_truncation(self, memory_db_session):
        """Content exceeding 4096 chars should be truncated."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        sync_url = "postgresql+psycopg2://skills:skills123@localhost:62620/skills_api_test"
        sync_engine = create_engine(sync_url)
        TestSyncSession = sessionmaker(sync_engine, expire_on_commit=False)

        with patch.object(memory_service.embedding_service, "embed_single", return_value=None):
            with patch.object(memory_service, "SyncSessionLocal", TestSyncSession):
                result = memory_service.save_memory_sync(
                    content="x" * 5000,
                    category="fact",
                )
                assert len(result["content"]) == 4096

        sync_engine.dispose()

    def test_short_content_unchanged(self, memory_db_session):
        """Content within limit should not be truncated."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        sync_url = "postgresql+psycopg2://skills:skills123@localhost:62620/skills_api_test"
        sync_engine = create_engine(sync_url)
        TestSyncSession = sessionmaker(sync_engine, expire_on_commit=False)

        with patch.object(memory_service.embedding_service, "embed_single", return_value=None):
            with patch.object(memory_service, "SyncSessionLocal", TestSyncSession):
                result = memory_service.save_memory_sync(
                    content="short content",
                    category="fact",
                )
                assert result["content"] == "short content"

        sync_engine.dispose()
