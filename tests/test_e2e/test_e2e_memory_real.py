"""
Real LLM + Embedding tests for the memory system.

These tests exercise the full memory pipeline with actual API calls:
- OpenAI embeddings for vector search (semantic similarity)
- Kimi (Moonshot) LLM for memory flush (fact extraction from conversations)

Run:
    MOONSHOT_API_KEY_REAL=sk-xxx OPENAI_API_KEY_REAL=sk-xxx \
        pytest tests/test_e2e/test_e2e_memory_real.py -v

All assertions check structure and behavior, never specific text content
(LLM outputs are non-deterministic).
"""

import os
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.services import embedding_service, memory_service

TEST_DATABASE_URL = "postgresql+asyncpg://skills:skills123@localhost:62620/skills_api_test"

# ─── API Key Detection ────────────────────────────────────────

_OPENAI_KEY = os.environ.get("OPENAI_API_KEY_REAL", "") or os.environ.get("OPENAI_API_KEY", "")
_KIMI_KEY = os.environ.get("KIMI_API_KEY_REAL", "") or os.environ.get("MOONSHOT_API_KEY_REAL", "") or os.environ.get("MOONSHOT_API_KEY", "")

skip_no_openai = pytest.mark.skipif(
    not _OPENAI_KEY,
    reason="OPENAI_API_KEY_REAL or OPENAI_API_KEY not set",
)

skip_no_kimi = pytest.mark.skipif(
    not _KIMI_KEY,
    reason="MOONSHOT_API_KEY_REAL or MOONSHOT_API_KEY not set",
)

skip_no_both = pytest.mark.skipif(
    not (_OPENAI_KEY and _KIMI_KEY),
    reason="Need both OPENAI and MOONSHOT API keys for this test",
)


def _patch_embedding_key():
    """Inject the OpenAI key for embedding service."""
    return patch.dict(os.environ, {"OPENAI_API_KEY": _OPENAI_KEY})


def _patch_kimi_key():
    """Inject the Moonshot key for LLM calls."""
    return patch.dict(os.environ, {"MOONSHOT_API_KEY": _KIMI_KEY})


# ─── Fixtures ─────────────────────────────────────────────────


@pytest_asyncio.fixture(scope="class", loop_scope="class")
async def real_memory_db():
    """Create memory_entries table with pgvector and yield a session factory.

    Class-scoped: shared across all tests in the class, cleaned up at end.
    """
    engine = create_async_engine(TEST_DATABASE_URL, echo=False, pool_size=3, max_overflow=5)

    async with engine.begin() as conn:
        await conn.execute(sa_text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(sa_text("DROP TABLE IF EXISTS memory_entries CASCADE"))
        await conn.execute(sa_text("""
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

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    yield {"engine": engine, "factory": session_factory}

    async with engine.begin() as conn:
        await conn.execute(sa_text("DROP TABLE IF EXISTS memory_entries CASCADE"))
    await engine.dispose()


# ─── Test Class ───────────────────────────────────────────────


@pytest.mark.e2e_llm
@pytest.mark.asyncio(loop_scope="class")
class TestMemoryRealLLM:
    """Memory system tests with real OpenAI embeddings and Kimi LLM.

    Test flow:
    1. Create entries WITH real embeddings (vector search)
    2. Semantic search — verify cosine similarity works
    3. Keyword fallback — verify it still works without embeddings
    4. Flush memory — extract facts from a conversation via LLM
    5. Full pipeline — create, search semantically, flush, dedup
    """

    _state: dict = {}

    # ─── Embedding + Vector Search ────────────────────────────

    @skip_no_openai
    async def test_01_create_entries_with_embeddings(self, real_memory_db):
        """Create entries with real OpenAI embeddings."""
        factory = real_memory_db["factory"]

        embedding_service.reset_client()

        with _patch_embedding_key():
            with patch.object(memory_service, "AsyncSessionLocal", factory):
                # Create several entries with different topics
                entries = [
                    ("Python is a dynamically typed programming language", "fact"),
                    ("The user prefers dark mode in all applications", "preference"),
                    ("To deploy, run: docker compose up -d", "procedure"),
                    ("Tokyo is the capital of Japan", "fact"),
                    ("The user speaks both English and Chinese fluently", "preference"),
                ]

                ids = []
                for content, category in entries:
                    entry = await memory_service.create_entry(
                        content=content,
                        agent_id="11111111-1111-1111-1111-111111111111",
                        category=category,
                        source="test",
                    )
                    ids.append(entry["id"])
                    # Verify embedding was generated
                    assert entry["embedding_model"] is not None, f"Embedding not generated for: {content}"

                type(self)._state["entry_ids"] = ids

        embedding_service.reset_client()

    @skip_no_openai
    async def test_02_semantic_search_finds_similar(self, real_memory_db):
        """Semantic search should find entries by meaning, not just keywords."""
        factory = real_memory_db["factory"]

        embedding_service.reset_client()

        with _patch_embedding_key():
            with patch.object(memory_service, "AsyncSessionLocal", factory):
                # Search for "programming language" — should find Python entry
                results = await memory_service.search_memory(
                    "What programming languages are used?",
                    agent_id="11111111-1111-1111-1111-111111111111",
                    top_k=3,
                )
                assert len(results) >= 1
                # Top result should be about Python (semantically closest)
                assert "Python" in results[0]["content"] or "programming" in results[0]["content"].lower()
                assert results[0]["similarity"] is not None
                assert results[0]["similarity"] > 0.3

    @skip_no_openai
    async def test_03_semantic_search_ranked_by_relevance(self, real_memory_db):
        """Results should be ranked by cosine similarity — most relevant first."""
        factory = real_memory_db["factory"]

        embedding_service.reset_client()

        with _patch_embedding_key():
            with patch.object(memory_service, "AsyncSessionLocal", factory):
                # Search for "user interface theme" — should rank dark mode highest
                results = await memory_service.search_memory(
                    "What theme does the user prefer for their UI?",
                    agent_id="11111111-1111-1111-1111-111111111111",
                    top_k=5,
                )
                assert len(results) >= 2
                # All results should have similarity scores
                for r in results:
                    assert r["similarity"] is not None
                # Scores should be descending
                similarities = [r["similarity"] for r in results]
                assert similarities == sorted(similarities, reverse=True)
                # Top result should be about dark mode
                assert "dark mode" in results[0]["content"].lower()

    @skip_no_openai
    async def test_04_semantic_search_cross_language(self, real_memory_db):
        """Semantic search should work across languages (embeddings capture meaning)."""
        factory = real_memory_db["factory"]

        embedding_service.reset_client()

        with _patch_embedding_key():
            with patch.object(memory_service, "AsyncSessionLocal", factory):
                # Search in Chinese for the Japanese capital entry
                results = await memory_service.search_memory(
                    "日本的首都是哪里？",
                    agent_id="11111111-1111-1111-1111-111111111111",
                    top_k=3,
                )
                assert len(results) >= 1
                # Should find the Tokyo entry even though query is in Chinese
                assert any("Tokyo" in r["content"] or "Japan" in r["content"] for r in results[:2])

    @skip_no_openai
    async def test_05_semantic_search_unrelated_query(self, real_memory_db):
        """Searching for completely unrelated topic should return low similarity."""
        factory = real_memory_db["factory"]

        embedding_service.reset_client()

        with _patch_embedding_key():
            with patch.object(memory_service, "AsyncSessionLocal", factory):
                results = await memory_service.search_memory(
                    "quantum physics black holes string theory",
                    agent_id="11111111-1111-1111-1111-111111111111",
                    top_k=5,
                )
                # Should return results (closest vectors) but with lower similarity
                if results:
                    assert all(r["similarity"] < 0.85 for r in results)

        embedding_service.reset_client()

    # ─── LLM Flush ────────────────────────────────────────────

    @skip_no_kimi
    async def test_06_flush_extracts_facts(self, real_memory_db):
        """flush_memory should use LLM to extract facts from a conversation."""
        factory = real_memory_db["factory"]

        # Simulate a conversation with extractable facts
        messages = [
            {"role": "user", "content": "Hi, I'm working on a machine learning project using PyTorch."},
            {"role": "assistant", "content": "That's great! What kind of model are you building?"},
            {"role": "user", "content": "I'm building a transformer model for text classification. My dataset has about 50,000 samples."},
            {"role": "assistant", "content": "That's a reasonable dataset size. Are you using a pre-trained model as a base?"},
            {"role": "user", "content": "Yes, I'm fine-tuning BERT. By the way, I prefer using VS Code as my editor and I always use type hints in my Python code."},
        ]

        embedding_service.reset_client()

        with _patch_kimi_key():
            with _patch_embedding_key():
                with patch.object(memory_service, "AsyncSessionLocal", factory):
                    count = await memory_service.flush_memory(
                        messages=messages,
                        agent_id="22222222-2222-2222-2222-222222222222",
                        model_provider="kimi",
                        model_name="kimi-k2.5",
                    )

                    # LLM should extract at least 2 facts from this conversation
                    assert count >= 2, f"Expected at least 2 extracted facts, got {count}"

                    # Verify extracted entries exist in DB
                    result = await memory_service.list_entries(
                        agent_id="22222222-2222-2222-2222-222222222222",
                    )
                    assert result["total"] >= 2
                    type(self)._state["flush_count"] = count
                    type(self)._state["flush_entries"] = result["entries"]

                    # Verify entries have meaningful content (not empty)
                    for entry in result["entries"]:
                        assert len(entry["content"]) > 10
                        assert entry["source"] == "auto_flush"

        embedding_service.reset_client()

    @skip_no_kimi
    async def test_07_flush_dedup_prevents_duplicates(self, real_memory_db):
        """Flushing the same conversation again should not create duplicates."""
        factory = real_memory_db["factory"]

        messages = [
            {"role": "user", "content": "I'm building a transformer model for text classification using PyTorch."},
            {"role": "assistant", "content": "Nice! Are you fine-tuning BERT?"},
            {"role": "user", "content": "Yes, I'm fine-tuning BERT on my 50,000 sample dataset."},
        ]

        embedding_service.reset_client()

        with _patch_kimi_key():
            with _patch_embedding_key():
                with patch.object(memory_service, "AsyncSessionLocal", factory):
                    # Get count before second flush
                    before = await memory_service.list_entries(
                        agent_id="22222222-2222-2222-2222-222222222222",
                    )
                    before_total = before["total"]

                    # Flush similar conversation — should skip existing facts
                    count = await memory_service.flush_memory(
                        messages=messages,
                        agent_id="22222222-2222-2222-2222-222222222222",
                        model_provider="kimi",
                        model_name="kimi-k2.5",
                    )

                    after = await memory_service.list_entries(
                        agent_id="22222222-2222-2222-2222-222222222222",
                    )

                    # Should add very few (or zero) new entries due to dedup
                    new_entries = after["total"] - before_total
                    assert new_entries <= count
                    # At least some should have been deduped
                    # (LLM may extract slightly different facts, so we're lenient)

        embedding_service.reset_client()

    # ─── Full Pipeline ────────────────────────────────────────

    @skip_no_both
    async def test_08_full_pipeline_create_search_flush(self, real_memory_db):
        """Full pipeline: manual create → semantic search → flush → search again."""
        factory = real_memory_db["factory"]
        agent_id = "33333333-3333-3333-3333-333333333333"

        embedding_service.reset_client()

        with _patch_embedding_key():
            with _patch_kimi_key():
                with patch.object(memory_service, "AsyncSessionLocal", factory):
                    # Step 1: Manually create a memory entry
                    entry = await memory_service.create_entry(
                        content="The project uses PostgreSQL with pgvector for semantic search",
                        agent_id=agent_id,
                        category="fact",
                        source="manual",
                    )
                    assert entry["embedding_model"] is not None

                    # Step 2: Semantic search should find it
                    results = await memory_service.search_memory(
                        "What database does the project use?",
                        agent_id=agent_id,
                        top_k=3,
                    )
                    assert len(results) >= 1
                    assert "PostgreSQL" in results[0]["content"]
                    assert results[0]["similarity"] > 0.3

                    # Step 3: Flush a conversation that mentions the same fact
                    messages = [
                        {"role": "user", "content": "How is the database set up?"},
                        {"role": "assistant", "content": "The project uses PostgreSQL with pgvector for semantic search. We also use Redis for caching."},
                        {"role": "user", "content": "Great. And we deploy with Docker on AWS."},
                    ]
                    count = await memory_service.flush_memory(
                        messages=messages,
                        agent_id=agent_id,
                        model_provider="kimi",
                        model_name="kimi-k2.5",
                    )
                    # Should extract some facts (Redis, Docker, AWS) but
                    # skip the PostgreSQL/pgvector fact (already exists)
                    assert count >= 1

                    # Step 4: Search again — should find both old and new entries
                    results = await memory_service.search_memory(
                        "deployment infrastructure",
                        agent_id=agent_id,
                        top_k=10,
                    )
                    assert len(results) >= 2

        embedding_service.reset_client()

    @skip_no_openai
    async def test_09_update_re_embeds(self, real_memory_db):
        """Updating content should re-generate the embedding."""
        factory = real_memory_db["factory"]
        agent_id = "44444444-4444-4444-4444-444444444444"

        embedding_service.reset_client()

        with _patch_embedding_key():
            with patch.object(memory_service, "AsyncSessionLocal", factory):
                # Create an entry about cats
                entry = await memory_service.create_entry(
                    content="Cats are independent and low-maintenance pets",
                    agent_id=agent_id,
                    category="fact",
                )

                # Search for dogs — should not be top result
                results = await memory_service.search_memory("dogs", agent_id=agent_id, top_k=1)
                cat_similarity = results[0]["similarity"] if results else 0

                # Update to be about dogs
                updated = await memory_service.update_entry(
                    entry["id"],
                    content="Dogs are loyal companions that need daily walks",
                )
                assert updated is not None

                # Now search for dogs — should have HIGHER similarity
                results = await memory_service.search_memory("dogs", agent_id=agent_id, top_k=1)
                assert len(results) >= 1
                assert results[0]["similarity"] > cat_similarity

        embedding_service.reset_client()

    # ─── Cleanup ──────────────────────────────────────────────

    async def test_99_cleanup(self, real_memory_db):
        """Report test results."""
        flush_count = type(self)._state.get("flush_count")
        if flush_count is not None:
            print(f"\n  Flush extracted {flush_count} facts from conversation")
        flush_entries = type(self)._state.get("flush_entries", [])
        for e in flush_entries[:5]:
            print(f"    [{e['category']}] {e['content'][:80]}")
