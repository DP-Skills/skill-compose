"""Memory service — bootstrap files + vector-searchable memory entries.

Provides:
- Bootstrap file CRUD (SOUL.md, USER.md, MEMORY.md) with global/per-agent scopes
- Memory entry CRUD with automatic embedding generation
- Semantic search via pgvector cosine distance
- Memory flush (LLM extraction of key facts from conversation)
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from app.config import get_settings
from app.db.database import AsyncSessionLocal, SyncSessionLocal
from app.services import embedding_service

logger = logging.getLogger(__name__)

# Bootstrap file constants
BOOTSTRAP_FILES = ["SOUL.md", "USER.md", "MEMORY.md"]
PER_FILE_CHAR_LIMIT = 20_000
TOTAL_CHAR_LIMIT = 60_000
HEAD_RATIO = 0.70
TAIL_RATIO = 0.20

# Flush prompt
FLUSH_PROMPT = """You are a memory extraction assistant. Analyze the following conversation messages and extract key facts, preferences, procedures, and context that would be useful for future conversations.

Return a JSON array of objects, each with:
- "content": the fact/preference/procedure (1-3 sentences)
- "category": one of "fact", "preference", "procedure", "context"

Only extract genuinely useful, non-obvious information. Skip trivial details.
Return ONLY the JSON array, no other text.

Messages:
{messages}"""


# ─── Bootstrap File Operations ─────────────────────────────────

def _memory_dir() -> Path:
    """Get the memory directory path."""
    settings = get_settings()
    return Path(settings.memory_dir)


def _bootstrap_path(scope: str, filename: str) -> Path:
    """Get the filesystem path for a bootstrap file.

    scope: "global" or an agent_id
    Raises ValueError if the resolved path escapes the memory directory.
    """
    if filename not in BOOTSTRAP_FILES:
        raise ValueError(f"Invalid filename: {filename}")
    base = _memory_dir()
    if scope == "global":
        result = base / "global" / filename
    else:
        result = base / "agents" / scope / filename
    # Path traversal protection: resolved path must stay within memory dir
    if not result.resolve().is_relative_to(base.resolve()):
        raise ValueError("Invalid scope: path traversal detected")
    return result


def list_bootstrap_files(agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available bootstrap files with metadata.

    Returns files for both global and per-agent scopes.
    Per-agent files override global files of the same name.
    """
    result = []
    base = _memory_dir()

    for filename in BOOTSTRAP_FILES:
        # Check global
        global_path = base / "global" / filename
        global_exists = global_path.exists()

        # Check per-agent
        agent_exists = False
        if agent_id:
            agent_path = base / "agents" / agent_id / filename
            agent_exists = agent_path.exists()

        # Determine effective content
        effective_scope = None
        size = 0
        if agent_id and agent_exists:
            effective_scope = agent_id
            size = agent_path.stat().st_size
        elif global_exists:
            effective_scope = "global"
            size = global_path.stat().st_size

        result.append({
            "filename": filename,
            "global_exists": global_exists,
            "agent_exists": agent_exists,
            "effective_scope": effective_scope,
            "size": size,
        })

    return result


def read_bootstrap_file(scope: str, filename: str) -> Optional[str]:
    """Read a bootstrap file's content."""
    if filename not in BOOTSTRAP_FILES:
        return None
    path = _bootstrap_path(scope, filename)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read bootstrap file {path}: {e}")
        return None


def write_bootstrap_file(scope: str, filename: str, content: str) -> bool:
    """Write content to a bootstrap file."""
    if filename not in BOOTSTRAP_FILES:
        return False
    path = _bootstrap_path(scope, filename)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return True
    except Exception as e:
        logger.warning(f"Failed to write bootstrap file {path}: {e}")
        return False


def delete_bootstrap_file(scope: str, filename: str) -> bool:
    """Delete a bootstrap file."""
    if filename not in BOOTSTRAP_FILES:
        return False
    path = _bootstrap_path(scope, filename)
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except Exception as e:
        logger.warning(f"Failed to delete bootstrap file {path}: {e}")
        return False


def _truncate_content(content: str, limit: int) -> str:
    """Truncate content keeping head + tail with a marker."""
    if len(content) <= limit:
        return content
    head_len = int(limit * HEAD_RATIO)
    tail_len = int(limit * TAIL_RATIO)
    marker = "\n\n... [content truncated] ...\n\n"
    return content[:head_len] + marker + content[-tail_len:]


def load_bootstrap_files(agent_id: Optional[str] = None) -> Dict[str, str]:
    """Load all bootstrap files with per-agent override logic.

    Returns dict mapping filename to content.
    Applies per-file and total character limits.
    """
    result = {}
    total_chars = 0
    base = _memory_dir()

    for filename in BOOTSTRAP_FILES:
        content = None

        # Per-agent override takes precedence
        if agent_id:
            agent_path = base / "agents" / agent_id / filename
            if agent_path.exists():
                try:
                    content = agent_path.read_text(encoding="utf-8")
                except Exception:
                    pass

        # Fall back to global
        if content is None:
            global_path = base / "global" / filename
            if global_path.exists():
                try:
                    content = global_path.read_text(encoding="utf-8")
                except Exception:
                    pass

        if content:
            # Per-file limit
            content = _truncate_content(content, PER_FILE_CHAR_LIMIT)
            # Total limit check
            if total_chars + len(content) > TOTAL_CHAR_LIMIT:
                remaining = TOTAL_CHAR_LIMIT - total_chars
                if remaining > 0:
                    content = _truncate_content(content, remaining)
                else:
                    break
            result[filename] = content
            total_chars += len(content)

    return result


# ─── Memory Entry Operations (DB + Vector) ──────────────────────

async def create_entry(
    content: str,
    agent_id: Optional[str] = None,
    category: Optional[str] = None,
    source: Optional[str] = "manual",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a memory entry with auto-generated embedding."""
    entry_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # Generate embedding
    embedding = await embedding_service.aembed_single(content)
    embedding_model = embedding_service.get_model() if embedding else None

    async with AsyncSessionLocal() as session:
        async with session.begin():
            await session.execute(
                text("""
                    INSERT INTO memory_entries (id, agent_id, content, category, source, embedding, embedding_model, session_id, created_at, updated_at)
                    VALUES (:id, :agent_id, :content, :category, :source, CAST(:embedding AS vector), :embedding_model, :session_id, :created_at, :updated_at)
                """),
                {
                    "id": entry_id,
                    "agent_id": agent_id,
                    "content": content,
                    "category": category,
                    "source": source,
                    "embedding": str(embedding) if embedding else None,
                    "embedding_model": embedding_model,
                    "session_id": session_id,
                    "created_at": now,
                    "updated_at": now,
                }
            )

    return {
        "id": entry_id,
        "agent_id": agent_id,
        "content": content,
        "category": category,
        "source": source,
        "embedding_model": embedding_model,
        "session_id": session_id,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }


async def update_entry(
    entry_id: str,
    content: Optional[str] = None,
    category: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Update a memory entry and re-embed if content changed."""
    now = datetime.now(timezone.utc)

    async with AsyncSessionLocal() as session:
        async with session.begin():
            # Fetch current entry
            result = await session.execute(
                text("SELECT id, agent_id, content, category, source, embedding_model, session_id, created_at FROM memory_entries WHERE id = :id"),
                {"id": entry_id}
            )
            row = result.fetchone()
            if not row:
                return None

            new_content = content if content is not None else row.content
            new_category = category if category is not None else row.category

            # Re-embed if content changed
            embedding = None
            embedding_model = row.embedding_model
            if content is not None and content != row.content:
                embedding = await embedding_service.aembed_single(new_content)
                if embedding:
                    embedding_model = embedding_service.get_model()

            if embedding is not None:
                await session.execute(
                    text("""
                        UPDATE memory_entries
                        SET content = :content, category = :category, embedding = CAST(:embedding AS vector),
                            embedding_model = :embedding_model, updated_at = :updated_at
                        WHERE id = :id
                    """),
                    {
                        "id": entry_id,
                        "content": new_content,
                        "category": new_category,
                        "embedding": str(embedding),
                        "embedding_model": embedding_model,
                        "updated_at": now,
                    }
                )
            else:
                await session.execute(
                    text("""
                        UPDATE memory_entries
                        SET content = :content, category = :category, updated_at = :updated_at
                        WHERE id = :id
                    """),
                    {
                        "id": entry_id,
                        "content": new_content,
                        "category": new_category,
                        "updated_at": now,
                    }
                )

    return {
        "id": entry_id,
        "agent_id": row.agent_id,
        "content": new_content,
        "category": new_category,
        "source": row.source,
        "embedding_model": embedding_model,
        "session_id": row.session_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": now.isoformat(),
    }


async def delete_entry(entry_id: str) -> bool:
    """Delete a memory entry."""
    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(
                text("DELETE FROM memory_entries WHERE id = :id"),
                {"id": entry_id}
            )
            return result.rowcount > 0


async def list_entries(
    agent_id: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """List memory entries with optional filters. Returns entries and total count."""
    conditions = []
    params: Dict[str, Any] = {"limit": limit, "offset": offset}

    if agent_id is not None:
        # Include both agent-specific AND global entries
        conditions.append("(agent_id = :agent_id OR agent_id IS NULL)")
        params["agent_id"] = agent_id
    if category:
        conditions.append("category = :category")
        params["category"] = category

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    async with AsyncSessionLocal() as session:
        # Get total count
        count_result = await session.execute(
            text(f"SELECT COUNT(*) FROM memory_entries {where}"),
            {k: v for k, v in params.items() if k not in ("limit", "offset")}
        )
        total = count_result.scalar()

        result = await session.execute(
            text(f"""
                SELECT id, agent_id, content, category, source, embedding_model, session_id, created_at, updated_at
                FROM memory_entries
                {where}
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """),
            params
        )
        rows = result.fetchall()

    entries = [
        {
            "id": r.id,
            "agent_id": r.agent_id,
            "content": r.content,
            "category": r.category,
            "source": r.source,
            "embedding_model": r.embedding_model,
            "session_id": r.session_id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in rows
    ]
    return {"entries": entries, "total": total}


async def search_memory(
    query: str,
    agent_id: Optional[str] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Semantic search over memory entries using pgvector cosine distance.

    Searches both global (agent_id IS NULL) and agent-specific entries.
    Falls back to keyword search if embedding is not available.
    """
    query_embedding = await embedding_service.aembed_single(query)

    if query_embedding is not None:
        return await _vector_search(query_embedding, agent_id, top_k)
    else:
        return await _keyword_search(query, agent_id, top_k)


def _build_vector_query(query_embedding: list[float], agent_id: Optional[str], top_k: int):
    """Build SQL + params for vector similarity search."""
    agent_filter = ""
    params: Dict[str, Any] = {"embedding": str(query_embedding), "top_k": top_k}
    if agent_id:
        agent_filter = "AND (agent_id = :agent_id OR agent_id IS NULL)"
        params["agent_id"] = agent_id
    sql = f"""
        SELECT id, agent_id, content, category, source, created_at,
               1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM memory_entries
        WHERE embedding IS NOT NULL {agent_filter}
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """
    return sql, params


def _build_keyword_query(query: str, agent_id: Optional[str], top_k: int):
    """Build SQL + params for keyword ILIKE search."""
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    agent_filter = ""
    params: Dict[str, Any] = {"query": f"%{escaped}%", "top_k": top_k}
    if agent_id:
        agent_filter = "AND (agent_id = :agent_id OR agent_id IS NULL)"
        params["agent_id"] = agent_id
    sql = f"""
        SELECT id, agent_id, content, category, source, created_at
        FROM memory_entries
        WHERE content ILIKE :query {agent_filter}
        ORDER BY created_at DESC
        LIMIT :top_k
    """
    return sql, params


def _rows_to_search_results(rows, has_similarity: bool = False) -> List[Dict[str, Any]]:
    """Convert search result rows to dicts."""
    return [
        {
            "id": r.id,
            "agent_id": r.agent_id,
            "content": r.content,
            "category": r.category,
            "source": r.source,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "similarity": round(r.similarity, 4) if has_similarity and r.similarity else None,
        }
        for r in rows
    ]


async def _vector_search(query_embedding: list[float], agent_id: Optional[str], top_k: int) -> List[Dict[str, Any]]:
    """Vector similarity search using pgvector cosine distance."""
    sql, params = _build_vector_query(query_embedding, agent_id, top_k)
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(text(sql), params)).fetchall()
    return _rows_to_search_results(rows, has_similarity=True)


async def _keyword_search(query: str, agent_id: Optional[str], top_k: int) -> List[Dict[str, Any]]:
    """Fallback keyword search using ILIKE."""
    sql, params = _build_keyword_query(query, agent_id, top_k)
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(text(sql), params)).fetchall()
    return _rows_to_search_results(rows)


# ─── Sync Wrappers (for agent tools running in threads) ─────────

def search_memory_sync(
    query: str,
    agent_id: Optional[str] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Sync version of search_memory for agent tools."""
    query_embedding = embedding_service.embed_single(query)
    if query_embedding is not None:
        sql, params = _build_vector_query(query_embedding, agent_id, top_k)
        with SyncSessionLocal() as session:
            rows = session.execute(text(sql), params).fetchall()
        return _rows_to_search_results(rows, has_similarity=True)
    else:
        sql, params = _build_keyword_query(query, agent_id, top_k)
        with SyncSessionLocal() as session:
            rows = session.execute(text(sql), params).fetchall()
        return _rows_to_search_results(rows)


def save_memory_sync(
    content: str,
    agent_id: Optional[str] = None,
    category: Optional[str] = None,
    source: Optional[str] = "agent_tool",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Sync version of create_entry for agent tools."""
    # Enforce content length limit (API layer validates via Pydantic,
    # but tool callers bypass it)
    if len(content) > 4096:
        content = content[:4096]

    entry_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    embedding = embedding_service.embed_single(content)
    embedding_model = embedding_service.get_model() if embedding else None

    with SyncSessionLocal() as session:
        with session.begin():
            session.execute(
                text("""
                    INSERT INTO memory_entries (id, agent_id, content, category, source, embedding, embedding_model, session_id, created_at, updated_at)
                    VALUES (:id, :agent_id, :content, :category, :source, CAST(:embedding AS vector), :embedding_model, :session_id, :created_at, :updated_at)
                """),
                {
                    "id": entry_id,
                    "agent_id": agent_id,
                    "content": content,
                    "category": category,
                    "source": source,
                    "embedding": str(embedding) if embedding else None,
                    "embedding_model": embedding_model,
                    "session_id": session_id,
                    "created_at": now,
                    "updated_at": now,
                }
            )

    return {
        "id": entry_id,
        "agent_id": agent_id,
        "content": content,
        "category": category,
        "source": source,
        "created_at": now.isoformat(),
    }


# ─── Memory Flush (pre-compression extraction) ──────────────────

async def flush_memory(
    messages: List[Dict],
    agent_id: Optional[str] = None,
    model_provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> int:
    """Extract key facts from recent messages and save as memory entries.

    Called before context compression to preserve important information.
    Returns the number of entries saved.
    """
    # Serialize messages for the prompt (caller may already pass a slice)
    serialized = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        text_parts.append(f"[tool: {block.get('name', '')}]")
                    elif block.get("type") == "tool_result":
                        result_content = block.get("content", "")
                        if isinstance(result_content, str):
                            text_parts.append(f"[result: {result_content[:200]}]")
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)
        if content:
            serialized.append(f"[{role}]: {content[:500]}")

    if not serialized:
        return 0

    messages_text = "\n".join(serialized)
    prompt = FLUSH_PROMPT.replace("{messages}", messages_text)

    try:
        from app.llm import LLMClient
        client = LLMClient(
            provider=model_provider or get_settings().default_model_provider,
            model=model_name or get_settings().default_model_name,
        )
        response = await client.acreate(
            messages=[{"role": "user", "content": prompt}],
            system="You extract structured memory entries from conversations.",
            max_tokens=2000,
        )

        # Parse the response
        response_text = ""
        for block in response.content:
            if hasattr(block, 'text'):
                response_text += block.text

        # Try to parse JSON array
        response_text = response_text.strip()
        if response_text.startswith("```"):
            # Strip markdown code fence
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        entries = json.loads(response_text)
        if not isinstance(entries, list):
            return 0

        saved = 0
        for entry in entries[:10]:  # Max 10 entries per flush
            if not isinstance(entry, dict) or "content" not in entry:
                continue
            content = entry["content"]
            # Dedup: skip if a very similar entry already exists
            try:
                existing = await search_memory(content, agent_id=agent_id, top_k=1)
                if existing:
                    sim = existing[0].get("similarity")
                    if sim is not None and sim > 0.95:
                        continue
                    # Keyword-only mode: similarity is None, fall back to exact match
                    if sim is None and existing[0].get("content", "").strip() == content.strip():
                        continue
            except Exception:
                pass  # search failure should not block saving
            await create_entry(
                content=content,
                agent_id=agent_id,
                category=entry.get("category", "context"),
                source="auto_flush",
            )
            saved += 1

        logger.info(f"Memory flush: saved {saved} entries for agent {agent_id}")
        return saved

    except Exception as e:
        logger.warning(f"Memory flush failed (non-fatal): {e}")
        return 0
