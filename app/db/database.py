"""
Database connection management for Skill Registry.

Uses SQLAlchemy 2.0 async API with asyncpg for PostgreSQL.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

from app.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def _get_database_url() -> str:
    """Get the effective database URL from settings."""
    return settings.effective_database_url


def _get_sync_database_url() -> str:
    """Convert async database URL to sync (psycopg2) URL."""
    url = _get_database_url()
    # postgresql+asyncpg://... -> postgresql+psycopg2://...
    return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")


# Get database URL
_db_url = _get_database_url()

# Create async engine with PostgreSQL connection pool settings
engine = create_async_engine(
    _db_url,
    echo=settings.database_echo,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    pool_pre_ping=True,  # Verify connections before use (prevents stale connection errors after restart)
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Sync engine and session factory (for agent tools running in threads)
_sync_db_url = _get_sync_database_url()
sync_engine = create_engine(
    _sync_db_url,
    echo=settings.database_echo,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
    pool_pre_ping=True,
)
SyncSessionLocal = sessionmaker(
    sync_engine,
    class_=Session,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database sessions.

    Usage in FastAPI:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """
    Initialize database tables.

    This creates all tables defined in the ORM models.
    Should be called on application startup.
    """
    # Import models to ensure they are registered with Base
    from app.db import models  # noqa: F401
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Run migrations for existing databases
    await _run_migrations()


async def _run_migrations():
    """
    Run simple migrations for PostgreSQL databases.
    Adds new columns that may be missing from older database versions.

    Uses PostgreSQL's DO block with exception handling to safely add columns
    (handles 'column already exists' gracefully).
    """
    from sqlalchemy import text

    async with engine.begin() as conn:
        # Enable pgvector extension for memory vector search
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Safely add columns using DO blocks (ignores duplicate_column errors)
        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE skills ADD COLUMN skill_type VARCHAR(32) DEFAULT 'user' NOT NULL;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """))

        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE skills ADD COLUMN tools JSONB DEFAULT NULL;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """))

        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE skills ADD COLUMN tags JSONB DEFAULT NULL;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """))

        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE skills ADD COLUMN icon_url VARCHAR(512) DEFAULT NULL;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """))

        # Migrate existing 'system' skill_type to 'meta'
        await conn.execute(
            text("UPDATE skills SET skill_type = 'meta' WHERE skill_type = 'system'")
        )

        # Update meta skills based on config
        meta_skills = settings.meta_skills
        if meta_skills:
            placeholders = ", ".join(f"'{s}'" for s in meta_skills)
            await conn.execute(
                text(f"UPDATE skills SET skill_type = 'meta' WHERE name IN ({placeholders})")
            )

    # Migrate agent_presets table
    async with engine.begin() as conn:
        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE agent_presets ADD COLUMN is_published BOOLEAN DEFAULT FALSE NOT NULL;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """))

        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE agent_presets ADD COLUMN api_response_mode VARCHAR(32) DEFAULT NULL;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """))

        # Backward compatibility: set existing published agents to streaming
        await conn.execute(
            text("UPDATE agent_presets SET api_response_mode = 'streaming' WHERE is_published = TRUE AND api_response_mode IS NULL")
        )

    # Create published_sessions table if not exists
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS published_sessions (
                id VARCHAR(36) PRIMARY KEY,
                agent_id VARCHAR(36) NOT NULL,
                messages JSONB DEFAULT '[]',
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_published_sessions_agent_id ON published_sessions (agent_id)"
        ))

    # Add agent_context column to published_sessions table
    async with engine.begin() as conn:
        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE published_sessions ADD COLUMN agent_context JSONB DEFAULT NULL;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """))

    # Add category and is_pinned columns to skills table
    async with engine.begin() as conn:
        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE skills ADD COLUMN category VARCHAR(64) DEFAULT NULL;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """))

        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE skills ADD COLUMN is_pinned BOOLEAN DEFAULT FALSE NOT NULL;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """))

    # Add session_id column to agent_traces table
    async with engine.begin() as conn:
        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE agent_traces ADD COLUMN session_id VARCHAR(36) DEFAULT NULL;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_agent_traces_session_id ON agent_traces (session_id)"
        ))

    # Create users table if not exists
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(36) PRIMARY KEY,
                username VARCHAR(64) UNIQUE NOT NULL,
                password_hash VARCHAR(256) NOT NULL,
                display_name VARCHAR(128),
                role VARCHAR(32) NOT NULL DEFAULT 'user',
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_users_username ON users (username)"
        ))

    # Add must_change_password column to users table
    async with engine.begin() as conn:
        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE users ADD COLUMN must_change_password BOOLEAN NOT NULL DEFAULT FALSE;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """))

    # Add password_changed_at and seed_hash columns
    async with engine.begin() as conn:
        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE users ADD COLUMN password_changed_at TIMESTAMP DEFAULT NULL;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """))

        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE skills ADD COLUMN seed_hash VARCHAR(64) DEFAULT NULL;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """))

        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE agent_presets ADD COLUMN seed_hash VARCHAR(64) DEFAULT NULL;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """))

    # Create scheduled_tasks and task_run_logs tables
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id VARCHAR(36) PRIMARY KEY,
                name VARCHAR(128) UNIQUE NOT NULL,
                agent_id VARCHAR(36) NOT NULL REFERENCES agent_presets(id) ON DELETE CASCADE,
                prompt TEXT NOT NULL,
                schedule_type VARCHAR(32) NOT NULL,
                schedule_value VARCHAR(128) NOT NULL,
                context_mode VARCHAR(32) NOT NULL DEFAULT 'isolated',
                session_id VARCHAR(36),
                channel_binding_id VARCHAR(36),
                status VARCHAR(32) NOT NULL DEFAULT 'active',
                next_run TIMESTAMP,
                last_run TIMESTAMP,
                max_runs INTEGER,
                run_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_scheduled_tasks_status ON scheduled_tasks (status)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_scheduled_tasks_next_run ON scheduled_tasks (next_run)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_scheduled_tasks_name ON scheduled_tasks (name)"))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS task_run_logs (
                id VARCHAR(36) PRIMARY KEY,
                task_id VARCHAR(36) NOT NULL REFERENCES scheduled_tasks(id) ON DELETE CASCADE,
                started_at TIMESTAMP NOT NULL DEFAULT NOW(),
                completed_at TIMESTAMP,
                duration_ms INTEGER,
                status VARCHAR(32) NOT NULL DEFAULT 'running',
                result_summary TEXT,
                error TEXT,
                trace_id VARCHAR(36),
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_task_run_logs_task_id ON task_run_logs (task_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_task_run_logs_started_at ON task_run_logs (started_at)"))

    # Add delivery_to column to scheduled_tasks table
    async with engine.begin() as conn:
        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE scheduled_tasks ADD COLUMN delivery_to VARCHAR(256) DEFAULT NULL;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """))

    # Create channel_bindings and channel_messages tables
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS channel_bindings (
                id VARCHAR(36) PRIMARY KEY,
                channel_type VARCHAR(32) NOT NULL,
                external_id VARCHAR(256) NOT NULL,
                name VARCHAR(128) NOT NULL,
                agent_id VARCHAR(36) NOT NULL REFERENCES agent_presets(id) ON DELETE CASCADE,
                trigger_pattern VARCHAR(512),
                enabled BOOLEAN NOT NULL DEFAULT TRUE,
                config JSONB,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                UNIQUE(channel_type, external_id)
            )
        """))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_channel_bindings_channel_type ON channel_bindings (channel_type)"))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS channel_messages (
                id VARCHAR(36) PRIMARY KEY,
                channel_binding_id VARCHAR(36) NOT NULL REFERENCES channel_bindings(id) ON DELETE CASCADE,
                direction VARCHAR(16) NOT NULL,
                external_message_id VARCHAR(256),
                sender_id VARCHAR(256),
                sender_name VARCHAR(256),
                content TEXT NOT NULL,
                message_type VARCHAR(32) NOT NULL DEFAULT 'text',
                metadata JSONB,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_channel_messages_binding_id ON channel_messages (channel_binding_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_channel_messages_created_at ON channel_messages (created_at)"))

    # Migrate channel_bindings unique constraint to partial indexes for global binding support
    async with engine.begin() as conn:
        # Drop old unique constraint (if exists)
        await conn.execute(text("""
            DO $$ BEGIN
                ALTER TABLE channel_bindings DROP CONSTRAINT IF EXISTS uq_channel_binding;
            END $$
        """))

        # Partial unique index for specific (non-global) bindings
        await conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_channel_binding_specific
            ON channel_bindings (channel_type, external_id)
            WHERE external_id != '*'
        """))

        # Partial unique index for global bindings: one per (channel_type, app_id)
        await conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_channel_binding_global
            ON channel_bindings (channel_type, (config->>'app_id'))
            WHERE external_id = '*'
        """))

    # Create memory_entries table with pgvector embedding column
    from app.config import get_settings as _get_settings
    _dim = _get_settings().embedding_dimensions
    if not (isinstance(_dim, int) and 0 < _dim <= 4096):
        raise ValueError(f"Invalid embedding dimension: {_dim}")
    async with engine.begin() as conn:
        await conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS memory_entries (
                id VARCHAR(36) PRIMARY KEY,
                agent_id VARCHAR(36),
                content TEXT NOT NULL,
                category VARCHAR(64),
                source VARCHAR(256),
                embedding vector({_dim}),
                embedding_model VARCHAR(128),
                session_id VARCHAR(36),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_memory_entries_agent_id ON memory_entries (agent_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_memory_entries_category ON memory_entries (category)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_memory_entries_created_at ON memory_entries (created_at)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_memory_entries_embedding ON memory_entries USING hnsw (embedding vector_cosine_ops)"))

    # Add embedding column to memory_entries if missing (for existing DBs before pgvector)
    async with engine.begin() as conn:
        await conn.execute(text(f"""
            DO $$ BEGIN
                ALTER TABLE memory_entries ADD COLUMN embedding vector({_dim});
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """))

    # Warn if configured dimension differs from existing column
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = 'memory_entries'::regclass AND attname = 'embedding' AND atttypmod > 0
        """))
        row = result.fetchone()
        if row and row[0] != _dim:
            logger.warning(
                f"EMBEDDING_DIMENSIONS={_dim} but existing embedding column has dimension {row[0]}. "
                "To change dimensions, DROP the embedding column and re-embed all entries: "
                "ALTER TABLE memory_entries DROP COLUMN embedding; then restart."
            )

    # Ensure meta skills from filesystem are registered in the database
    await _ensure_meta_skills_registered()

    # Ensure seed agents are created
    await _ensure_seed_agents_exist()

    # Ensure default admin user exists
    await _ensure_default_admin()


async def _ensure_meta_skills_registered():
    """
    Ensure all skills from filesystem are registered in the database.
    This syncs filesystem-based skills to the registry on startup.

    - Meta skills (from config.meta_skills) are marked as 'meta' type
    - Other skills are marked as 'user' type
    - Creates skill_versions records with SKILL.md content for skills without versions
    - Applies seed metadata (category, source, author, is_pinned) from seed_skills.json
    - Uses seed_hash for three-way comparison to update seed metadata without overwriting user edits
    """
    from sqlalchemy import text
    from datetime import datetime
    from pathlib import Path
    from app.core.skill_manager import find_all_skills
    import uuid

    meta_skill_names = set(settings.meta_skills)

    # Load seed skill metadata (category, source, author, is_pinned)
    seed_skills = _load_seed_skills()

    # Get ALL filesystem skills
    filesystem_skills = find_all_skills()

    if not filesystem_skills:
        return

    async with AsyncSessionLocal() as session:
        async with session.begin():
            for skill in filesystem_skills:
                # Determine skill type
                is_meta = skill.name in meta_skill_names
                skill_type = "meta" if is_meta else "user"
                skill_id = f"meta-{skill.name}" if is_meta else str(uuid.uuid4())

                # Get seed metadata for this skill
                seed = seed_skills.get(skill.name, {})
                new_seed_hash = _compute_skill_seed_hash(seed) if seed else None

                # Check if skill exists in database
                result = await session.execute(
                    text("SELECT id, skill_type, current_version, category, source, author, is_pinned, seed_hash FROM skills WHERE name = :name"),
                    {"name": skill.name}
                )
                existing = result.fetchone()

                if not existing:
                    # Insert new skill with seed metadata
                    now = datetime.utcnow()
                    await session.execute(
                        text("""
                            INSERT INTO skills (id, name, description, status, skill_type, is_pinned, category, source, author, seed_hash, created_at, updated_at)
                            VALUES (:id, :name, :description, 'active', :skill_type, :is_pinned, :category, :source, :author, :seed_hash, :created_at, :updated_at)
                        """),
                        {
                            "id": skill_id,
                            "name": skill.name,
                            "description": skill.description or f"Skill: {skill.name}",
                            "skill_type": skill_type,
                            "is_pinned": seed.get("is_pinned", False),
                            "category": seed.get("category"),
                            "source": seed.get("source"),
                            "author": seed.get("author"),
                            "seed_hash": new_seed_hash,
                            "created_at": now,
                            "updated_at": now,
                        }
                    )
                    # Create initial version with SKILL.md content
                    await _create_version_from_filesystem(session, skill_id, skill.path, now)
                else:
                    em = existing._mapping
                    existing_id = em["id"]
                    existing_type = em["skill_type"]
                    existing_version = em["current_version"]
                    stored_seed_hash = em["seed_hash"]

                    # Update skill_type if it changed (e.g., user -> meta)
                    if existing_type != skill_type:
                        await session.execute(
                            text("UPDATE skills SET skill_type = :skill_type WHERE name = :name"),
                            {"skill_type": skill_type, "name": skill.name}
                        )
                    # If skill has no version, create one from filesystem
                    if not existing_version:
                        await _create_version_from_filesystem(session, existing_id, skill.path, datetime.utcnow())

                    # Seed metadata hash comparison (only if this skill has seed data)
                    if seed and new_seed_hash:
                        db_seed_dict = _db_row_to_skill_seed_dict(existing)

                        if stored_seed_hash is None:
                            # Case 2: seed_hash IS NULL (first run after migration)
                            # Only backfill the hash — don't update data. See agent Case 2 comment.
                            db_hash = _compute_skill_seed_hash(db_seed_dict)
                            if db_hash == new_seed_hash:
                                backfill_hash = new_seed_hash
                            else:
                                backfill_hash = db_hash
                            await session.execute(
                                text("UPDATE skills SET seed_hash = :seed_hash WHERE id = :id"),
                                {"seed_hash": backfill_hash, "id": existing_id}
                            )
                        elif stored_seed_hash == new_seed_hash:
                            # Case 3: Seed hasn't changed → SKIP
                            pass
                        else:
                            # Case 4: Seed changed
                            db_hash = _compute_skill_seed_hash(db_seed_dict)
                            if db_hash == stored_seed_hash:
                                # User hasn't edited → UPDATE from seed
                                now = datetime.utcnow()
                                await session.execute(
                                    text("""
                                        UPDATE skills SET
                                            category = :category,
                                            source = :source,
                                            author = :author,
                                            is_pinned = :is_pinned,
                                            seed_hash = :seed_hash,
                                            updated_at = :updated_at
                                        WHERE id = :id
                                    """),
                                    {
                                        "id": existing_id,
                                        "category": seed.get("category"),
                                        "source": seed.get("source"),
                                        "author": seed.get("author"),
                                        "is_pinned": seed.get("is_pinned", False),
                                        "seed_hash": new_seed_hash,
                                        "updated_at": now,
                                    }
                                )
                            else:
                                # User edited → don't overwrite data,
                                # but advance seed_hash so we don't re-check every boot
                                await session.execute(
                                    text("UPDATE skills SET seed_hash = :seed_hash WHERE id = :id"),
                                    {"seed_hash": new_seed_hash, "id": existing_id}
                                )


def _load_seed_skills() -> dict:
    """Load seed skill metadata from config/seed_skills.json."""
    for path in [
        Path(settings.config_dir) / "seed_skills.json",
        Path("config/seed_skills.json"),
    ]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("skills", {})
            except Exception as e:
                logger.warning("Failed to load seed_skills.json: %s", e)
                return {}
    return {}


def _compute_agent_seed_hash(data: dict) -> str:
    """Compute SHA-256 hash from agent seed data dict.

    Fields: system_prompt, description, skill_ids(sorted), mcp_servers(sorted),
    builtin_tools(sorted/None), max_turns, model_provider, model_name, executor_name

    Note: description IS included here because agent descriptions come from
    seed_agents.json (unlike skills, where description comes from SKILL.md
    via filesystem sync — see _compute_skill_seed_hash).
    """
    bt = data.get("builtin_tools")  # None = all tools, [] = no tools
    canonical = {
        "system_prompt": data.get("system_prompt") or "",
        "description": data.get("description") or "",
        "skill_ids": sorted(data["skill_ids"]) if data.get("skill_ids") else [],
        "mcp_servers": sorted(data["mcp_servers"]) if data.get("mcp_servers") else [],
        "builtin_tools": sorted(bt) if bt else bt,  # None→None, []→[], non-empty→sorted
        "max_turns": data.get("max_turns", 60),
        "model_provider": data.get("model_provider") or "",
        "model_name": data.get("model_name") or "",
        "executor_name": data.get("executor_name") or "",
    }
    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _compute_skill_seed_hash(data: dict) -> str:
    """Compute SHA-256 hash from skill seed metadata dict.

    Fields: category, source, author, is_pinned

    Note: description is intentionally excluded — it comes from SKILL.md parsing
    during filesystem sync, not from seed_skills.json metadata.
    """
    canonical = {
        "category": data.get("category") or "",
        "source": data.get("source") or "",
        "author": data.get("author") or "",
        "is_pinned": bool(data.get("is_pinned", False)),
    }
    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _parse_jsonb(val):
    """Parse a JSONB value that may be stored as a JSON string or native Python list/dict."""
    if val is None:
        return None
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            logger.warning("_parse_jsonb failed to parse string value: %r", val)
            return val
    return val


def _db_row_to_agent_dict(row) -> dict:
    """Convert a DB row (from agent_presets SELECT) to a dict for hash computation.

    Expects row._mapping (SQLAlchemy Row from raw SQL fetchone()).
    Handles JSONB columns that may be stored as strings or lists.
    """
    m = row._mapping

    return {
        "system_prompt": m.get("system_prompt") or "",
        "description": m.get("description") or "",
        "skill_ids": _parse_jsonb(m.get("skill_ids")),
        "mcp_servers": _parse_jsonb(m.get("mcp_servers")),
        "builtin_tools": _parse_jsonb(m.get("builtin_tools")),
        "max_turns": m.get("max_turns", 60),
        "model_provider": m.get("model_provider") or "",
        "model_name": m.get("model_name") or "",
        "executor_name": m.get("executor_name") or "",
    }


def _db_row_to_skill_seed_dict(row) -> dict:
    """Convert a DB row (from skills SELECT) to a dict for skill seed hash computation.

    Expects row._mapping (SQLAlchemy Row from raw SQL fetchone()).
    """
    m = row._mapping

    return {
        "category": m.get("category") or "",
        "source": m.get("source") or "",
        "author": m.get("author") or "",
        "is_pinned": bool(m.get("is_pinned", False)),
    }


async def _create_version_from_filesystem(session, skill_id: str, skill_dir_path: str, now):
    """Create a skill_versions record from filesystem with all files (SKILL.md + scripts/ + references/ + assets/)."""
    from sqlalchemy import text
    from pathlib import Path
    import uuid

    skill_dir = Path(skill_dir_path)
    skill_md_path = skill_dir / "SKILL.md"

    if not skill_md_path.exists():
        return

    try:
        skill_md_content = skill_md_path.read_text(encoding="utf-8")
    except Exception:
        return

    version = "0.0.1"
    version_id = str(uuid.uuid4())

    # Insert version record
    await session.execute(
        text("""
            INSERT INTO skill_versions (id, skill_id, version, skill_md, created_at, commit_message)
            VALUES (:id, :skill_id, :version, :skill_md, :created_at, :commit_message)
        """),
        {
            "id": version_id,
            "skill_id": skill_id,
            "version": version,
            "skill_md": skill_md_content,
            "created_at": now,
            "commit_message": "Initial version synced from filesystem",
        }
    )

    # Read and save all other files from skill directory
    skill_files = _read_skill_files_for_init(skill_dir)
    for file_path, (content, file_type, size) in skill_files.items():
        content_hash = hashlib.sha256(content).hexdigest()
        file_id = str(uuid.uuid4())
        await session.execute(
            text("""
                INSERT INTO skill_files (id, version_id, file_path, file_type, content, content_hash, size_bytes, created_at)
                VALUES (:id, :version_id, :file_path, :file_type, :content, :content_hash, :size_bytes, NOW())
            """),
            {
                "id": file_id,
                "version_id": version_id,
                "file_path": file_path,
                "file_type": file_type,
                "content": content,
                "content_hash": content_hash,
                "size_bytes": size,
            }
        )

    # Update skill's current_version
    await session.execute(
        text("UPDATE skills SET current_version = :version WHERE id = :id"),
        {"version": version, "id": skill_id}
    )


def _read_skill_files_for_init(skill_dir: Path) -> dict:
    """Read all files from a skill directory for initial registration.

    Returns dict of {relative_path: (content_bytes, file_type, size)}
    Skips SKILL.md (stored separately), binary artifacts, and files larger than 1MB.
    """
    files = {}
    max_size = 1024 * 1024  # 1MB limit

    # File type mapping based on directory
    type_mapping = {
        "scripts": "script",
        "references": "reference",
        "assets": "asset",
    }

    # Skip compiled/build artifacts only (not resource files like images, fonts, etc.)
    skip_extensions = {
        # Python compiled
        ".pyc", ".pyo", ".pyd",
        # Java compiled
        ".class",
        # C/C++ compiled
        ".o", ".a", ".so", ".dylib", ".dll", ".exe",
        # Other build artifacts
        ".wasm",
    }

    for file_path in skill_dir.rglob("*"):
        if not file_path.is_file():
            continue

        # Skip hidden files and common non-essential files
        if file_path.name.startswith(".") or file_path.name.endswith(".pyc"):
            continue
        if "__pycache__" in str(file_path):
            continue
        if file_path.name in ["SKILL.md"]:  # SKILL.md is stored separately
            continue
        if ".backup" in file_path.name or "UPDATE_REPORT" in file_path.name:
            continue

        # Skip compiled/build artifacts
        suffix = file_path.suffix.lower()
        if suffix in skip_extensions:
            continue

        # Skip large files
        try:
            size = file_path.stat().st_size
            if size > max_size:
                continue
        except OSError:
            continue

        # Determine file type
        rel_path = file_path.relative_to(skill_dir)
        parts = rel_path.parts
        file_type = "other"
        if parts and parts[0] in type_mapping:
            file_type = type_mapping[parts[0]]

        # Read file content (binary)
        try:
            content = file_path.read_bytes()
            files[str(rel_path)] = (content, file_type, size)
        except OSError:
            # Skip files that can't be read
            continue

    return files


async def _sync_one_seed_agent(session, agent: dict):
    """
    Sync a single seed agent dict to the database using seed_hash three-way comparison.

    This is the core logic extracted for testability — called by _ensure_seed_agents_exist()
    on startup and directly by integration tests.

    Four cases:
    1. Not in DB → INSERT with seed_hash
    2. In DB, seed_hash IS NULL (migration) → backfill seed_hash
    3. seed_hash == new hash → seed unchanged, SKIP
    4. seed_hash != new hash → seed changed:
       - DB matches stored seed_hash → user didn't edit → UPDATE
       - DB differs from stored seed_hash → user edited → SKIP (advance hash)

    Note: Case 4a UPDATE only syncs content fields (description, system_prompt, skills,
    tools, mcp, max_turns, model, executor). It intentionally does NOT update is_published,
    api_response_mode, or is_system — those are deployment/user actions, not seed data.
    """
    from sqlalchemy import text
    from datetime import datetime
    import uuid

    name = agent.get("name")
    if not name:
        return

    new_seed_hash = _compute_agent_seed_hash(agent)

    # Fetch existing record with all fields needed for hash comparison
    result = await session.execute(
        text("""
            SELECT id, seed_hash, description, system_prompt,
                   skill_ids, mcp_servers, builtin_tools,
                   max_turns, model_provider, model_name, executor_name
            FROM agent_presets WHERE name = :name
        """),
        {"name": name}
    )
    existing = result.fetchone()

    def _dumps(key):
        """Serialize a list field to JSON string, or None if absent."""
        val = agent.get(key)
        return json.dumps(val) if val is not None else None

    if not existing:
        # Case 1: Not in DB → INSERT
        logger.debug("Seed agent '%s': Case 1 — inserting new record", name)
        now = datetime.utcnow()
        agent_id = str(uuid.uuid4())

        await session.execute(
            text("""
                INSERT INTO agent_presets (
                    id, name, description, system_prompt,
                    skill_ids, mcp_servers, builtin_tools,
                    max_turns, model_provider, model_name,
                    executor_name, seed_hash,
                    is_system, is_published, api_response_mode,
                    created_at, updated_at
                ) VALUES (
                    :id, :name, :description, :system_prompt,
                    :skill_ids, :mcp_servers, :builtin_tools,
                    :max_turns, :model_provider, :model_name,
                    :executor_name, :seed_hash,
                    :is_system, :is_published, :api_response_mode,
                    :created_at, :updated_at
                )
            """),
            {
                "id": agent_id,
                "name": name,
                "description": agent.get("description"),
                "system_prompt": agent.get("system_prompt"),
                "skill_ids": _dumps("skill_ids"),
                "mcp_servers": _dumps("mcp_servers"),
                "builtin_tools": _dumps("builtin_tools"),
                "max_turns": agent.get("max_turns", 60),
                "model_provider": agent.get("model_provider"),
                "model_name": agent.get("model_name"),
                "executor_name": agent.get("executor_name"),
                "seed_hash": new_seed_hash,
                "is_system": agent.get("is_system", True),
                "is_published": agent.get("is_published", False),
                "api_response_mode": agent.get("api_response_mode"),
                "created_at": now,
                "updated_at": now,
            }
        )
        return

    em = existing._mapping
    existing_id = em["id"]
    stored_seed_hash = em["seed_hash"]

    if stored_seed_hash is None:
        # Case 2: seed_hash IS NULL (first run after migration)
        # Only backfill the hash — don't update data even if DB matches seed.
        # This is conservative: actual data sync happens on the *next* seed
        # change (Case 4), after we have a reliable baseline hash.
        # If DB != seed now, the first update is deferred to the next boot
        # where seed_agents.json actually changes (two-step).
        db_dict = _db_row_to_agent_dict(existing)
        db_hash = _compute_agent_seed_hash(db_dict)
        if db_hash == new_seed_hash:
            backfill_hash = new_seed_hash
            logger.debug("Seed agent '%s': Case 2a — backfill hash (DB matches seed)", name)
        else:
            backfill_hash = db_hash
            logger.debug("Seed agent '%s': Case 2b — backfill hash (DB diverged from seed)", name)
        await session.execute(
            text("UPDATE agent_presets SET seed_hash = :seed_hash WHERE id = :id"),
            {"seed_hash": backfill_hash, "id": existing_id}
        )
        return

    if stored_seed_hash == new_seed_hash:
        # Case 3: Seed hasn't changed → SKIP
        return

    # Case 4: Seed changed (stored_seed_hash != new_seed_hash)
    db_dict = _db_row_to_agent_dict(existing)
    db_hash = _compute_agent_seed_hash(db_dict)

    if db_hash != stored_seed_hash:
        # User has edited the record → don't overwrite data,
        # but advance seed_hash so we don't re-check every boot
        logger.debug("Seed agent '%s': Case 4b — seed changed but user edited, skipping", name)
        await session.execute(
            text("UPDATE agent_presets SET seed_hash = :seed_hash WHERE id = :id"),
            {"seed_hash": new_seed_hash, "id": existing_id}
        )
        return

    # User hasn't edited (DB still matches stored seed) → UPDATE from seed
    logger.debug("Seed agent '%s': Case 4a — seed changed, updating DB", name)
    now = datetime.utcnow()

    await session.execute(
        text("""
            UPDATE agent_presets SET
                description = :description,
                system_prompt = :system_prompt,
                skill_ids = :skill_ids,
                mcp_servers = :mcp_servers,
                builtin_tools = :builtin_tools,
                max_turns = :max_turns,
                model_provider = :model_provider,
                model_name = :model_name,
                executor_name = :executor_name,
                seed_hash = :seed_hash,
                updated_at = :updated_at
            WHERE id = :id
        """),
        {
            "id": existing_id,
            "description": agent.get("description"),
            "system_prompt": agent.get("system_prompt"),
            "skill_ids": _dumps("skill_ids"),
            "mcp_servers": _dumps("mcp_servers"),
            "builtin_tools": _dumps("builtin_tools"),
            "max_turns": agent.get("max_turns", 60),
            "model_provider": agent.get("model_provider"),
            "model_name": agent.get("model_name"),
            "executor_name": agent.get("executor_name"),
            "seed_hash": new_seed_hash,
            "updated_at": now,
        }
    )


async def _ensure_seed_agents_exist():
    """
    Ensure seed agents from config/seed_agents.json are registered in the database.
    Loads the seed file and delegates per-agent logic to _sync_one_seed_agent().
    """
    # Find seed_agents.json (same precedence as _load_seed_skills: config_dir first)
    seed_file = None
    for path in [
        Path(settings.config_dir) / "seed_agents.json",
        Path("config/seed_agents.json"),
    ]:
        if path.exists():
            seed_file = path
            break

    if not seed_file:
        return

    try:
        with open(seed_file, "r", encoding="utf-8") as f:
            seed_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to load seed_agents.json: %s", e)
        return

    agents = seed_data.get("agents", [])
    if not agents:
        return

    async with AsyncSessionLocal() as session:
        async with session.begin():
            # Migration: rename skill-evolve-helper → agent-skill-evolver
            await _migrate_rename_seed_agent(
                session,
                old_name="skill-evolve-helper",
                new_name="agent-skill-evolver",
            )

            for agent in agents:
                await _sync_one_seed_agent(session, agent)


async def _migrate_rename_seed_agent(session, old_name: str, new_name: str):
    """
    Rename a seed agent in-place, preserving its ID and session history.

    - If old exists and new does not → rename in-place (UPDATE name, SET seed_hash=NULL)
    - If both exist → delete old one
    - If only new exists or neither → no-op

    Setting seed_hash=NULL triggers Case 2 (backfill) on this boot, then normal
    sync on next boot.
    """
    from sqlalchemy import text

    old_row = await session.execute(
        text("SELECT id FROM agent_presets WHERE name = :name"),
        {"name": old_name},
    )
    new_row = await session.execute(
        text("SELECT id FROM agent_presets WHERE name = :name"),
        {"name": new_name},
    )
    old_exists = old_row.fetchone()
    new_exists = new_row.fetchone()

    if old_exists and not new_exists:
        # Rename in-place — preserves ID, sessions, traces
        logger.info("Migrating seed agent '%s' → '%s' (rename in-place)", old_name, new_name)
        await session.execute(
            text("UPDATE agent_presets SET name = :new_name, seed_hash = NULL WHERE name = :old_name"),
            {"new_name": new_name, "old_name": old_name},
        )
    elif old_exists and new_exists:
        # Both exist (shouldn't happen normally) — delete old and its orphan references
        logger.info("Both '%s' and '%s' exist; deleting old '%s'", old_name, new_name, old_name)
        old_id = old_exists._mapping["id"]
        await session.execute(
            text("DELETE FROM published_sessions WHERE agent_id = :id"),
            {"id": old_id},
        )
        # agent_traces has no FK to agent_presets, but clean up to avoid orphan records
        await session.execute(
            text("UPDATE agent_traces SET preset_id = NULL WHERE preset_id = :id"),
            {"id": old_id},
        )
        await session.execute(
            text("DELETE FROM agent_presets WHERE id = :id"),
            {"id": old_id},
        )


async def _ensure_default_admin():
    """Create default admin user if no users exist."""
    from sqlalchemy import text
    import uuid

    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(text("SELECT COUNT(*) FROM users"))
            count = result.scalar()
            if count == 0:
                try:
                    import bcrypt
                    password_hash = bcrypt.hashpw(b"admin", bcrypt.gensalt()).decode("utf-8")
                except Exception:
                    # Fallback if bcrypt not installed yet
                    password_hash = ""
                    print("Warning: bcrypt not available, default admin created without password")
                    return

                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                await session.execute(
                    text("""
                        INSERT INTO users (id, username, password_hash, display_name, role, is_active, must_change_password, created_at, updated_at)
                        VALUES (:id, :username, :password_hash, :display_name, :role, :is_active, :must_change_password, :created_at, :updated_at)
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "username": "admin",
                        "password_hash": password_hash,
                        "display_name": "Administrator",
                        "role": "admin",
                        "is_active": True,
                        "must_change_password": True,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                print("Created default admin user (admin/admin) - password change required on first login")


async def drop_db():
    """
    Drop all database tables.

    WARNING: This is destructive. Use only for testing.
    """
    from app.db import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
