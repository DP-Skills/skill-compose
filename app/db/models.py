"""
SQLAlchemy ORM models for Skill Registry.

Tables:
- skills: Main skill registry
- skill_versions: Version history for each skill
- skill_files: Files associated with a version (resources, scripts)
- skill_tests: Test cases for regression testing
- skill_changelogs: Audit trail for changes
- memory_entries: Vector-searchable memory entries for agents
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import (
    String,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    LargeBinary,
    UniqueConstraint,
    Index,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None  # pgvector not installed — embedding column omitted from ORM

from app.db.database import Base


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


class UserDB(Base):
    """
    User accounts for authentication.
    """
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    username: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    password_hash: Mapped[str] = mapped_column(
        String(256), nullable=False
    )
    display_name: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True
    )
    role: Mapped[str] = mapped_column(
        String(32), default="user", nullable=False
    )  # "admin" or "user"
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )
    must_change_password: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    password_changed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<User(username={self.username}, role={self.role})>"


class SkillDB(Base):
    """
    Main skill registry table.

    Each skill has a unique name and can have multiple versions.
    """
    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    name: Mapped[str] = mapped_column(
        String(128), unique=True, nullable=False, index=True
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    owner_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True
    )  # Reserved for multi-tenancy
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    current_version: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )  # e.g., "1.2.3"
    status: Mapped[str] = mapped_column(
        String(32), default="draft", nullable=False
    )  # draft/active/deprecated
    skill_type: Mapped[str] = mapped_column(
        String(32), default="user", nullable=False
    )  # user/meta
    tags: Mapped[Optional[List[str]]] = mapped_column(
        JSONB, nullable=True, default=list
    )  # categorization tags
    icon_url: Mapped[Optional[str]] = mapped_column(
        String(512), nullable=True
    )  # URL to skill icon image
    source: Mapped[Optional[str]] = mapped_column(
        String(1024), nullable=True
    )  # Import source URL (e.g., GitHub URL)
    author: Mapped[Optional[str]] = mapped_column(
        String(256), nullable=True
    )  # Author or organization name
    category: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )  # Skill category for grouping
    is_pinned: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false"), nullable=False
    )  # Whether skill is pinned to top
    seed_hash: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )  # SHA-256 hash of seed data for change detection

    # Relationships
    versions: Mapped[List["SkillVersionDB"]] = relationship(
        "SkillVersionDB", back_populates="skill", cascade="all, delete-orphan"
    )
    changelogs: Mapped[List["SkillChangelogDB"]] = relationship(
        "SkillChangelogDB", back_populates="skill", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Skill(name={self.name}, version={self.current_version}, status={self.status})>"


class SkillVersionDB(Base):
    """
    Version history for skills.

    Each version stores the complete skill.md content and schema.
    """
    __tablename__ = "skill_versions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    skill_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("skills.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # SemVer: "1.2.3"
    parent_version: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )  # For diff tracking
    skill_md: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # SKILL.md content
    schema_json: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )  # Input/output schema
    manifest_json: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )  # manifest.json content
    extra_metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )  # Additional metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    created_by: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True
    )  # User who created this version
    commit_message: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Version description

    # Relationships
    skill: Mapped["SkillDB"] = relationship("SkillDB", back_populates="versions")
    files: Mapped[List["SkillFileDB"]] = relationship(
        "SkillFileDB", back_populates="version", cascade="all, delete-orphan"
    )
    tests: Mapped[List["SkillTestDB"]] = relationship(
        "SkillTestDB", back_populates="version", cascade="all, delete-orphan"
    )

    # Unique constraint: skill + version
    __table_args__ = (
        UniqueConstraint("skill_id", "version", name="uq_skill_version"),
        Index("ix_skill_versions_skill_id", "skill_id"),
    )

    def __repr__(self) -> str:
        return f"<SkillVersion(skill_id={self.skill_id}, version={self.version})>"


class SkillFileDB(Base):
    """
    Files associated with a skill version.

    Stores resources, scripts, and other files.
    Small files are stored in content, large files use storage_path.
    """
    __tablename__ = "skill_files"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("skill_versions.id", ondelete="CASCADE"), nullable=False
    )
    file_path: Mapped[str] = mapped_column(
        String(512), nullable=False
    )  # Relative path within skill package
    file_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # resource/script/test/other
    content_hash: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )  # SHA256 hash
    content: Mapped[Optional[bytes]] = mapped_column(
        LargeBinary, nullable=True
    )  # Small file content
    storage_path: Mapped[Optional[str]] = mapped_column(
        String(512), nullable=True
    )  # Path for large files
    size_bytes: Mapped[Optional[int]] = mapped_column(
        nullable=True
    )  # File size
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationships
    version: Mapped["SkillVersionDB"] = relationship(
        "SkillVersionDB", back_populates="files"
    )

    __table_args__ = (
        UniqueConstraint("version_id", "file_path", name="uq_version_file"),
        Index("ix_skill_files_version_id", "version_id"),
    )

    def __repr__(self) -> str:
        return f"<SkillFile(version_id={self.version_id}, path={self.file_path})>"


class SkillTestDB(Base):
    """
    Test cases for skill regression testing.

    Stores input/output pairs for automated testing.
    """
    __tablename__ = "skill_tests"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("skill_versions.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(
        String(128), nullable=False
    )  # Test case name
    description: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    input_data: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )  # Test input
    expected_output: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )  # Expected result
    is_golden: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )  # Golden test case
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationships
    version: Mapped["SkillVersionDB"] = relationship(
        "SkillVersionDB", back_populates="tests"
    )

    __table_args__ = (
        Index("ix_skill_tests_version_id", "version_id"),
    )

    def __repr__(self) -> str:
        return f"<SkillTest(version_id={self.version_id}, name={self.name})>"


class AgentTraceDB(Base):
    """
    Agent execution traces.

    Stores complete execution history for debugging, replay, and self-evolution.
    """
    __tablename__ = "agent_traces"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    request: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # User's original request
    skills_used: Mapped[Optional[List[str]]] = mapped_column(
        JSONB, nullable=True
    )  # List of skills that were activated/used
    model_provider: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )  # LLM provider: anthropic, openrouter, openai, google
    model: Mapped[str] = mapped_column(
        String(128), nullable=False
    )  # Model used (e.g., claude-sonnet-4-5-20250929)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="running"
    )  # running/completed/failed
    success: Mapped[bool] = mapped_column(
        Boolean, nullable=False
    )  # Whether execution succeeded
    answer: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Final answer
    error: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Error message if failed
    total_turns: Mapped[int] = mapped_column(
        nullable=False, default=0
    )  # Number of agent turns
    total_input_tokens: Mapped[int] = mapped_column(
        nullable=False, default=0
    )  # Total input tokens used
    total_output_tokens: Mapped[int] = mapped_column(
        nullable=False, default=0
    )  # Total output tokens used
    steps: Mapped[Optional[List[dict]]] = mapped_column(
        JSONB, nullable=True
    )  # List of AgentStep as dicts
    llm_calls: Mapped[Optional[List[dict]]] = mapped_column(
        JSONB, nullable=True
    )  # List of LLMCall as dicts
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    duration_ms: Mapped[Optional[int]] = mapped_column(
        nullable=True
    )  # Execution duration in milliseconds
    executor_name: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )  # Executor used (e.g., "remotion", "base", None for local)
    session_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True
    )  # Session ID linking this trace to a chat session

    __table_args__ = (
        Index("ix_agent_traces_created_at", "created_at"),
        Index("ix_agent_traces_success", "success"),
        Index("ix_agent_traces_session_id", "session_id"),
    )

    def __repr__(self) -> str:
        return f"<AgentTrace(id={self.id}, success={self.success}, turns={self.total_turns})>"


class BackgroundTaskDB(Base):
    """
    Background task tracking.

    Persists task state to survive server restarts.
    """
    __tablename__ = "background_tasks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    task_type: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # create_skill, evolve_skill, etc.
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending"
    )  # pending, running, completed, failed
    metadata_json: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )  # Task-specific metadata (skill_name, etc.)
    result_json: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )  # Task result
    error: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Error message if failed
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )

    __table_args__ = (
        Index("ix_background_tasks_status", "status"),
        Index("ix_background_tasks_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<BackgroundTask(id={self.id}, type={self.task_type}, status={self.status})>"


class ExecutorDB(Base):
    """
    Executor configuration for Agent code execution environments.

    Each executor represents a Docker container with a specific runtime environment
    (Python version, ML libraries, GPU support, etc.).
    """
    __tablename__ = "executors"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    name: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )  # base, ml, cuda, custom-name
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    image: Mapped[str] = mapped_column(
        String(255), nullable=False
    )  # Docker image: ghcr.io/skill-composer/executor:ml
    port: Mapped[int] = mapped_column(
        nullable=False, default=9000
    )  # Executor service port

    # Resource limits
    memory_limit: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True, default="2G"
    )  # Docker memory limit: 2G, 8G, etc.
    cpu_limit: Mapped[Optional[float]] = mapped_column(
        nullable=True
    )  # CPU core limit: 1.0, 2.0, etc.
    gpu_required: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )  # Whether GPU is required

    # Metadata
    is_builtin: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )  # Built-in executors cannot be deleted
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<Executor(name={self.name}, image={self.image}, is_builtin={self.is_builtin})>"


class AgentPresetDB(Base):
    """
    Agent Preset configuration.

    Stores reusable Agent configurations with specific combinations of
    system prompt, skills, tools, MCP servers, and max turns.
    """
    __tablename__ = "agent_presets"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    name: Mapped[str] = mapped_column(
        String(128), unique=True, nullable=False, index=True
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    system_prompt: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Custom system prompt
    skill_ids: Mapped[Optional[List[str]]] = mapped_column(
        JSONB, nullable=True
    )  # List of skill names to bind
    mcp_servers: Mapped[Optional[List[str]]] = mapped_column(
        JSONB, nullable=True
    )  # List of MCP server names to enable
    builtin_tools: Mapped[Optional[List[str]]] = mapped_column(
        JSONB, nullable=True
    )  # List of built-in tools to enable (null = all)
    max_turns: Mapped[int] = mapped_column(
        nullable=False, default=30
    )  # Default max turns
    model_provider: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )  # LLM provider: anthropic, openrouter, openai, google
    model_name: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True
    )  # Model name/ID for the provider
    is_system: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )  # System preset vs user-created
    is_published: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )  # Whether this preset is published (publicly accessible)
    api_response_mode: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )  # "streaming" or "non_streaming", null = unpublished
    executor_name: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # Executor name (e.g. "base", "ml", "cuda") for code execution
    seed_hash: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )  # SHA-256 hash of seed data for change detection
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_agent_presets_is_system", "is_system"),
    )

    def __repr__(self) -> str:
        return f"<AgentPreset(name={self.name}, is_system={self.is_system})>"


class PublishedSessionDB(Base):
    """
    Published agent chat sessions.

    Stores conversation history for published agent sessions,
    enabling server-side session management instead of client-side history.
    """
    __tablename__ = "published_sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    agent_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )
    messages: Mapped[Optional[List[dict]]] = mapped_column(
        JSONB, nullable=True, default=list
    )  # Append-only display history — never compressed, preserves all tool_use/tool_result blocks
    agent_context: Mapped[Optional[List[dict]]] = mapped_column(
        JSONB, nullable=True, default=None
    )  # Agent working message list — whole-replaced each request, may contain compression summaries
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<PublishedSession(id={self.id}, agent_id={self.agent_id})>"


class ScheduledTaskDB(Base):
    """
    Scheduled tasks for periodic agent execution.

    Supports cron, interval, and one-time schedules.
    """
    __tablename__ = "scheduled_tasks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    name: Mapped[str] = mapped_column(
        String(128), unique=True, nullable=False, index=True
    )
    agent_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_presets.id", ondelete="CASCADE"), nullable=False
    )
    prompt: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # The prompt to send to the agent
    schedule_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # cron / interval / once
    schedule_value: Mapped[str] = mapped_column(
        String(128), nullable=False
    )  # cron expr or interval seconds or ISO datetime
    context_mode: Mapped[str] = mapped_column(
        String(32), default="isolated", nullable=False
    )  # isolated / session
    session_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True
    )  # For session context_mode
    channel_binding_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True
    )  # Optional: send results to a channel
    delivery_to: Mapped[Optional[str]] = mapped_column(
        String(256), nullable=True
    )  # Explicit target chat_id (overrides binding's external_id)
    status: Mapped[str] = mapped_column(
        String(32), default="active", nullable=False
    )  # active / paused / completed
    next_run: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    last_run: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    max_runs: Mapped[Optional[int]] = mapped_column(
        nullable=True
    )  # null = unlimited
    run_count: Mapped[int] = mapped_column(
        nullable=False, default=0
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    run_logs: Mapped[List["TaskRunLogDB"]] = relationship(
        "TaskRunLogDB", back_populates="task", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_scheduled_tasks_status", "status"),
        Index("ix_scheduled_tasks_next_run", "next_run"),
    )

    def __repr__(self) -> str:
        return f"<ScheduledTask(name={self.name}, status={self.status})>"


class TaskRunLogDB(Base):
    """
    Execution log for scheduled task runs.
    """
    __tablename__ = "task_run_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("scheduled_tasks.id", ondelete="CASCADE"), nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    duration_ms: Mapped[Optional[int]] = mapped_column(
        nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="running"
    )  # running / completed / failed
    result_summary: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    error: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    trace_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationships
    task: Mapped["ScheduledTaskDB"] = relationship(
        "ScheduledTaskDB", back_populates="run_logs"
    )

    __table_args__ = (
        Index("ix_task_run_logs_task_id", "task_id"),
        Index("ix_task_run_logs_started_at", "started_at"),
    )

    def __repr__(self) -> str:
        return f"<TaskRunLog(task_id={self.task_id}, status={self.status})>"


class ChannelBindingDB(Base):
    """
    Channel bindings connect external messaging platforms to published agents.
    """
    __tablename__ = "channel_bindings"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    channel_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # feishu / telegram / webhook
    external_id: Mapped[str] = mapped_column(
        String(256), nullable=False
    )  # Platform-side group/chat ID
    name: Mapped[str] = mapped_column(
        String(128), nullable=False
    )
    agent_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_presets.id", ondelete="CASCADE"), nullable=False
    )
    trigger_pattern: Mapped[Optional[str]] = mapped_column(
        String(512), nullable=True
    )  # Regex pattern to trigger the agent
    enabled: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )
    config: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )  # Adapter-specific configuration
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    messages: Mapped[List["ChannelMessageDB"]] = relationship(
        "ChannelMessageDB", back_populates="binding", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # UniqueConstraint replaced by partial indexes (uq_channel_binding_specific,
        # uq_channel_binding_global) created in _run_migrations() for global binding support
        Index("ix_channel_bindings_channel_type", "channel_type"),
    )

    def __repr__(self) -> str:
        return f"<ChannelBinding(name={self.name}, type={self.channel_type})>"


class ChannelMessageDB(Base):
    """
    Messages sent/received through channel bindings.
    """
    __tablename__ = "channel_messages"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    channel_binding_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("channel_bindings.id", ondelete="CASCADE"), nullable=False
    )
    direction: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # inbound / outbound
    external_message_id: Mapped[Optional[str]] = mapped_column(
        String(256), nullable=True
    )
    sender_id: Mapped[Optional[str]] = mapped_column(
        String(256), nullable=True
    )
    sender_name: Mapped[Optional[str]] = mapped_column(
        String(256), nullable=True
    )
    content: Mapped[str] = mapped_column(
        Text, nullable=False
    )
    message_type: Mapped[str] = mapped_column(
        String(32), default="text", nullable=False
    )  # text / image / file
    msg_metadata: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSONB, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationships
    binding: Mapped["ChannelBindingDB"] = relationship(
        "ChannelBindingDB", back_populates="messages"
    )

    __table_args__ = (
        Index("ix_channel_messages_binding_id", "channel_binding_id"),
        Index("ix_channel_messages_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<ChannelMessage(direction={self.direction}, type={self.message_type})>"


class SkillChangelogDB(Base):
    """
    Audit trail for skill changes.

    Records who changed what, when, and why.
    """
    __tablename__ = "skill_changelogs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    skill_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("skills.id", ondelete="CASCADE"), nullable=False
    )
    version_from: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )  # Previous version
    version_to: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )  # New version
    change_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # create/update/rollback/delete
    diff_content: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Unified diff
    changed_by: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True
    )  # User who made the change
    changed_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    comment: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Change description

    # Relationships
    skill: Mapped["SkillDB"] = relationship("SkillDB", back_populates="changelogs")

    __table_args__ = (
        Index("ix_skill_changelogs_skill_id", "skill_id"),
        Index("ix_skill_changelogs_changed_at", "changed_at"),
    )

    def __repr__(self) -> str:
        return f"<SkillChangelog(skill_id={self.skill_id}, type={self.change_type})>"


class MemoryEntryDB(Base):
    """
    Vector-searchable memory entries for agents.

    Stores facts, preferences, procedures, and context that agents can
    recall via semantic search. Supports global (agent_id=NULL) and
    per-agent scopes.

    Note: The `embedding` column uses pgvector's Vector type and is created
    via raw SQL migration in database.py (dimension controlled by
    EMBEDDING_DIMENSIONS config). The ORM column is defined when pgvector
    is installed; otherwise it is omitted (raw SQL still works).
    """
    __tablename__ = "memory_entries"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid
    )
    agent_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True
    )  # NULL = global memory
    content: Mapped[str] = mapped_column(
        Text, nullable=False
    )
    category: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )  # fact / preference / procedure / context / session_summary
    source: Mapped[Optional[str]] = mapped_column(
        String(256), nullable=True
    )  # manual / auto_flush / agent_tool / session_end
    embedding_model: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc), nullable=False
    )

    __table_args__ = (
        Index("ix_memory_entries_agent_id", "agent_id"),
        Index("ix_memory_entries_category", "category"),
        Index("ix_memory_entries_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<MemoryEntry(id={self.id}, agent_id={self.agent_id}, category={self.category})>"


# Embedding column for MemoryEntryDB — added via Column to the underlying Table
# because the Vector type requires pgvector to be installed. When pgvector is absent,
# the column is omitted from the ORM but still exists in the DB (created by raw SQL
# migration in database.py). All embedding operations use raw SQL, so this is safe.
#
# The dimension reads from EMBEDDING_DIMENSIONS at import time. If the setting is
# unavailable (e.g. in tests without env), the default 1536 from BaseSettings applies.
if Vector is not None:
    from sqlalchemy import Column
    try:
        from app.config import get_settings as _get_settings
        _embedding_dim = _get_settings().embedding_dimensions
    except Exception:
        _embedding_dim = 1536
    MemoryEntryDB.__table__.append_column(
        Column("embedding", Vector(_embedding_dim), nullable=True)
    )
