"""
Tests for the unified agent execution service layer (app/services/agent_runner.py).

Covers:
- AgentConfig dataclass defaults and construction
- config_from_preset() mapping from AgentPresetDB
- create_agent() SkillsAgent construction
- build_completed_trace() trace creation from agent result
- build_initial_trace() "running" trace creation for streaming
- Integration: scheduler._execute_task uses shared service
- Integration: channel_manager._run_agent uses shared service (trace creation fix)
"""

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentPresetDB, AgentTraceDB, PublishedSessionDB
from app.services.agent_runner import (
    AgentConfig,
    config_from_preset,
    create_agent,
    build_completed_trace,
    build_initial_trace,
)


# ---------------------------------------------------------------------------
# Mock dataclasses for AgentResult
# ---------------------------------------------------------------------------

@dataclass
class MockStep:
    role: str = "assistant"
    content: str = "Mock answer"
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_result: Optional[str] = None


@dataclass
class MockLLMCall:
    turn: int = 1
    timestamp: str = "2024-01-01T00:00:00"
    model: str = "kimi-k2.5"
    input_tokens: int = 100
    output_tokens: int = 50
    stop_reason: str = "end_turn"


@dataclass
class MockAgentResult:
    success: bool = True
    answer: str = "Mock answer"
    total_turns: int = 3
    total_input_tokens: int = 500
    total_output_tokens: int = 250
    steps: List = field(default_factory=lambda: [MockStep()])
    llm_calls: List = field(default_factory=lambda: [MockLLMCall()])
    error: Optional[str] = None
    output_files: List = field(default_factory=list)
    skills_used: List = field(default_factory=lambda: ["test-skill"])
    final_messages: Optional[List] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AgentConfig tests
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_defaults(self):
        """AgentConfig has sensible defaults."""
        config = AgentConfig()
        assert config.model_provider is None
        assert config.model_name is None
        assert config.max_turns == 60
        assert config.skills is None
        assert config.allowed_tools is None
        assert config.equipped_mcp_servers is None
        assert config.system_prompt is None
        assert config.executor_name is None
        assert config.agent_id is None
        assert config.is_meta_agent is False
        assert config.verbose is True

    def test_custom_construction(self):
        """AgentConfig constructed with custom values preserves them."""
        config = AgentConfig(
            model_provider="anthropic",
            model_name="claude-sonnet-4-5-20250929",
            max_turns=10,
            skills=["skill-a", "skill-b"],
            allowed_tools=["execute_code"],
            equipped_mcp_servers=["fetch"],
            system_prompt="You are a helpful assistant",
            executor_name="base",
            agent_id="preset-123",
            is_meta_agent=True,
            verbose=False,
        )
        assert config.model_provider == "anthropic"
        assert config.model_name == "claude-sonnet-4-5-20250929"
        assert config.max_turns == 10
        assert config.skills == ["skill-a", "skill-b"]
        assert config.allowed_tools == ["execute_code"]
        assert config.equipped_mcp_servers == ["fetch"]
        assert config.system_prompt == "You are a helpful assistant"
        assert config.executor_name == "base"
        assert config.agent_id == "preset-123"
        assert config.is_meta_agent is True
        assert config.verbose is False

    def test_mutable_after_creation(self):
        """AgentConfig fields can be mutated (used for model override in preset mode)."""
        config = AgentConfig(model_provider="kimi")
        config.model_provider = "anthropic"
        config.verbose = False
        assert config.model_provider == "anthropic"
        assert config.verbose is False


# ---------------------------------------------------------------------------
# config_from_preset tests
# ---------------------------------------------------------------------------


class TestConfigFromPreset:
    def _make_preset(self, **overrides) -> AgentPresetDB:
        """Create a test AgentPresetDB."""
        defaults = dict(
            id=str(uuid.uuid4()),
            name="test-preset",
            description="A test preset",
            system_prompt="Be helpful",
            skill_ids=["skill-a"],
            mcp_servers=["fetch"],
            builtin_tools=["execute_code", "read_file"],
            max_turns=30,
            model_provider="anthropic",
            model_name="claude-sonnet-4-5-20250929",
            is_system=False,
            executor_name="base",
        )
        defaults.update(overrides)
        return AgentPresetDB(**defaults)

    def test_basic_mapping(self):
        """config_from_preset maps all preset fields to AgentConfig."""
        preset = self._make_preset()
        config = config_from_preset(preset)

        assert config.model_provider == "anthropic"
        assert config.model_name == "claude-sonnet-4-5-20250929"
        assert config.max_turns == 30
        assert config.skills == ["skill-a"]
        assert config.allowed_tools == ["execute_code", "read_file"]
        assert config.equipped_mcp_servers == ["fetch"]
        assert config.system_prompt == "Be helpful"
        assert config.executor_name == "base"
        assert config.agent_id == preset.id
        assert config.is_meta_agent is False

    def test_system_preset_maps_to_meta_agent(self):
        """is_system=True on preset maps to is_meta_agent=True."""
        preset = self._make_preset(is_system=True)
        config = config_from_preset(preset)
        assert config.is_meta_agent is True

    def test_none_model_fields(self):
        """Preset with None model fields produces config with None model fields."""
        preset = self._make_preset(model_provider=None, model_name=None)
        config = config_from_preset(preset)
        assert config.model_provider is None
        assert config.model_name is None

    def test_none_executor_treated_as_none(self):
        """Preset with empty executor_name is normalized to None."""
        preset = self._make_preset(executor_name="")
        config = config_from_preset(preset)
        assert config.executor_name is None

    def test_none_max_turns_defaults_to_60(self):
        """Preset with max_turns=None defaults to 60."""
        preset = self._make_preset(max_turns=None)
        config = config_from_preset(preset)
        assert config.max_turns == 60

    def test_default_verbose_is_true(self):
        """config_from_preset always sets verbose=True (caller can override for streaming)."""
        preset = self._make_preset()
        config = config_from_preset(preset)
        assert config.verbose is True


# ---------------------------------------------------------------------------
# create_agent tests
# ---------------------------------------------------------------------------


class TestCreateAgent:
    @patch("app.agent.SkillsAgent")
    def test_passes_all_config_fields(self, MockAgent):
        """create_agent passes all AgentConfig fields to SkillsAgent constructor."""
        mock_instance = MagicMock()
        MockAgent.return_value = mock_instance

        config = AgentConfig(
            model_provider="anthropic",
            model_name="claude-sonnet-4-5-20250929",
            max_turns=15,
            skills=["my-skill"],
            allowed_tools=["execute_code"],
            equipped_mcp_servers=["fetch"],
            system_prompt="Custom prompt",
            executor_name="base",
            is_meta_agent=True,
            verbose=False,
        )

        result = create_agent(config, workspace_id="ws-123")

        MockAgent.assert_called_once_with(
            model="claude-sonnet-4-5-20250929",
            model_provider="anthropic",
            max_turns=15,
            verbose=False,
            allowed_skills=["my-skill"],
            allowed_tools=["execute_code"],
            equipped_mcp_servers=["fetch"],
            custom_system_prompt="Custom prompt",
            executor_name="base",
            workspace_id="ws-123",
            is_meta_agent=True,
        )
        assert result is mock_instance

    @patch("app.agent.SkillsAgent")
    def test_none_workspace_id(self, MockAgent):
        """create_agent without workspace_id passes None."""
        MockAgent.return_value = MagicMock()
        config = AgentConfig()
        create_agent(config)
        call_kwargs = MockAgent.call_args[1]
        assert call_kwargs["workspace_id"] is None

    @patch("app.agent.SkillsAgent")
    def test_defaults_produce_valid_call(self, MockAgent):
        """create_agent with default AgentConfig doesn't crash."""
        MockAgent.return_value = MagicMock()
        config = AgentConfig()
        agent = create_agent(config)
        assert agent is not None
        MockAgent.assert_called_once()


# ---------------------------------------------------------------------------
# build_completed_trace tests
# ---------------------------------------------------------------------------


class TestBuildCompletedTrace:
    def _make_mock_agent(self):
        agent = MagicMock()
        agent.model_provider = "anthropic"
        agent.model = "claude-sonnet-4-5-20250929"
        return agent

    def test_success_trace(self):
        """build_completed_trace creates a proper success trace."""
        result = MockAgentResult()
        agent = self._make_mock_agent()

        trace = build_completed_trace(
            request_text="Test request",
            result=result,
            agent=agent,
            duration_ms=1500,
            executor_name="base",
            session_id="session-123",
        )

        assert isinstance(trace, AgentTraceDB)
        assert trace.request == "Test request"
        assert trace.status == "completed"
        assert trace.success is True
        assert trace.answer == "Mock answer"
        assert trace.error is None
        assert trace.total_turns == 3
        assert trace.total_input_tokens == 500
        assert trace.total_output_tokens == 250
        assert trace.model_provider == "anthropic"
        assert trace.model == "claude-sonnet-4-5-20250929"
        assert trace.duration_ms == 1500
        assert trace.executor_name == "base"
        assert trace.session_id == "session-123"
        assert trace.skills_used == ["test-skill"]
        assert len(trace.steps) == 1
        assert len(trace.llm_calls) == 1

    def test_failure_trace(self):
        """build_completed_trace with failed result sets status='failed'."""
        result = MockAgentResult(
            success=False,
            answer="",
            error="LLM stream failed",
        )
        agent = self._make_mock_agent()

        trace = build_completed_trace(
            request_text="Failing request",
            result=result,
            agent=agent,
            duration_ms=500,
        )

        assert trace.status == "failed"
        assert trace.success is False
        assert trace.error == "LLM stream failed"

    def test_empty_skills_used(self):
        """build_completed_trace handles None skills_used gracefully."""
        result = MockAgentResult(skills_used=None)
        agent = self._make_mock_agent()

        trace = build_completed_trace(
            request_text="Test",
            result=result,
            agent=agent,
            duration_ms=100,
        )

        assert trace.skills_used == []

    def test_no_executor_or_session(self):
        """build_completed_trace works without executor_name and session_id."""
        result = MockAgentResult()
        agent = self._make_mock_agent()

        trace = build_completed_trace(
            request_text="Test",
            result=result,
            agent=agent,
            duration_ms=100,
        )

        assert trace.executor_name is None
        assert trace.session_id is None

    def test_trace_uses_agent_resolved_model(self):
        """build_completed_trace reads model from agent (which has defaults resolved), not config."""
        result = MockAgentResult()
        agent = MagicMock()
        agent.model_provider = "kimi"  # Agent resolved the default
        agent.model = "kimi-k2.5"

        trace = build_completed_trace(
            request_text="Test",
            result=result,
            agent=agent,
            duration_ms=100,
        )

        assert trace.model_provider == "kimi"
        assert trace.model == "kimi-k2.5"

    def test_trace_is_unsaved(self):
        """build_completed_trace returns an unsaved ORM object (no id set by default mechanism)."""
        result = MockAgentResult()
        agent = self._make_mock_agent()

        trace = build_completed_trace(
            request_text="Test",
            result=result,
            agent=agent,
            duration_ms=100,
        )

        # ORM default generates UUID, but object is not in a session
        assert trace.request == "Test"  # Field is set
        # Not committed — we can check that no session is attached
        from sqlalchemy import inspect as sa_inspect
        assert sa_inspect(trace).transient is True


# ---------------------------------------------------------------------------
# build_initial_trace tests
# ---------------------------------------------------------------------------


class TestBuildInitialTrace:
    def test_running_state(self):
        """build_initial_trace creates a trace in 'running' state with zero counters."""
        trace = build_initial_trace(
            request_text="Stream request",
            model_provider="anthropic",
            model_name="claude-sonnet-4-5-20250929",
            executor_name="base",
            session_id="session-456",
        )

        assert isinstance(trace, AgentTraceDB)
        assert trace.request == "Stream request"
        assert trace.status == "running"
        assert trace.success is False
        assert trace.answer == ""
        assert trace.error is None
        assert trace.total_turns == 0
        assert trace.total_input_tokens == 0
        assert trace.total_output_tokens == 0
        assert trace.skills_used == []
        assert trace.steps == []
        assert trace.llm_calls == []
        assert trace.duration_ms == 0
        assert trace.model_provider == "anthropic"
        assert trace.model == "claude-sonnet-4-5-20250929"
        assert trace.executor_name == "base"
        assert trace.session_id == "session-456"

    def test_no_executor_or_session(self):
        """build_initial_trace works without optional fields."""
        trace = build_initial_trace(
            request_text="Test",
            model_provider="kimi",
            model_name="kimi-k2.5",
        )

        assert trace.executor_name is None
        assert trace.session_id is None


# ---------------------------------------------------------------------------
# Integration: agent.py uses shared service
# ---------------------------------------------------------------------------


class TestAgentAPIIntegration:
    """Verify that the agent API endpoints use the shared service layer."""

    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock,
           return_value=MagicMock(session_id="test-sid", agent_context=None, display_messages=[]))
    @patch("app.api.v1.agent.create_agent")
    async def test_run_endpoint_uses_create_agent(self, mock_create, _mock_load, _mock_save, client):
        """POST /agent/run calls create_agent from shared service."""
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MockAgentResult())
        mock_agent.model = "kimi-k2.5"
        mock_agent.model_provider = "kimi"
        mock_agent.cleanup = MagicMock()
        mock_create.return_value = mock_agent

        resp = await client.post(
            "/api/v1/agent/run",
            json={"request": "Test", "session_id": "test-sid"},
        )

        assert resp.status_code == 200
        mock_create.assert_called_once()
        # Verify first arg is an AgentConfig
        config_arg = mock_create.call_args[0][0]
        assert isinstance(config_arg, AgentConfig)

    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock,
           return_value=MagicMock(session_id="test-sid", agent_context=None, display_messages=[]))
    @patch("app.api.v1.agent.create_agent")
    async def test_run_endpoint_uses_build_completed_trace(self, mock_create, _mock_load, _mock_save, client, db_session):
        """POST /agent/run uses build_completed_trace and saves to DB."""
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MockAgentResult())
        mock_agent.model = "test-model"
        mock_agent.model_provider = "test-provider"
        mock_agent.cleanup = MagicMock()
        mock_create.return_value = mock_agent

        resp = await client.post(
            "/api/v1/agent/run",
            json={"request": "Trace test", "session_id": "test-sid"},
        )

        assert resp.status_code == 200
        body = resp.json()
        # Trace should have been saved
        assert body["trace_id"] is not None

        # Verify trace in DB
        from sqlalchemy import select
        result = await db_session.execute(
            select(AgentTraceDB).where(AgentTraceDB.id == body["trace_id"])
        )
        trace = result.scalar_one_or_none()
        assert trace is not None
        assert trace.model_provider == "test-provider"
        assert trace.model == "test-model"
        assert trace.status == "completed"

    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock,
           return_value=MagicMock(session_id="test-sid", agent_context=None, display_messages=[]))
    @patch("app.api.v1.agent.create_agent")
    async def test_run_endpoint_calls_cleanup(self, mock_create, _mock_load, _mock_save, client):
        """POST /agent/run always calls agent.cleanup() in finally block."""
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MockAgentResult())
        mock_agent.model = "kimi-k2.5"
        mock_agent.model_provider = "kimi"
        mock_agent.cleanup = MagicMock()
        mock_create.return_value = mock_agent

        await client.post(
            "/api/v1/agent/run",
            json={"request": "Test", "session_id": "test-sid"},
        )

        mock_agent.cleanup.assert_called_once()


# ---------------------------------------------------------------------------
# Integration: scheduler uses shared service
# ---------------------------------------------------------------------------


class TestSchedulerIntegration:
    """Verify that the scheduler uses the shared service layer."""

    @patch("app.agent.SkillsAgent")
    def test_execute_task_uses_config_from_preset(self, MockSkillsAgent):
        """scheduler._execute_task constructs agent via config_from_preset + create_agent.

        Uses mock SyncSessionLocal to avoid needing real test tables for the
        sync DB path the scheduler uses.
        """
        from app.services.scheduler import TaskScheduler
        from app.db.models import ScheduledTaskDB, TaskRunLogDB, generate_uuid
        import asyncio

        # Build in-memory test objects
        preset = AgentPresetDB(
            id=str(uuid.uuid4()),
            name="sched-test",
            max_turns=10,
            skill_ids=["test-skill"],
            model_provider="kimi",
            model_name="kimi-k2.5",
        )
        task = ScheduledTaskDB(
            id=str(uuid.uuid4()),
            name="task-sched",
            agent_id=preset.id,
            prompt="Run test",
            schedule_type="once",
            schedule_value="2099-01-01T00:00:00Z",
            context_mode="isolated",
            status="active",
            run_count=0,
        )
        run_log = TaskRunLogDB(
            id=generate_uuid(),
            task_id=task.id,
            status="running",
        )

        # Mock SyncSessionLocal to return objects without real DB
        mock_sync_session = MagicMock()
        mock_sync_session.get = MagicMock(side_effect=lambda model, id_: {
            (ScheduledTaskDB, task.id): task,
            (AgentPresetDB, preset.id): preset,
            (TaskRunLogDB, run_log.id): run_log,
        }.get((model, id_)))
        mock_sync_session.add = MagicMock()
        mock_sync_session.commit = MagicMock()
        mock_sync_session.close = MagicMock()

        # Setup mock agent — use a plain coroutine to avoid event loop affinity
        # (scheduler creates its own event loop via asyncio.new_event_loop())
        mock_instance = MagicMock()
        mock_instance.model_provider = "kimi"
        mock_instance.model = "kimi-k2.5"
        mock_instance.cleanup = MagicMock()

        mock_result = MockAgentResult()

        async def mock_run(*args, **kwargs):
            return mock_result

        mock_instance.run = mock_run
        MockSkillsAgent.return_value = mock_instance

        # Execute with mocked DB (SyncSessionLocal is lazy-imported from app.db.database)
        scheduler = TaskScheduler()
        with patch("app.db.database.SyncSessionLocal", return_value=mock_sync_session):
            scheduler._execute_task(task.id, run_log.id)

        # Verify agent was created with correct params from preset
        MockSkillsAgent.assert_called_once()
        call_kwargs = MockSkillsAgent.call_args[1]
        assert call_kwargs["max_turns"] == 10
        assert call_kwargs["allowed_skills"] == ["test-skill"]
        assert call_kwargs["model_provider"] == "kimi"
        assert call_kwargs["model"] == "kimi-k2.5"

        # Verify cleanup was called
        mock_instance.cleanup.assert_called_once()

        # Verify trace was added to session
        added_objects = [call.args[0] for call in mock_sync_session.add.call_args_list]
        trace_objects = [o for o in added_objects if isinstance(o, AgentTraceDB)]
        assert len(trace_objects) == 1
        trace = trace_objects[0]
        assert trace.model_provider == "kimi"
        assert trace.model == "kimi-k2.5"
        assert trace.request == "Run test"


# ---------------------------------------------------------------------------
# Integration: channel_manager uses shared service
# ---------------------------------------------------------------------------


class TestChannelManagerIntegration:
    """Verify that ChannelManager._run_agent uses the shared service layer."""

    @patch("app.agent.SkillsAgent")
    async def test_run_agent_creates_trace(self, MockSkillsAgent):
        """channel_manager._run_agent now creates a trace (the core bug fix)."""
        from app.services.channel_manager import ChannelManager

        # Setup mock agent
        mock_instance = MagicMock()
        mock_instance.model_provider = "kimi"
        mock_instance.model = "kimi-k2.5"
        mock_instance.cleanup = MagicMock()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.answer = "Channel response"
        mock_result.total_turns = 1
        mock_result.total_input_tokens = 100
        mock_result.total_output_tokens = 50
        mock_result.steps = []
        mock_result.llm_calls = []
        mock_result.skills_used = ["test-skill"]
        mock_result.error = None
        mock_result.output_files = []
        mock_result.final_messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Channel response"}]

        mock_instance.run = AsyncMock(return_value=mock_result)
        MockSkillsAgent.return_value = mock_instance

        # Create a preset
        preset = AgentPresetDB(
            id=str(uuid.uuid4()),
            name="channel-test",
            max_turns=20,
            skill_ids=["test-skill"],
            model_provider="kimi",
            model_name="kimi-k2.5",
        )

        session_id = f"channel-session-{uuid.uuid4().hex[:8]}"

        # Mock AsyncSessionLocal to capture the trace object (lazy-imported from app.db.database)
        mock_trace_session = AsyncMock()
        mock_trace_session.__aenter__ = AsyncMock(return_value=mock_trace_session)
        mock_trace_session.__aexit__ = AsyncMock(return_value=False)
        added_objects = []
        mock_trace_session.add = MagicMock(side_effect=lambda obj: added_objects.append(obj))
        mock_trace_session.commit = AsyncMock()

        # Execute — mock both AsyncSessionLocal and save_session_messages (lazy imports)
        with patch("app.db.database.AsyncSessionLocal", return_value=mock_trace_session), \
             patch("app.api.v1.sessions.save_session_messages", new_callable=AsyncMock):
            manager = ChannelManager.__new__(ChannelManager)
            manager._adapters = {}
            answer, output_files = await manager._run_agent(
                preset, "Hello from Feishu", conversation_history=None,
                session_id=session_id,
            )

        assert answer == "Channel response"

        # THE CORE BUG FIX: Verify trace was created
        trace_objects = [o for o in added_objects if isinstance(o, AgentTraceDB)]
        assert len(trace_objects) == 1
        trace = trace_objects[0]
        assert trace.request == "Hello from Feishu"
        assert trace.status == "completed"
        assert trace.success is True
        assert trace.answer == "Channel response"
        assert trace.model_provider == "kimi"
        assert trace.model == "kimi-k2.5"
        assert trace.session_id == session_id

    @patch("app.agent.SkillsAgent")
    async def test_run_agent_calls_cleanup(self, MockSkillsAgent):
        """channel_manager._run_agent calls agent.cleanup() in finally block."""
        from app.services.channel_manager import ChannelManager

        mock_instance = MagicMock()
        mock_instance.model_provider = "kimi"
        mock_instance.model = "kimi-k2.5"
        mock_instance.cleanup = MagicMock()
        mock_instance.run = AsyncMock(return_value=MagicMock(
            success=True, answer="OK", total_turns=1,
            total_input_tokens=0, total_output_tokens=0,
            steps=[], llm_calls=[], skills_used=[], error=None,
            output_files=[], final_messages=[],
        ))
        MockSkillsAgent.return_value = mock_instance

        preset = AgentPresetDB(
            id=str(uuid.uuid4()), name="cleanup-test",
            max_turns=5, model_provider="kimi", model_name="kimi-k2.5",
        )

        manager = ChannelManager.__new__(ChannelManager)
        manager._adapters = {}
        await manager._run_agent(preset, "Test cleanup")

        mock_instance.cleanup.assert_called_once()

    @patch("app.agent.SkillsAgent")
    async def test_run_agent_cleanup_on_error(self, MockSkillsAgent):
        """channel_manager._run_agent calls cleanup even when agent.run() raises."""
        from app.services.channel_manager import ChannelManager

        mock_instance = MagicMock()
        mock_instance.model_provider = "kimi"
        mock_instance.model = "kimi-k2.5"
        mock_instance.cleanup = MagicMock()
        mock_instance.run = AsyncMock(side_effect=RuntimeError("LLM failed"))
        MockSkillsAgent.return_value = mock_instance

        preset = AgentPresetDB(
            id=str(uuid.uuid4()), name="error-test",
            max_turns=5, model_provider="kimi", model_name="kimi-k2.5",
        )

        manager = ChannelManager.__new__(ChannelManager)
        manager._adapters = {}
        answer, _ = await manager._run_agent(preset, "Will fail")

        assert "Error:" in answer
        mock_instance.cleanup.assert_called_once()

    @patch("app.agent.SkillsAgent")
    async def test_run_agent_updates_session_via_save_session_messages(self, MockSkillsAgent):
        """channel_manager._run_agent updates session using save_session_messages (dual-store)."""
        from app.services.channel_manager import ChannelManager

        mock_instance = MagicMock()
        mock_instance.model_provider = "kimi"
        mock_instance.model = "kimi-k2.5"
        mock_instance.cleanup = MagicMock()
        mock_instance.run = AsyncMock(return_value=MagicMock(
            success=True, answer="Updated response", total_turns=1,
            total_input_tokens=50, total_output_tokens=25,
            steps=[], llm_calls=[], skills_used=[], error=None,
            output_files=[],
            final_messages=[
                {"role": "user", "content": "Update test"},
                {"role": "assistant", "content": "Updated response"},
            ],
        ))
        MockSkillsAgent.return_value = mock_instance

        preset = AgentPresetDB(
            id=str(uuid.uuid4()), name="session-update-test",
            max_turns=5, model_provider="kimi", model_name="kimi-k2.5",
        )

        session_id = f"session-update-{uuid.uuid4().hex[:8]}"

        # Mock AsyncSessionLocal for trace saving and save_session_messages for session update
        # Both are lazy-imported inside _run_agent from their source modules
        mock_trace_session = AsyncMock()
        mock_trace_session.__aenter__ = AsyncMock(return_value=mock_trace_session)
        mock_trace_session.__aexit__ = AsyncMock(return_value=False)
        mock_trace_session.add = MagicMock()
        mock_trace_session.commit = AsyncMock()

        with patch("app.db.database.AsyncSessionLocal", return_value=mock_trace_session), \
             patch("app.api.v1.sessions.save_session_messages", new_callable=AsyncMock) as mock_save:
            manager = ChannelManager.__new__(ChannelManager)
            manager._adapters = {}
            await manager._run_agent(
                preset, "Update test", session_id=session_id,
            )

            # Verify save_session_messages was called with correct arguments
            mock_save.assert_called_once_with(
                session_id,
                "Updated response",
                "Update test",
                final_messages=[
                    {"role": "user", "content": "Update test"},
                    {"role": "assistant", "content": "Updated response"},
                ],
            )

    @patch("app.agent.SkillsAgent")
    async def test_run_agent_pre_compress_called(self, MockSkillsAgent):
        """channel_manager._run_agent calls pre_compress_if_needed when history exists."""
        from app.services.channel_manager import ChannelManager

        mock_instance = MagicMock()
        mock_instance.model_provider = "kimi"
        mock_instance.model = "kimi-k2.5"
        mock_instance.cleanup = MagicMock()
        mock_instance.run = AsyncMock(return_value=MagicMock(
            success=True, answer="OK", total_turns=1,
            total_input_tokens=0, total_output_tokens=0,
            steps=[], llm_calls=[], skills_used=[], error=None,
            output_files=[], final_messages=[],
        ))
        MockSkillsAgent.return_value = mock_instance

        preset = AgentPresetDB(
            id=str(uuid.uuid4()), name="compress-test",
            max_turns=5, model_provider="kimi", model_name="kimi-k2.5",
        )

        manager = ChannelManager.__new__(ChannelManager)
        manager._adapters = {}

        history = [{"role": "user", "content": "Old message"}]
        with patch("app.api.v1.sessions.pre_compress_if_needed", new_callable=AsyncMock, return_value=history) as mock_compress:
            await manager._run_agent(
                preset, "New message", conversation_history=history,
            )
            mock_compress.assert_called_once_with(history, "kimi", "kimi-k2.5")
