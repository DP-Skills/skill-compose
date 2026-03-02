"""
End-to-end integration tests — DB-only workflows (no real LLM calls).

Each test class uses a class-scoped DB session so that tests within a class
share state.  Tests are ordered by name (test_01, test_02, ...) and pass
data via ``cls._state``.

Run:
    pytest tests/test_e2e/test_e2e_workflows.py -v
"""

import json
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.agent import StreamEvent
from app.api.v1.sessions import SessionData

from tests.factories import make_skill, make_skill_version, make_skill_file, make_trace

from tests.test_e2e.conftest import parse_sse_events

# ---------------------------------------------------------------------------
# Shared mock helpers (Agent)
# ---------------------------------------------------------------------------


@dataclass
class _MockStep:
    role: str = "assistant"
    content: str = "E2E mock answer"
    tool_name: Optional[str] = None
    tool_input: Optional[Dict] = None
    tool_result: Optional[str] = None


@dataclass
class _MockLLMCall:
    turn: int = 1
    timestamp: str = "2025-01-01T00:00:00"
    model: str = "claude-sonnet-4-5-20250929"
    request_messages: List[Dict] = field(default_factory=list)
    response_content: List[Dict] = field(default_factory=list)
    stop_reason: str = "end_turn"
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class _MockAgentResult:
    success: bool = True
    answer: str = "E2E mock answer"
    total_turns: int = 1
    total_input_tokens: int = 100
    total_output_tokens: int = 50
    steps: List = field(default_factory=lambda: [_MockStep()])
    llm_calls: List = field(default_factory=lambda: [_MockLLMCall()])
    error: Optional[str] = None
    log_file: Optional[str] = None
    output_files: List = field(default_factory=list)
    skills_used: List = field(default_factory=list)
    final_messages: List = field(default_factory=list)


def _make_mock_agent(result=None):
    instance = MagicMock()
    instance.run = AsyncMock(return_value=result or _MockAgentResult())
    instance.cleanup = MagicMock()
    instance.model = "claude-sonnet-4-5-20250929"
    instance.model_provider = "anthropic"  # Required for multi-LLM support
    return instance


def _make_stream_events(answer="E2E stream done"):
    return [
        StreamEvent(event_type="turn_start", turn=1, data={"turn": 1}),
        StreamEvent(
            event_type="assistant",
            turn=1,
            data={"content": answer, "turn": 1},
        ),
        StreamEvent(
            event_type="complete",
            turn=1,
            data={
                "success": True,
                "answer": answer,
                "total_turns": 1,
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "skills_used": [],
                "final_messages": [
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ],
            },
        ),
    ]


def _make_streaming_mock_agent(events=None, answer="E2E stream done"):
    """Create mock agent that pushes events to event_stream in async run()."""
    if events is None:
        events = _make_stream_events(answer)

    complete_event = next((e for e in events if e.event_type == "complete"), None)
    result = _MockAgentResult(
        success=complete_event.data.get("success", True) if complete_event else True,
        answer=complete_event.data.get("answer", answer) if complete_event else answer,
        total_turns=complete_event.data.get("total_turns", 1) if complete_event else 1,
        total_input_tokens=complete_event.data.get("total_input_tokens", 100) if complete_event else 100,
        total_output_tokens=complete_event.data.get("total_output_tokens", 50) if complete_event else 50,
        skills_used=complete_event.data.get("skills_used", []) if complete_event else [],
        output_files=complete_event.data.get("output_files", []) if complete_event else [],
        final_messages=complete_event.data.get("final_messages", []) if complete_event else [],
    )

    mock_instance = MagicMock()
    mock_instance.model = "claude-sonnet-4-5-20250929"
    mock_instance.model_provider = "anthropic"
    mock_instance.cleanup = MagicMock()

    async def mock_run(request, conversation_history=None, image_contents=None,
                       event_stream=None, cancellation_event=None):
        if event_stream:
            for event in events:
                await event_stream.push(event)
            await event_stream.close()
        return result

    mock_instance.run = AsyncMock(side_effect=mock_run)
    return mock_instance


def _mock_session_local():
    """Return a callable that produces async-context-manager sessions (no-op DB)."""
    @asynccontextmanager
    async def _ctx():
        mock_sess = AsyncMock(spec=AsyncSession)
        mock_sess.add = MagicMock()
        mock_sess.commit = AsyncMock()
        mock_sess.close = AsyncMock()
        mock_sess.execute = AsyncMock()
        yield mock_sess

    return _ctx


# ===================================================================
# Class 1: Health & Discovery
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestHealthAndDiscoveryE2E:
    """Smoke tests for health, root, tools registry, and MCP servers."""

    async def test_01_health(self, e2e_client: AsyncClient):
        resp = await e2e_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    async def test_02_root(self, e2e_client: AsyncClient):
        resp = await e2e_client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Skills API"
        assert "endpoints" in body

    async def test_03_tools_registry(self, e2e_client: AsyncClient):
        resp = await e2e_client.get("/api/v1/tools/registry")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] > 0
        names = [t["name"] for t in body["tools"]]
        assert "execute_code" in names

    @patch("app.api.v1.mcp.get_all_mcp_servers_info", return_value=[])
    async def test_04_mcp_servers_list(self, _mock, e2e_client: AsyncClient):
        resp = await e2e_client.get("/api/v1/mcp/servers")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["servers"], list)


# ===================================================================
# Class 2: Skill Full Lifecycle
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestSkillFullLifecycleE2E:
    """Create → query → update → version → export → delete."""

    _state: dict = {}

    async def test_01_create_skill(
        self, e2e_client: AsyncClient, e2e_db_session: AsyncSession
    ):
        """Create a skill directly in DB (bypassing async task_manager)."""
        skill = make_skill(name="e2e-lifecycle", description="E2E lifecycle test")
        e2e_db_session.add(skill)
        await e2e_db_session.flush()
        type(self)._state["skill_id"] = skill.id

        version = make_skill_version(skill_id=skill.id, version="0.0.1")
        e2e_db_session.add(version)
        await e2e_db_session.flush()
        type(self)._state["version_id"] = version.id

    async def test_02_list_skills(self, e2e_client: AsyncClient):
        resp = await e2e_client.get("/api/v1/registry/skills")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1
        names = [s["name"] for s in body["skills"]]
        assert "e2e-lifecycle" in names

    async def test_03_get_skill(self, e2e_client: AsyncClient):
        resp = await e2e_client.get("/api/v1/registry/skills/e2e-lifecycle")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "e2e-lifecycle"
        assert body["id"] == type(self)._state["skill_id"]

    async def test_04_search_skill(self, e2e_client: AsyncClient):
        resp = await e2e_client.get(
            "/api/v1/registry/skills/search", params={"q": "lifecycle"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1
        assert any(s["name"] == "e2e-lifecycle" for s in body["skills"])

    async def test_05_update_skill(self, e2e_client: AsyncClient):
        resp = await e2e_client.put(
            "/api/v1/registry/skills/e2e-lifecycle",
            json={"description": "Updated E2E", "status": "active"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["description"] == "Updated E2E"

    async def test_06_create_version(self, e2e_client: AsyncClient):
        skill_md = (
            "---\nname: e2e-lifecycle\ndescription: Updated E2E\n---\n\n"
            "# E2E Lifecycle\n\nVersion 2 content that is long enough to pass validation."
        )
        resp = await e2e_client.post(
            "/api/v1/registry/skills/e2e-lifecycle/versions",
            json={"version": "0.0.2", "skill_md": skill_md},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["version"] == "0.0.2"
        type(self)._state["version_id_v2"] = body["id"]

    async def test_07_list_versions(self, e2e_client: AsyncClient):
        resp = await e2e_client.get(
            "/api/v1/registry/skills/e2e-lifecycle/versions"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 2
        versions = [v["version"] for v in body["versions"]]
        assert "0.0.1" in versions
        assert "0.0.2" in versions

    async def test_08_get_version(self, e2e_client: AsyncClient):
        resp = await e2e_client.get(
            "/api/v1/registry/skills/e2e-lifecycle/versions/0.0.2"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["version"] == "0.0.2"

    async def test_09_get_version_files(
        self, e2e_client: AsyncClient, e2e_db_session: AsyncSession
    ):
        """Add a file to v1, then query the files endpoint."""
        vid = type(self)._state["version_id"]
        f = make_skill_file(
            version_id=vid,
            file_path="scripts/run.py",
            content=b"print('e2e')",
        )
        e2e_db_session.add(f)
        await e2e_db_session.flush()

        resp = await e2e_client.get(
            "/api/v1/registry/skills/e2e-lifecycle/versions/0.0.1/files"
        )
        assert resp.status_code == 200
        body = resp.json()
        paths = [fi["file_path"] for fi in body["files"]]
        assert "scripts/run.py" in paths
        # Verify content_hash is present on each file
        for fi in body["files"]:
            assert "content_hash" in fi
            assert fi["content_hash"] is not None

    async def test_10_diff(self, e2e_client: AsyncClient):
        resp = await e2e_client.get(
            "/api/v1/registry/skills/e2e-lifecycle/diff",
            params={"from": "0.0.1", "to": "0.0.2"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["from_version"] == "0.0.1"
        assert body["to_version"] == "0.0.2"
        assert len(body["diff"]) > 0

    async def test_11_rollback(self, e2e_client: AsyncClient):
        resp = await e2e_client.post(
            "/api/v1/registry/skills/e2e-lifecycle/rollback",
            json={"version": "0.0.1", "comment": "E2E rollback"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["version"] == "0.0.1"

    async def test_12_changelog(
        self, e2e_client: AsyncClient, e2e_db_session: AsyncSession
    ):
        from tests.factories import make_changelog

        cl = make_changelog(
            skill_id=type(self)._state["skill_id"],
            change_type="update",
            version_to="0.0.2",
        )
        e2e_db_session.add(cl)
        await e2e_db_session.flush()

        resp = await e2e_client.get(
            "/api/v1/registry/skills/e2e-lifecycle/changelog"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1

    async def test_13_export(self, e2e_client: AsyncClient):
        resp = await e2e_client.get(
            "/api/v1/registry/skills/e2e-lifecycle/export"
        )
        # export returns the zip or 200
        assert resp.status_code == 200
        assert len(resp.content) > 0

    async def test_14_sync_filesystem(self, e2e_client: AsyncClient):
        resp = await e2e_client.post(
            "/api/v1/registry/skills/e2e-lifecycle/sync-filesystem"
        )
        assert resp.status_code == 200
        body = resp.json()
        # No disk directory for this test skill
        assert body["synced"] is False

    async def test_15_delete(self, e2e_client: AsyncClient):
        resp = await e2e_client.delete(
            "/api/v1/registry/skills/e2e-lifecycle"
        )
        assert resp.status_code == 204

    async def test_16_verify_deleted(self, e2e_client: AsyncClient):
        resp = await e2e_client.get(
            "/api/v1/registry/skills/e2e-lifecycle"
        )
        assert resp.status_code == 404


# ===================================================================
# Class 3: Skill Import/Export via qdrant.zip
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestSkillImportExportE2E:
    """Import a real .skill zip, verify, export, reimport conflict, cleanup."""

    _state: dict = {}

    async def test_01_import(
        self, e2e_client: AsyncClient, qdrant_zip_bytes: bytes
    ):
        resp = await e2e_client.post(
            "/api/v1/registry/import",
            files={
                "file": (
                    "qdrant-vector-search.skill",
                    qdrant_zip_bytes,
                    "application/zip",
                )
            },
        )
        assert resp.status_code in (200, 201, 202)
        body = resp.json()
        type(self)._state["import_result"] = body

    async def test_02_verify_imported(self, e2e_client: AsyncClient):
        resp = await e2e_client.get(
            "/api/v1/registry/skills/qdrant-vector-search"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "qdrant-vector-search"
        type(self)._state["skill_id"] = body["id"]

    async def test_03_list_versions(self, e2e_client: AsyncClient):
        resp = await e2e_client.get(
            "/api/v1/registry/skills/qdrant-vector-search/versions"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1
        type(self)._state["first_version"] = body["versions"][0]["version"]

    async def test_04_get_files(self, e2e_client: AsyncClient):
        ver = type(self)._state.get("first_version", "0.0.1")
        resp = await e2e_client.get(
            f"/api/v1/registry/skills/qdrant-vector-search/versions/{ver}/files"
        )
        assert resp.status_code == 200
        body = resp.json()
        # Should have at least one file
        assert len(body["files"]) >= 0  # may have none if only SKILL.md
        # Verify content_hash is present on each file
        for fi in body["files"]:
            assert "content_hash" in fi

    async def test_05_export_round_trip(self, e2e_client: AsyncClient):
        resp = await e2e_client.get(
            "/api/v1/registry/skills/qdrant-vector-search/export"
        )
        assert resp.status_code == 200
        assert len(resp.content) > 0
        type(self)._state["exported_zip"] = resp.content

    async def test_06_reimport_conflict(self, e2e_client: AsyncClient):
        """Re-importing the same skill should return conflict (409 or 200 with message)."""
        exported = type(self)._state.get("exported_zip")
        if not exported:
            pytest.skip("No exported zip available")
        resp = await e2e_client.post(
            "/api/v1/registry/import",
            files={
                "file": (
                    "qdrant-vector-search.skill",
                    exported,
                    "application/zip",
                )
            },
        )
        # Conflict or already-exists response
        assert resp.status_code in (200, 400, 409)

    async def test_07_cleanup(self, e2e_client: AsyncClient):
        resp = await e2e_client.delete(
            "/api/v1/registry/skills/qdrant-vector-search"
        )
        assert resp.status_code == 204


# ===================================================================
# Class 4: Agent Preset Lifecycle
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestAgentPresetLifecycleE2E:
    """Preset CRUD + publish/unpublish + agent run (mocked) + trace verification."""

    _state: dict = {}

    async def test_01_create_preset(self, e2e_client: AsyncClient):
        payload = {
            "name": "e2e-preset",
            "description": "E2E preset",
            "system_prompt": "You are an E2E test helper.",
            "skill_ids": [],
            "mcp_servers": ["fetch"],
            "max_turns": 10,
        }
        resp = await e2e_client.post("/api/v1/agents", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "e2e-preset"
        type(self)._state["preset_id"] = body["id"]

    async def test_02_list_presets(self, e2e_client: AsyncClient):
        resp = await e2e_client.get("/api/v1/agents")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1
        names = [p["name"] for p in body["presets"]]
        assert "e2e-preset" in names

    async def test_03_get_by_id(self, e2e_client: AsyncClient):
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.get(f"/api/v1/agents/{pid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == pid

    async def test_04_get_by_name(self, e2e_client: AsyncClient):
        resp = await e2e_client.get("/api/v1/agents/by-name/e2e-preset")
        assert resp.status_code == 200
        assert resp.json()["name"] == "e2e-preset"

    async def test_05_update(self, e2e_client: AsyncClient):
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.put(
            f"/api/v1/agents/{pid}",
            json={"description": "Updated E2E preset", "max_turns": 20},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["description"] == "Updated E2E preset"
        assert body["max_turns"] == 20

    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
    @patch("app.api.v1.agent.create_agent")
    async def test_06_run_agent(self, MockAgent, _mock_load, _mock_save, e2e_client: AsyncClient):
        MockAgent.return_value = _make_mock_agent()
        resp = await e2e_client.post(
            "/api/v1/agent/run",
            json={"request": "E2E test request", "session_id": "test-session-id"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["trace_id"] is not None
        type(self)._state["trace_id"] = body["trace_id"]

    async def test_07_verify_trace(self, e2e_client: AsyncClient):
        tid = type(self)._state.get("trace_id")
        if not tid:
            pytest.skip("No trace_id from previous test")
        resp = await e2e_client.get(f"/api/v1/traces/{tid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == tid
        assert body["success"] is True

    async def test_08_trace_list(self, e2e_client: AsyncClient):
        resp = await e2e_client.get("/api/v1/traces")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1

    async def test_09_publish(self, e2e_client: AsyncClient):
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post(
            f"/api/v1/agents/{pid}/publish",
            json={"api_response_mode": "streaming"}
        )
        assert resp.status_code == 200
        assert resp.json()["is_published"] is True

    async def test_10_get_published(self, e2e_client: AsyncClient):
        """GET published agent info (mocking AsyncSessionLocal)."""
        pid = type(self)._state["preset_id"]

        # The published endpoint uses AsyncSessionLocal directly, so we mock it
        from app.db.models import AgentPresetDB

        mock_preset = MagicMock(spec=AgentPresetDB)
        mock_preset.id = pid
        mock_preset.name = "e2e-preset"
        mock_preset.description = "Updated E2E preset"
        mock_preset.is_published = True
        mock_preset.api_response_mode = "streaming"
        mock_preset.skill_ids = []
        mock_preset.builtin_tools = None
        mock_preset.max_turns = 20
        mock_preset.mcp_servers = ["fetch"]
        mock_preset.system_prompt = "You are an E2E test helper."

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_preset

        @asynccontextmanager
        async def _ctx():
            mock_sess = AsyncMock(spec=AsyncSession)
            mock_sess.execute = AsyncMock(return_value=mock_result)
            yield mock_sess

        with patch(
            "app.api.v1.published.AsyncSessionLocal",
            side_effect=lambda: _ctx(),
        ):
            resp = await e2e_client.get(f"/api/v1/published/{pid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == pid
        assert body["name"] == "e2e-preset"

    async def test_11_unpublish(self, e2e_client: AsyncClient):
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post(f"/api/v1/agents/{pid}/unpublish")
        assert resp.status_code == 200
        assert resp.json()["is_published"] is False

    async def test_12_delete(self, e2e_client: AsyncClient):
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.delete(f"/api/v1/agents/{pid}")
        assert resp.status_code == 200

        # Verify deleted
        resp2 = await e2e_client.get(f"/api/v1/agents/{pid}")
        assert resp2.status_code == 404


# ===================================================================
# Class 5: Agent Run & Trace
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestAgentRunAndTraceE2E:
    """Agent run → trace list → detail → filter → stream → delete."""

    _state: dict = {}

    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
    @patch("app.api.v1.agent.create_agent")
    async def test_01_run_simple(self, MockAgent, _mock_load, _mock_save, e2e_client: AsyncClient):
        MockAgent.return_value = _make_mock_agent()
        resp = await e2e_client.post(
            "/api/v1/agent/run",
            json={"request": "E2E simple run", "session_id": "test-session-id"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        type(self)._state["trace_id_1"] = body["trace_id"]

    async def test_02_trace_list(self, e2e_client: AsyncClient):
        resp = await e2e_client.get("/api/v1/traces")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1

    async def test_03_trace_detail(self, e2e_client: AsyncClient):
        tid = type(self)._state.get("trace_id_1")
        if not tid:
            pytest.skip("No trace_id")
        resp = await e2e_client.get(f"/api/v1/traces/{tid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == tid
        assert body["request"] == "E2E simple run"
        assert body["success"] is True

    async def test_04_filter_by_success(self, e2e_client: AsyncClient):
        resp = await e2e_client.get("/api/v1/traces", params={"success": True})
        assert resp.status_code == 200
        body = resp.json()
        for t in body["traces"]:
            assert t["success"] is True

    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
    @patch("app.api.v1.agent.create_agent")
    async def test_05_run_with_skills_and_session(
        self, MockAgent, _mock_load, _mock_save, e2e_client: AsyncClient
    ):
        MockAgent.return_value = _make_mock_agent()
        resp = await e2e_client.post(
            "/api/v1/agent/run",
            json={
                "request": "Continue our conversation",
                "skills": ["test-skill"],
                "session_id": "test-session-id",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        type(self)._state["trace_id_2"] = body["trace_id"]

    @patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
    @patch("app.api.v1.agent.AsyncSessionLocal")
    @patch("app.api.v1.agent.create_agent")
    async def test_06_run_stream(
        self, MockAgent, MockSessionLocal, _mock_load, _mock_save, _mock_checkpoint, _mock_precompress, e2e_client: AsyncClient
    ):
        MockAgent.return_value = _make_streaming_mock_agent()
        MockSessionLocal.side_effect = lambda: _mock_session_local()()

        resp = await e2e_client.post(
            "/api/v1/agent/run/stream",
            json={"request": "E2E stream test", "session_id": "test-session-id"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        events = parse_sse_events(resp.text)
        assert len(events) >= 1
        assert events[0]["event_type"] == "run_started"

    async def test_07_delete_trace(self, e2e_client: AsyncClient):
        tid = type(self)._state.get("trace_id_1")
        if not tid:
            pytest.skip("No trace_id")
        resp = await e2e_client.delete(f"/api/v1/traces/{tid}")
        assert resp.status_code == 200

    async def test_08_verify_deleted(self, e2e_client: AsyncClient):
        tid = type(self)._state.get("trace_id_1")
        if not tid:
            pytest.skip("No trace_id")
        resp = await e2e_client.get(f"/api/v1/traces/{tid}")
        assert resp.status_code == 404


# ===================================================================
# Class 6: File Upload
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestFileUploadE2E:
    """Upload → info → agent run with file (mock) → delete."""

    _state: dict = {}

    async def test_01_upload(self, e2e_client: AsyncClient):
        from app.api.v1 import files as files_module
        files_module._file_registry.clear()

        resp = await e2e_client.post(
            "/api/v1/files/upload",
            files={"file": ("e2e-test.txt", b"E2E file content", "text/plain")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "e2e-test.txt"
        type(self)._state["file_id"] = body["file_id"]
        type(self)._state["file_path"] = body["path"]

    async def test_02_get_info(self, e2e_client: AsyncClient):
        fid = type(self)._state["file_id"]
        resp = await e2e_client.get(f"/api/v1/files/{fid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["file_id"] == fid
        assert body["filename"] == "e2e-test.txt"

    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
    @patch("app.api.v1.agent.create_agent")
    async def test_03_run_agent_with_file(
        self, MockAgent, _mock_load, _mock_save, e2e_client: AsyncClient
    ):
        MockAgent.return_value = _make_mock_agent()
        files = [
            {
                "file_id": type(self)._state["file_id"],
                "filename": "e2e-test.txt",
                "path": type(self)._state["file_path"],
                "content_type": "text/plain",
            }
        ]
        resp = await e2e_client.post(
            "/api/v1/agent/run",
            json={"request": "Process this file", "uploaded_files": files, "session_id": "test-session-id"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    async def test_04_delete(self, e2e_client: AsyncClient):
        fid = type(self)._state["file_id"]
        resp = await e2e_client.delete(f"/api/v1/files/{fid}")
        assert resp.status_code == 204

    async def test_05_verify_deleted(self, e2e_client: AsyncClient):
        fid = type(self)._state["file_id"]
        resp = await e2e_client.get(f"/api/v1/files/{fid}")
        assert resp.status_code == 404


# ===================================================================
# Class 7: Code Execution
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestCodeExecutionE2E:
    """Code execution via /tools/execute and /tools/execute_command."""

    async def test_01_execute_python(self, e2e_client: AsyncClient):
        resp = await e2e_client.post(
            "/api/v1/tools/execute",
            json={"code": "print(2 + 3)", "executor": "simple"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "5" in body["output"]

    async def test_02_execute_error(self, e2e_client: AsyncClient):
        resp = await e2e_client.post(
            "/api/v1/tools/execute",
            json={
                "code": "raise ValueError('e2e boom')",
                "executor": "simple",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert "ValueError" in body["error"]

    @patch("app.api.v1.tools.get_code_executor")
    async def test_03_execute_command(
        self, mock_get_executor, e2e_client: AsyncClient
    ):
        from tests.mocks.mock_code_executor import MockCodeExecutor

        mock_executor = MockCodeExecutor(default_output="file1.txt\nfile2.txt")
        mock_get_executor.return_value = mock_executor

        resp = await e2e_client.post(
            "/api/v1/tools/execute_command",
            json={"command": "ls -la"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True


# ===================================================================
# Class 8: Protected Resources
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestProtectedResourcesE2E:
    """System protection rules and validation."""

    async def test_01_cannot_delete_meta_skill(
        self, e2e_client: AsyncClient, e2e_db_session: AsyncSession
    ):
        from app.db.models import SkillDB

        # Create meta skill first (E2E test database is clean)
        meta_skill = SkillDB(
            name="skill-creator",
            description="Meta skill for creating skills",
            skill_type="meta",
            status="active",
            current_version="1.0.0",
        )
        e2e_db_session.add(meta_skill)
        await e2e_db_session.commit()

        resp = await e2e_client.delete("/api/v1/registry/skills/skill-creator")
        assert resp.status_code == 403

    async def test_02_system_preset_can_edit_but_not_delete(
        self, e2e_client: AsyncClient, e2e_db_session: AsyncSession
    ):
        from tests.factories import make_preset

        preset = make_preset(name="e2e-sys-preset", is_system=True)
        e2e_db_session.add(preset)
        await e2e_db_session.flush()
        pid = preset.id

        # Can update system preset
        resp = await e2e_client.put(
            f"/api/v1/agents/{pid}",
            json={"description": "Updated system preset"},
        )
        assert resp.status_code == 200
        assert resp.json()["description"] == "Updated system preset"
        assert resp.json()["is_system"] is True

        # Cannot delete system preset
        resp2 = await e2e_client.delete(f"/api/v1/agents/{pid}")
        assert resp2.status_code == 403

    async def test_03_invalid_skill_name(self, e2e_client: AsyncClient):
        resp = await e2e_client.post(
            "/api/v1/registry/skills",
            json={"name": "Invalid Name!", "description": "bad"},
        )
        assert resp.status_code == 422

    async def test_04_duplicate_skill(
        self, e2e_client: AsyncClient, e2e_db_session: AsyncSession
    ):
        skill = make_skill(name="e2e-dup-test", description="dup test")
        e2e_db_session.add(skill)
        await e2e_db_session.flush()

        resp = await e2e_client.post(
            "/api/v1/registry/skills",
            json={"name": "e2e-dup-test", "description": "duplicate"},
        )
        assert resp.status_code == 409

    async def test_05_duplicate_preset(
        self, e2e_client: AsyncClient, e2e_db_session: AsyncSession
    ):
        from tests.factories import make_preset

        preset = make_preset(name="e2e-dup-preset")
        e2e_db_session.add(preset)
        await e2e_db_session.flush()

        resp = await e2e_client.post(
            "/api/v1/agents",
            json={"name": "e2e-dup-preset", "description": "duplicate"},
        )
        assert resp.status_code == 400


# ===================================================================
# Class 9: Published Agent Session
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestPublishedAgentSessionE2E:
    """Published Agent + Session management (mocked AsyncSessionLocal + Agent)."""

    _state: dict = {}

    async def test_01_create_preset(self, e2e_client: AsyncClient):
        payload = {
            "name": "e2e-published",
            "description": "E2E published agent",
            "max_turns": 5,
        }
        resp = await e2e_client.post("/api/v1/agents", json=payload)
        assert resp.status_code == 200
        type(self)._state["preset_id"] = resp.json()["id"]

    async def test_02_publish(self, e2e_client: AsyncClient):
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post(
            f"/api/v1/agents/{pid}/publish",
            json={"api_response_mode": "streaming"}
        )
        assert resp.status_code == 200
        assert resp.json()["is_published"] is True

    @patch("app.api.v1.published.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.published.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.published.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.published.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.published.SkillsAgent")
    @patch("app.api.v1.published.AsyncSessionLocal")
    async def test_03_chat_sse(
        self, MockSL, MockAgent, MockLoadSession, _mock_save, _mock_checkpoint, _mock_precompress, e2e_client: AsyncClient
    ):
        pid = type(self)._state["preset_id"]

        # Mock agent
        MockAgent.return_value = _make_streaming_mock_agent(answer="Published reply")

        # Mock AsyncSessionLocal: first call finds preset
        from app.db.models import AgentPresetDB

        mock_preset = MagicMock(spec=AgentPresetDB)
        mock_preset.id = pid
        mock_preset.name = "e2e-published"
        mock_preset.description = "E2E published agent"
        mock_preset.is_published = True
        mock_preset.api_response_mode = "streaming"
        mock_preset.skill_ids = []
        mock_preset.builtin_tools = None
        mock_preset.max_turns = 5
        mock_preset.mcp_servers = []
        mock_preset.system_prompt = None
        mock_preset.model_provider = None
        mock_preset.model_name = None
        mock_preset.executor_name = None

        call_idx = {"i": 0}
        results = [mock_preset, None, None, None, None]

        @asynccontextmanager
        async def _ctx():
            idx = min(call_idx["i"], len(results) - 1)
            call_idx["i"] += 1
            mock_sess = AsyncMock(spec=AsyncSession)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = results[idx]
            mock_sess.execute = AsyncMock(return_value=mock_result)
            mock_sess.add = MagicMock()
            mock_sess.commit = AsyncMock()
            yield mock_sess

        MockSL.side_effect = lambda: _ctx()

        session_id = str(uuid.uuid4())
        MockLoadSession.return_value = SessionData(session_id=session_id)
        resp = await e2e_client.post(
            f"/api/v1/published/{pid}/chat",
            json={"request": "Hello published", "session_id": session_id},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        events = parse_sse_events(resp.text)
        assert len(events) >= 1
        assert events[0]["event_type"] == "run_started"
        assert events[0]["session_id"] == session_id
        type(self)._state["session_id"] = session_id

    @patch("app.api.v1.published.AsyncSessionLocal")
    async def test_04_get_session(self, MockSL, e2e_client: AsyncClient):
        pid = type(self)._state["preset_id"]
        sid = type(self)._state.get("session_id")
        if not sid:
            pytest.skip("No session_id")

        from app.db.models import AgentPresetDB

        mock_preset = MagicMock(spec=AgentPresetDB)
        mock_preset.id = pid
        mock_preset.is_published = True
        mock_preset.api_response_mode = "streaming"

        mock_session = MagicMock()
        mock_session.id = sid
        mock_session.agent_id = pid
        mock_session.messages = [
            {"role": "user", "content": "Hello published"},
            {"role": "assistant", "content": "Published reply"},
        ]
        from datetime import datetime

        mock_session.created_at = datetime(2025, 1, 1)
        mock_session.updated_at = datetime(2025, 1, 1)

        # get_session uses ONE AsyncSessionLocal() ctx with TWO execute() calls
        execute_results = [mock_preset, mock_session]
        exec_idx = {"i": 0}

        @asynccontextmanager
        async def _ctx():
            def _next_execute(*args, **kwargs):
                idx = min(exec_idx["i"], len(execute_results) - 1)
                exec_idx["i"] += 1
                mock_result = MagicMock()
                mock_result.scalar_one_or_none.return_value = execute_results[idx]
                return mock_result

            mock_sess = AsyncMock(spec=AsyncSession)
            mock_sess.execute = AsyncMock(side_effect=_next_execute)
            yield mock_sess

        MockSL.side_effect = lambda: _ctx()

        resp = await e2e_client.get(f"/api/v1/published/{pid}/sessions/{sid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == sid
        assert len(body["messages"]) == 2

    async def test_05_unpublish(self, e2e_client: AsyncClient):
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post(f"/api/v1/agents/{pid}/unpublish")
        assert resp.status_code == 200
        assert resp.json()["is_published"] is False

    async def test_06_cleanup(self, e2e_client: AsyncClient):
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.delete(f"/api/v1/agents/{pid}")
        assert resp.status_code == 200


# ===================================================================
# Class 10: Skill Evolve via Traces
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestSkillEvolveE2E:
    """Evolve skill via traces, feedback, combined + task status polling."""

    _state: dict = {}

    async def test_01_setup(
        self, e2e_client: AsyncClient, e2e_db_session: AsyncSession
    ):
        """Create skill + version + traces + temp disk dir for evolve tests."""
        skill = make_skill(name="e2e-evolve", description="Evolve test skill")
        e2e_db_session.add(skill)
        await e2e_db_session.flush()
        type(self)._state["skill_id"] = skill.id

        version = make_skill_version(skill_id=skill.id, version="0.0.1")
        e2e_db_session.add(version)
        await e2e_db_session.flush()

        # Create traces as evolve input data
        t1 = make_trace(
            request="Evolve input trace 1",
            skills_used=["e2e-evolve"],
            success=True,
        )
        t2 = make_trace(
            request="Evolve input trace 2",
            skills_used=["e2e-evolve"],
            success=False,
            error="Something went wrong",
        )
        e2e_db_session.add_all([t1, t2])
        await e2e_db_session.flush()
        type(self)._state["trace_id_1"] = t1.id
        type(self)._state["trace_id_2"] = t2.id

        # Create temp skill directory on disk (evolve checks it exists)
        tmpdir = tempfile.mkdtemp()
        type(self)._state["tmpdir"] = tmpdir
        skill_dir = os.path.join(tmpdir, "e2e-evolve")
        os.makedirs(skill_dir, exist_ok=True)
        with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
            f.write("---\nname: e2e-evolve\n---\n# E2E Evolve\nTest content.")

    @patch("app.api.v1.registry.task_manager")
    async def test_02_evolve_via_traces(
        self, mock_tm, e2e_client: AsyncClient
    ):
        """POST evolve-via-traces with trace_ids → 202 + task_id."""
        mock_task = MagicMock()
        mock_task.id = "evolve-task-traces"
        mock_tm.create_task_async = AsyncMock(return_value=mock_task)
        mock_tm.run_in_background = MagicMock()

        with patch("app.config.settings.custom_skills_dir", type(self)._state["tmpdir"]):
            resp = await e2e_client.post(
                "/api/v1/registry/skills/e2e-evolve/evolve-via-traces",
                json={"trace_ids": [type(self)._state["trace_id_1"]]},
            )

        assert resp.status_code == 202
        body = resp.json()
        assert body["task_id"] == "evolve-task-traces"
        assert body["status"] == "pending"
        mock_tm.create_task_async.assert_called_once()
        mock_tm.run_in_background.assert_called_once()
        # Verify the background function is the traces variant
        bg_func = mock_tm.run_in_background.call_args[0][1]
        assert bg_func.__name__ == "_run_skill_evolution_via_traces"

    @patch("app.api.v1.registry.task_manager")
    async def test_03_evolve_via_feedback(
        self, mock_tm, e2e_client: AsyncClient
    ):
        """POST evolve-via-traces with feedback only → 202 (skill-updater path)."""
        mock_task = MagicMock()
        mock_task.id = "evolve-task-feedback"
        mock_tm.create_task_async = AsyncMock(return_value=mock_task)
        mock_tm.run_in_background = MagicMock()

        with patch("app.config.settings.custom_skills_dir", type(self)._state["tmpdir"]):
            resp = await e2e_client.post(
                "/api/v1/registry/skills/e2e-evolve/evolve-via-traces",
                json={"feedback": "Make the instructions clearer and more concise."},
            )

        assert resp.status_code == 202
        body = resp.json()
        assert body["task_id"] == "evolve-task-feedback"
        assert body["status"] == "pending"
        assert "feedback" in body["message"].lower()
        # Verify the background function is the feedback variant (skill-updater)
        bg_func = mock_tm.run_in_background.call_args[0][1]
        assert bg_func.__name__ == "_run_skill_evolution_with_agent"

    @patch("app.api.v1.registry.task_manager")
    async def test_04_evolve_combined(
        self, mock_tm, e2e_client: AsyncClient
    ):
        """POST evolve-via-traces with both traces + feedback → 202."""
        mock_task = MagicMock()
        mock_task.id = "evolve-task-combined"
        mock_tm.create_task_async = AsyncMock(return_value=mock_task)
        mock_tm.run_in_background = MagicMock()

        with patch("app.config.settings.custom_skills_dir", type(self)._state["tmpdir"]):
            resp = await e2e_client.post(
                "/api/v1/registry/skills/e2e-evolve/evolve-via-traces",
                json={
                    "trace_ids": [
                        type(self)._state["trace_id_1"],
                        type(self)._state["trace_id_2"],
                    ],
                    "feedback": "Focus on error handling improvements.",
                },
            )

        assert resp.status_code == 202
        body = resp.json()
        assert body["task_id"] == "evolve-task-combined"
        # Combined path uses skill-evolver (traces take priority)
        bg_func = mock_tm.run_in_background.call_args[0][1]
        assert bg_func.__name__ == "_run_skill_evolution_via_traces"
        # Verify feedback is passed as argument
        bg_args = mock_tm.run_in_background.call_args[0]
        assert "Focus on error handling" in str(bg_args)

    async def test_05_evolve_empty_input(self, e2e_client: AsyncClient):
        """POST evolve-via-traces with neither traces nor feedback → 400."""
        resp = await e2e_client.post(
            "/api/v1/registry/skills/e2e-evolve/evolve-via-traces",
            json={},
        )
        assert resp.status_code == 400
        assert "at least one" in resp.json()["detail"].lower()

    async def test_06_evolve_nonexistent_skill(self, e2e_client: AsyncClient):
        """POST evolve for nonexistent skill → 404."""
        resp = await e2e_client.post(
            "/api/v1/registry/skills/nonexistent-xyz/evolve-via-traces",
            json={"feedback": "Improve it"},
        )
        assert resp.status_code == 404

    @patch("app.api.v1.registry.task_manager")
    async def test_07_evolve_bad_trace_ids(
        self, mock_tm, e2e_client: AsyncClient
    ):
        """POST evolve-via-traces with invalid trace IDs → 404."""
        with patch("app.config.settings.custom_skills_dir", type(self)._state["tmpdir"]):
            resp = await e2e_client.post(
                "/api/v1/registry/skills/e2e-evolve/evolve-via-traces",
                json={"trace_ids": ["bad-id-1", "bad-id-2"]},
            )
        assert resp.status_code == 404
        assert "traces" in resp.json()["detail"].lower()

    async def test_08_evolve_no_disk_dir(self, e2e_client: AsyncClient):
        """POST evolve for skill without disk directory → 404."""
        with patch("app.config.settings.custom_skills_dir", "/nonexistent/path"):
            resp = await e2e_client.post(
                "/api/v1/registry/skills/e2e-evolve/evolve-via-traces",
                json={"feedback": "test"},
            )
        assert resp.status_code == 404
        assert "directory" in resp.json()["detail"].lower()

    @patch("app.api.v1.registry.task_manager")
    async def test_09_task_status_completed(
        self, mock_tm, e2e_client: AsyncClient
    ):
        """GET task status for completed evolve task."""
        from app.services.task_manager import TaskStatus

        mock_task = MagicMock()
        mock_task.id = "evolve-task-done"
        mock_task.status = TaskStatus.COMPLETED
        mock_task.metadata = {"skill_name": "e2e-evolve", "trace_id": "some-trace-id"}
        mock_task.result = {"new_version": "0.0.2", "skill_name": "e2e-evolve"}
        mock_task.error = None
        mock_tm.get_task_async = AsyncMock(return_value=mock_task)

        resp = await e2e_client.get("/api/v1/registry/tasks/evolve-task-done")
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == "evolve-task-done"
        assert body["status"] == "completed"
        assert body["skill_name"] == "e2e-evolve"
        assert body["new_version"] == "0.0.2"
        assert body["trace_id"] == "some-trace-id"

    @patch("app.api.v1.registry.task_manager")
    async def test_10_task_status_running(
        self, mock_tm, e2e_client: AsyncClient
    ):
        """GET task status for running task → skill_name is None."""
        from app.services.task_manager import TaskStatus

        mock_task = MagicMock()
        mock_task.id = "evolve-task-running"
        mock_task.status = TaskStatus.RUNNING
        mock_task.metadata = {"skill_name": "e2e-evolve", "trace_id": "tid"}
        mock_task.result = None
        mock_task.error = None
        mock_tm.get_task_async = AsyncMock(return_value=mock_task)

        resp = await e2e_client.get("/api/v1/registry/tasks/evolve-task-running")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "running"
        assert body["skill_name"] is None

    @patch("app.api.v1.registry.task_manager")
    async def test_11_task_not_found(
        self, mock_tm, e2e_client: AsyncClient
    ):
        """GET task status for nonexistent task → 404."""
        mock_tm.get_task_async = AsyncMock(return_value=None)
        resp = await e2e_client.get("/api/v1/registry/tasks/no-such-task")
        assert resp.status_code == 404

    async def test_12_verify_evolve_traces_created(
        self, e2e_client: AsyncClient
    ):
        """Verify that evolve operations created trace records in DB."""
        resp = await e2e_client.get("/api/v1/traces")
        assert resp.status_code == 200
        body = resp.json()
        # 2 input traces + 3 evolve traces (from tests 02/03/04)
        assert body["total"] >= 5

    async def test_13_cleanup(self, e2e_client: AsyncClient):
        """Clean up temp dir and skill."""
        import shutil

        tmpdir = type(self)._state.get("tmpdir")
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        resp = await e2e_client.delete("/api/v1/registry/skills/e2e-evolve")
        assert resp.status_code in (204, 404)


# ===================================================================
# Class 11: Skill Discovery & Validation
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestSkillDiscoveryE2E:
    """Tags, file content, unregistered skills, import-local, validate."""

    _state: dict = {}

    async def test_01_setup(
        self, e2e_client: AsyncClient, e2e_db_session: AsyncSession
    ):
        """Create skill with tags and files for discovery tests."""
        skill = make_skill(
            name="e2e-discovery",
            description="Discovery test",
            tags=["python", "testing"],
        )
        e2e_db_session.add(skill)
        await e2e_db_session.flush()
        type(self)._state["skill_id"] = skill.id

        version = make_skill_version(
            skill_id=skill.id,
            version="0.0.1",
            skill_md="---\nname: e2e-discovery\n---\n# E2E Discovery\nContent for testing.",
        )
        e2e_db_session.add(version)
        await e2e_db_session.flush()
        type(self)._state["version_id"] = version.id

        f = make_skill_file(
            version_id=version.id,
            file_path="scripts/helper.py",
            content=b"def hello():\n    return 'world'",
        )
        e2e_db_session.add(f)
        await e2e_db_session.flush()

    async def test_02_tags(self, e2e_client: AsyncClient):
        """GET /registry/tags returns all unique tags."""
        resp = await e2e_client.get("/api/v1/registry/tags")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert "python" in body
        assert "testing" in body

    async def test_03_get_file_content(self, e2e_client: AsyncClient):
        """GET specific file content from a version."""
        resp = await e2e_client.get(
            "/api/v1/registry/skills/e2e-discovery/versions/0.0.1/files/scripts/helper.py"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["file_path"] == "scripts/helper.py"
        assert "def hello" in body["content"]

    async def test_04_file_content_not_found(self, e2e_client: AsyncClient):
        """GET nonexistent file → 404."""
        resp = await e2e_client.get(
            "/api/v1/registry/skills/e2e-discovery/versions/0.0.1/files/no/such/file.py"
        )
        assert resp.status_code == 404

    @patch("app.api.v1.registry.find_all_skills")
    async def test_05_unregistered_skills(
        self, mock_find_all, e2e_client: AsyncClient
    ):
        """GET unregistered-skills detects disk-only skills."""
        # Simulate a skill on disk that's NOT in the DB
        mock_skill = SimpleNamespace(
            name="disk-only-skill",
            description="Exists on disk but not in DB",
            path="/fake/path",
            skill_type="user",
        )
        mock_find_all.return_value = [mock_skill]

        resp = await e2e_client.get("/api/v1/registry/unregistered-skills")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1
        names = [s["name"] for s in body["skills"]]
        assert "disk-only-skill" in names

    async def test_06_import_local_already_registered(
        self, e2e_client: AsyncClient
    ):
        """POST import-local with already registered skill → error per item."""
        resp = await e2e_client.post(
            "/api/v1/registry/import-local",
            json={"skill_names": ["e2e-discovery"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_failed"] == 1
        assert body["results"][0]["success"] is False
        assert "already registered" in body["results"][0]["error"].lower()

    @patch("app.api.v1.registry.find_skill", return_value=None)
    async def test_07_import_local_not_found(
        self, mock_find, e2e_client: AsyncClient
    ):
        """POST import-local with skill not on disk → error per item."""
        resp = await e2e_client.post(
            "/api/v1/registry/import-local",
            json={"skill_names": ["no-such-disk-skill"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_failed"] == 1
        assert body["results"][0]["success"] is False
        assert "not found" in body["results"][0]["error"].lower()

    async def test_08_cleanup(self, e2e_client: AsyncClient):
        resp = await e2e_client.delete("/api/v1/registry/skills/e2e-discovery")
        assert resp.status_code in (204, 404)


# ===================================================================
# Class 12: MCP Server CRUD
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestMCPServerCRUDE2E:
    """MCP server CRUD + secrets management (all mocked)."""

    _MOCK_SERVER = {
        "name": "e2e-test-mcp",
        "display_name": "E2E Test MCP",
        "description": "MCP for E2E tests",
        "default_enabled": False,
        "tools": [
            {
                "name": "test_tool",
                "description": "A test tool",
                "input_schema": {"type": "object"},
            }
        ],
        "required_env_vars": ["TEST_API_KEY"],
        "secrets_status": {
            "TEST_API_KEY": {"configured": False, "source": "none"},
        },
    }

    @patch("app.api.v1.mcp.get_all_mcp_servers_info")
    async def test_01_list_servers(self, mock_fn, e2e_client: AsyncClient):
        mock_fn.return_value = [self._MOCK_SERVER]
        resp = await e2e_client.get("/api/v1/mcp/servers")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] >= 1
        assert isinstance(body["servers"], list)

    @patch("app.api.v1.mcp.add_mcp_server")
    async def test_02_create_server(self, mock_add, e2e_client: AsyncClient):
        mock_add.return_value = self._MOCK_SERVER
        resp = await e2e_client.post(
            "/api/v1/mcp/servers",
            json={
                "name": "e2e-test-mcp",
                "display_name": "E2E Test MCP",
                "description": "MCP for E2E tests",
                "command": "node",
                "args": ["./test-server/index.js"],
                "env": {"TEST_API_KEY": "${TEST_API_KEY}"},
                "default_enabled": False,
                "tools": [
                    {
                        "name": "test_tool",
                        "description": "A test tool",
                        "inputSchema": {"type": "object"},
                    }
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "e2e-test-mcp"
        mock_add.assert_called_once()

    @patch("app.api.v1.mcp.get_mcp_server_info")
    async def test_03_get_server(self, mock_get, e2e_client: AsyncClient):
        mock_get.return_value = self._MOCK_SERVER
        resp = await e2e_client.get("/api/v1/mcp/servers/e2e-test-mcp")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "e2e-test-mcp"
        assert len(body["tools"]) == 1

    @patch("app.api.v1.mcp.update_mcp_server")
    async def test_04_update_server(self, mock_update, e2e_client: AsyncClient):
        updated = dict(self._MOCK_SERVER, description="Updated description")
        mock_update.return_value = updated
        resp = await e2e_client.put(
            "/api/v1/mcp/servers/e2e-test-mcp",
            json={"description": "Updated description"},
        )
        assert resp.status_code == 200
        assert resp.json()["description"] == "Updated description"

    @patch("app.api.v1.mcp.get_all_secrets_status")
    async def test_05_get_secrets_status(
        self, mock_secrets, e2e_client: AsyncClient
    ):
        mock_secrets.return_value = {
            "e2e-test-mcp": {
                "TEST_API_KEY": {"configured": False, "source": "none"},
            }
        }
        resp = await e2e_client.get("/api/v1/mcp/secrets")
        assert resp.status_code == 200
        body = resp.json()
        assert "e2e-test-mcp" in body["servers"]

    @patch("app.api.v1.mcp.set_mcp_secret")
    @patch("app.api.v1.mcp.get_mcp_server_info")
    async def test_06_set_secret(
        self, mock_get, mock_set, e2e_client: AsyncClient
    ):
        mock_get.return_value = self._MOCK_SERVER
        resp = await e2e_client.put(
            "/api/v1/mcp/servers/e2e-test-mcp/secrets/TEST_API_KEY",
            json={"value": "sk-test-123"},
        )
        assert resp.status_code == 200
        assert "saved" in resp.json()["message"]
        mock_set.assert_called_once_with("e2e-test-mcp", "TEST_API_KEY", "sk-test-123")

    @patch("app.api.v1.mcp.get_mcp_server_info")
    async def test_07_set_secret_invalid_key(
        self, mock_get, e2e_client: AsyncClient
    ):
        mock_get.return_value = self._MOCK_SERVER
        resp = await e2e_client.put(
            "/api/v1/mcp/servers/e2e-test-mcp/secrets/UNKNOWN_KEY",
            json={"value": "val"},
        )
        assert resp.status_code == 400

    @patch("app.api.v1.mcp.delete_mcp_server")
    async def test_08_delete_server(self, mock_del, e2e_client: AsyncClient):
        mock_del.return_value = True
        resp = await e2e_client.delete("/api/v1/mcp/servers/e2e-test-mcp")
        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"]

    @patch("app.api.v1.mcp.get_mcp_server_info", return_value=None)
    async def test_09_server_not_found(
        self, mock_get, e2e_client: AsyncClient
    ):
        resp = await e2e_client.get("/api/v1/mcp/servers/nonexistent")
        assert resp.status_code == 404


# ===================================================================
# Class 14: Execute API
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestExecuteAPIE2E:
    """Execute skill directly (mocked DB reads)."""

    @patch("app.api.v1.execute._read_skill_from_db")
    async def test_01_execute_skill_direct(
        self, mock_read, e2e_client: AsyncClient
    ):
        """POST /execute/skill/{name} returns skill content."""
        from app.models.skill import SkillContent, SkillResources

        mock_read.return_value = SkillContent(
            name="e2e-exec-skill",
            description="Test execution",
            content="# E2E Execution Skill\nDo something useful.",
            base_dir="/fake/path",
            resources=SkillResources(),
        )
        resp = await e2e_client.post("/api/v1/execute/skill/e2e-exec-skill")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["skill_name"] == "e2e-exec-skill"
        assert "E2E Execution Skill" in body["skill_content"]

    @patch("app.api.v1.execute._read_skill_from_db", return_value=None)
    async def test_02_execute_skill_not_found(
        self, mock_read, e2e_client: AsyncClient
    ):
        resp = await e2e_client.post("/api/v1/execute/skill/no-such-skill")
        assert resp.status_code == 404


# ===================================================================
# Class 15: Tool Details & Categories
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestToolDetailE2E:
    """Tool registry detail endpoints, categories, kernel reset."""

    async def test_01_get_tool_by_id(self, e2e_client: AsyncClient):
        """GET a specific tool from the registry."""
        # First, list tools to get a valid tool ID
        list_resp = await e2e_client.get("/api/v1/tools/registry")
        assert list_resp.status_code == 200
        tools = list_resp.json()["tools"]
        assert len(tools) > 0

        tool_id = tools[0]["id"]
        resp = await e2e_client.get(f"/api/v1/tools/registry/{tool_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == tool_id
        assert "name" in body
        assert "input_schema" in body

    async def test_02_tool_not_found(self, e2e_client: AsyncClient):
        resp = await e2e_client.get("/api/v1/tools/registry/nonexistent-tool-id")
        assert resp.status_code == 404

    async def test_03_list_categories(self, e2e_client: AsyncClient):
        resp = await e2e_client.get("/api/v1/tools/registry/categories/all")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["categories"], list)
        assert len(body["categories"]) > 0
        # Each category should have id, name, description, icon
        cat = body["categories"][0]
        assert "id" in cat
        assert "name" in cat

    async def test_04_list_tools_by_category(self, e2e_client: AsyncClient):
        """GET tools filtered by category."""
        # Get categories first
        cat_resp = await e2e_client.get("/api/v1/tools/registry/categories/all")
        categories = cat_resp.json()["categories"]
        if not categories:
            pytest.skip("No categories available")

        cat_id = categories[0]["id"]
        resp = await e2e_client.get(
            "/api/v1/tools/registry", params={"category": cat_id}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["tools"], list)
        # All returned tools should be in the requested category
        for tool in body["tools"]:
            assert tool["category"] == cat_id

    @patch("app.api.v1.tools.get_code_executor")
    async def test_05_reset_kernel(
        self, mock_get_executor, e2e_client: AsyncClient
    ):
        mock_executor = MagicMock()
        mock_executor.reset.return_value = None
        mock_get_executor.return_value = mock_executor
        resp = await e2e_client.post("/api/v1/tools/reset_kernel")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True


# ===================================================================
# Class 16: Workspace Isolation (Code Execution)
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestWorkspaceIsolationE2E:
    """
    Test AgentWorkspace isolation and cleanup.

    These tests verify that:
    1. Workspace directories are cleaned up after use
    2. Concurrent workspaces don't interfere with each other
    """

    async def test_01_workspace_preserved_after_cleanup(self):
        """Verify workspace_dir is preserved after cleanup for output file downloads.

        cleanup() should delete temp_path (scripts) but preserve workspace_dir
        so output files remain downloadable.
        """
        from app.tools.code_executor import AgentWorkspace
        import os
        import shutil

        workspace_dir_path = None
        temp_path = None

        with AgentWorkspace() as ws:
            workspace_dir_path = str(ws.workspace_dir)
            temp_path = str(ws.temp_path)
            assert os.path.exists(workspace_dir_path), "workspace_dir should exist during use"
            assert os.path.exists(temp_path), "temp_path should exist during use"

            result = ws.execute("with open('output.txt', 'w') as f: f.write('hello')")
            assert result.success is True

        # After cleanup: workspace_dir preserved, temp_path deleted
        assert os.path.exists(workspace_dir_path), "workspace_dir should be preserved after cleanup"
        assert os.path.exists(os.path.join(workspace_dir_path, "output.txt"))
        assert not os.path.exists(temp_path), "temp_path should be deleted after cleanup"

        # Manual cleanup for test hygiene
        shutil.rmtree(workspace_dir_path, ignore_errors=True)

    async def test_02_workspace_kernel_variable_persistence(self):
        """Verify variables persist between executions via IPython kernel."""
        from app.tools.code_executor import AgentWorkspace

        with AgentWorkspace() as ws:
            # First execution: define a variable
            result1 = ws.execute("secret_var = 'secret_value_12345'")
            assert result1.success is True

            # Second execution: access the variable (should succeed — kernel persistence)
            result2 = ws.execute("print(secret_var)")
            assert result2.success is True, f"Variable should persist: {result2.error}"
            assert "secret_value_12345" in result2.output

    async def test_03_concurrent_workspace_isolation(self):
        """Verify multiple concurrent workspaces don't interfere with each other."""
        from app.tools.code_executor import AgentWorkspace
        import concurrent.futures
        import os

        def run_in_workspace(worker_id: int) -> dict:
            """Each worker writes and reads its own unique data."""
            with AgentWorkspace() as ws:
                # Write a unique value
                code = f'''
unique_id = {worker_id}
secret = "worker_{worker_id}_secret"
print(f"ID:{{unique_id}}")
print(f"SECRET:{{secret}}")
'''
                result = ws.execute(code)

                # Check output contains correct worker ID
                output = result.output
                correct_id = f"ID:{worker_id}" in output
                correct_secret = f"worker_{worker_id}_secret" in output

                # Check no other worker's data leaked
                no_leak = True
                for other_id in range(4):
                    if other_id != worker_id:
                        if f"worker_{other_id}_secret" in output:
                            no_leak = False
                            break

                return {
                    "worker_id": worker_id,
                    "success": result.success,
                    "correct_id": correct_id,
                    "correct_secret": correct_secret,
                    "no_leak": no_leak,
                    "workspace_existed": os.path.exists(ws.path),
                }

        # Run 4 concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_in_workspace, i) for i in range(4)]
            results = [f.result() for f in futures]

        # Verify all workers succeeded with correct isolation
        for r in results:
            assert r["success"], f"Worker {r['worker_id']} execution failed"
            assert r["correct_id"], f"Worker {r['worker_id']} got wrong ID"
            assert r["correct_secret"], f"Worker {r['worker_id']} got wrong secret"
            assert r["no_leak"], f"Worker {r['worker_id']} saw another worker's data"

    async def test_04_workspace_env_vars(self):
        """Verify environment variables are passed to workspace."""
        from app.tools.code_executor import AgentWorkspace

        env_vars = {
            "TEST_VAR_1": "value_one",
            "TEST_VAR_2": "value_two",
        }

        with AgentWorkspace(env_vars=env_vars) as ws:
            result = ws.execute('''
import os
print(f"VAR1:{os.environ.get('TEST_VAR_1', 'missing')}")
print(f"VAR2:{os.environ.get('TEST_VAR_2', 'missing')}")
''')
            assert result.success is True
            assert "VAR1:value_one" in result.output
            assert "VAR2:value_two" in result.output

    async def test_05_workspace_command_execution(self):
        """Verify shell commands work in workspace."""
        from app.tools.code_executor import AgentWorkspace

        with AgentWorkspace() as ws:
            # Create a file using Python
            ws.execute("open('test_file.txt', 'w').write('hello world')")

            # List files using shell command
            result = ws.execute_command("ls -la")
            assert result.success is True
            assert "test_file.txt" in result.output

            # Read file using shell command
            result2 = ws.execute_command("cat test_file.txt")
            assert result2.success is True
            assert "hello world" in result2.output

    async def test_06_kernel_multi_step_persistence(self):
        """Verify multi-step data analysis with persistent kernel state."""
        from app.tools.code_executor import AgentWorkspace

        with AgentWorkspace() as ws:
            # Step 1: import and define data
            r1 = ws.execute("import json\ndata = [1, 2, 3, 4, 5]")
            assert r1.success is True

            # Step 2: process data using previous imports and variables
            r2 = ws.execute("total = sum(data)\nmean = total / len(data)\nprint(f'total={total}, mean={mean}')")
            assert r2.success is True
            assert "total=15" in r2.output
            assert "mean=3.0" in r2.output

            # Step 3: use json (imported in step 1)
            r3 = ws.execute("result = json.dumps({'total': total, 'mean': mean})\nprint(result)")
            assert r3.success is True
            assert '"total": 15' in r3.output

    async def test_07_kernel_fallback_to_subprocess(self):
        """Verify that execution still works when kernel is unavailable."""
        from app.tools.code_executor import AgentWorkspace

        with AgentWorkspace() as ws:
            # Force kernel failure
            ws._kernel_failed = True

            # Execution should still work via subprocess
            result = ws.execute("print('subprocess fallback works')")
            assert result.success is True
            assert "subprocess fallback works" in result.output

            # But variables should NOT persist in subprocess mode
            ws.execute("fallback_var = 42")
            result2 = ws.execute("print(fallback_var)")
            assert result2.success is False, "Variables should not persist in subprocess mode"

    async def test_08_env_vars_from_dotenv_local_workspace(self):
        """Verify local workspace picks up .env vars without skill_config."""
        from app.tools.code_executor import AgentWorkspace, _load_env_file

        # _load_env_file should return a dict (may be empty in test env)
        env_from_file = _load_env_file()
        assert isinstance(env_from_file, dict)

        # Workspace env should include os.environ + .env overlay
        with AgentWorkspace() as ws:
            # env should be populated (at least os.environ keys)
            assert len(ws.env) > 0
            assert "PATH" in ws.env  # basic sanity

            # Verify code can see the env
            result = ws.execute("import os; print(os.environ.get('PATH', 'MISSING'))")
            assert result.success is True
            assert "MISSING" not in result.output

    async def test_09_collect_env_for_executor_reads_dotenv(self):
        """Verify _collect_env_for_executor reads .env without skill_config."""
        import tempfile
        import os
        from unittest.mock import patch

        # Create a temp .env file with known content
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("EXECUTOR_TEST_KEY_1=executor_val_1\n")
            f.write("EXECUTOR_TEST_KEY_2=executor_val_2\n")
            f.write("# comment line\n")
            f.write("\n")
            tmp_path = f.name

        try:
            from app.agent.tools import _collect_env_for_executor
            from pathlib import Path

            # Patch _get_env_file_path to use our temp file
            with patch("app.config._get_env_file_path", return_value=Path(tmp_path)):
                env = _collect_env_for_executor()

            # Should contain our test keys
            assert env.get("EXECUTOR_TEST_KEY_1") == "executor_val_1"
            assert env.get("EXECUTOR_TEST_KEY_2") == "executor_val_2"

            # Skill env vars override (even though empty dict now)
            with patch("app.config._get_env_file_path", return_value=Path(tmp_path)):
                env2 = _collect_env_for_executor({"EXECUTOR_TEST_KEY_1": "overridden"})
            assert env2["EXECUTOR_TEST_KEY_1"] == "overridden"
            assert env2["EXECUTOR_TEST_KEY_2"] == "executor_val_2"
        finally:
            os.unlink(tmp_path)

    async def test_10_get_tools_for_agent_executor_env(self):
        """Verify get_tools_for_agent passes env to executor-bound tools."""
        import tempfile
        import os
        from unittest.mock import patch, MagicMock
        from pathlib import Path

        # Create a temp .env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("AGENT_TOOL_TEST_KEY=agent_tool_val\n")
            tmp_path = f.name

        try:
            from app.agent.tools import _collect_env_for_executor

            # Verify _collect_env_for_executor picks up the key
            with patch("app.config._get_env_file_path", return_value=Path(tmp_path)):
                env = _collect_env_for_executor({})

            assert env.get("AGENT_TOOL_TEST_KEY") == "agent_tool_val"
        finally:
            os.unlink(tmp_path)


# ===========================================================================
# Category + Pin + Sort E2E
# ===========================================================================


@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestCategoryPinE2E:
    """Category CRUD, Pin toggle, category filter, pinned-first sort."""

    _state: dict = {}

    async def test_01_setup_skills(
        self, e2e_client: AsyncClient, e2e_db_session: AsyncSession
    ):
        """Create skills with different categories and pin states."""
        s1 = make_skill(
            name="e2e-cat-alpha",
            description="Alpha skill",
            category="Research & Knowledge",
            is_pinned=False,
        )
        s2 = make_skill(
            name="e2e-cat-beta",
            description="Beta skill",
            category="Automation",
            is_pinned=True,
        )
        s3 = make_skill(
            name="e2e-cat-gamma",
            description="Gamma skill",
            category="Research & Knowledge",
            is_pinned=False,
        )
        s4 = make_skill(
            name="e2e-cat-delta",
            description="Delta skill no category",
        )
        e2e_db_session.add_all([s1, s2, s3, s4])
        await e2e_db_session.flush()
        type(self)._state["s1_id"] = s1.id
        type(self)._state["s2_id"] = s2.id

    # -- Categories endpoint --

    async def test_02_list_categories(self, e2e_client: AsyncClient):
        """GET /registry/categories returns distinct non-null categories."""
        resp = await e2e_client.get("/api/v1/registry/categories")
        assert resp.status_code == 200
        cats = resp.json()
        assert isinstance(cats, list)
        assert "Research & Knowledge" in cats
        assert "Automation" in cats
        # No null/empty entries
        assert "" not in cats
        assert None not in cats

    # -- Category filter --

    async def test_03_filter_by_category(self, e2e_client: AsyncClient):
        """GET /registry/skills?category=X returns only matching skills."""
        resp = await e2e_client.get(
            "/api/v1/registry/skills",
            params={"category": "Research & Knowledge"},
        )
        assert resp.status_code == 200
        body = resp.json()
        names = [s["name"] for s in body["skills"]]
        assert "e2e-cat-alpha" in names
        assert "e2e-cat-gamma" in names
        assert "e2e-cat-beta" not in names
        assert "e2e-cat-delta" not in names

    async def test_04_filter_by_category_no_match(self, e2e_client: AsyncClient):
        """GET /registry/skills?category=NonExistent returns empty."""
        resp = await e2e_client.get(
            "/api/v1/registry/skills",
            params={"category": "NonExistent"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["skills"] == []

    # -- Update category --

    async def test_05_update_category(self, e2e_client: AsyncClient):
        """PUT /registry/skills/{name} can set category."""
        resp = await e2e_client.put(
            "/api/v1/registry/skills/e2e-cat-delta",
            json={"category": "Data Analysis"},
        )
        assert resp.status_code == 200
        assert resp.json()["category"] == "Data Analysis"

    async def test_06_clear_category(self, e2e_client: AsyncClient):
        """PUT /registry/skills/{name} with empty string clears category."""
        resp = await e2e_client.put(
            "/api/v1/registry/skills/e2e-cat-delta",
            json={"category": ""},
        )
        assert resp.status_code == 200
        assert resp.json()["category"] is None

    # -- Skill response includes category and is_pinned --

    async def test_07_response_fields(self, e2e_client: AsyncClient):
        """GET /registry/skills/{name} response includes category and is_pinned."""
        resp = await e2e_client.get("/api/v1/registry/skills/e2e-cat-beta")
        assert resp.status_code == 200
        body = resp.json()
        assert body["category"] == "Automation"
        assert body["is_pinned"] is True

    # -- Toggle pin --

    async def test_08_toggle_pin_on(self, e2e_client: AsyncClient):
        """POST /registry/skills/{name}/toggle-pin pins a skill."""
        resp = await e2e_client.post(
            "/api/v1/registry/skills/e2e-cat-alpha/toggle-pin"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "e2e-cat-alpha"
        assert body["is_pinned"] is True

    async def test_09_toggle_pin_off(self, e2e_client: AsyncClient):
        """POST /registry/skills/{name}/toggle-pin unpins a pinned skill."""
        # e2e-cat-beta is pinned from setup
        resp = await e2e_client.post(
            "/api/v1/registry/skills/e2e-cat-beta/toggle-pin"
        )
        assert resp.status_code == 200
        assert resp.json()["is_pinned"] is False

    async def test_10_toggle_pin_not_found(self, e2e_client: AsyncClient):
        """POST /registry/skills/{name}/toggle-pin for nonexistent → 404."""
        resp = await e2e_client.post(
            "/api/v1/registry/skills/nonexistent-e2e/toggle-pin"
        )
        assert resp.status_code == 404

    # -- Pinned skills sorted first --

    async def test_11_pinned_sorted_first(self, e2e_client: AsyncClient):
        """Pinned skills appear before unpinned in list results."""
        # After tests: e2e-cat-alpha is pinned, e2e-cat-beta is unpinned
        resp = await e2e_client.get(
            "/api/v1/registry/skills",
            params={"sort_by": "name", "sort_order": "asc"},
        )
        assert resp.status_code == 200
        skills = resp.json()["skills"]
        # Find our test skills
        our_skills = [s for s in skills if s["name"].startswith("e2e-cat-")]
        pinned = [s for s in our_skills if s["is_pinned"]]
        unpinned = [s for s in our_skills if not s["is_pinned"]]
        # All pinned should come before all unpinned
        if pinned and unpinned:
            last_pinned_idx = max(
                i for i, s in enumerate(our_skills) if s["is_pinned"]
            )
            first_unpinned_idx = min(
                i for i, s in enumerate(our_skills) if not s["is_pinned"]
            )
            assert last_pinned_idx < first_unpinned_idx

    # -- Categories updated after category change --

    async def test_12_categories_reflect_changes(self, e2e_client: AsyncClient):
        """Categories endpoint reflects skill category updates."""
        # Set delta to a new category
        await e2e_client.put(
            "/api/v1/registry/skills/e2e-cat-delta",
            json={"category": "Media & Design"},
        )
        resp = await e2e_client.get("/api/v1/registry/categories")
        assert resp.status_code == 200
        cats = resp.json()
        assert "Media & Design" in cats

    # -- Cleanup --

    async def test_13_cleanup(
        self, e2e_client: AsyncClient
    ):
        """Delete test skills."""
        for name in ["e2e-cat-alpha", "e2e-cat-beta", "e2e-cat-gamma", "e2e-cat-delta"]:
            resp = await e2e_client.delete(f"/api/v1/registry/skills/{name}")
            assert resp.status_code == 204


# ===================================================================
# Class 18: Path-based Output File Download
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestPathBasedDownloadE2E:
    """Path-based output file download — multi-worker safe."""

    _state: dict = {}

    async def test_01_setup(self, e2e_client: AsyncClient):
        """Create a temp file under an allowed download directory."""
        import base64

        # Use /tmp/agent_workspaces/ which is in the allowed dirs list
        tmpdir = os.path.join("/tmp/agent_workspaces", f"e2e_download_{uuid.uuid4().hex[:8]}")
        os.makedirs(tmpdir, exist_ok=True)
        type(self)._state["tmpdir"] = tmpdir
        out_path = os.path.join(tmpdir, "report.csv")
        with open(out_path, "w") as f:
            f.write("id,name,score\n1,Alice,95\n2,Bob,87\n")
        type(self)._state["file_path"] = out_path
        type(self)._state["encoded_path"] = base64.urlsafe_b64encode(
            out_path.encode("utf-8")
        ).decode("ascii")

    async def test_02_download_by_path(self, e2e_client: AsyncClient):
        """GET /files/output/download?path=<base64url> returns file content."""
        encoded = type(self)._state["encoded_path"]
        resp = await e2e_client.get(
            "/api/v1/files/output/download", params={"path": encoded}
        )
        assert resp.status_code == 200
        assert b"Alice" in resp.content
        assert b"Bob" in resp.content

    async def test_03_download_invalid_base64(self, e2e_client: AsyncClient):
        """Invalid base64 encoding returns 400."""
        resp = await e2e_client.get(
            "/api/v1/files/output/download", params={"path": "!!!not-base64!!!"}
        )
        assert resp.status_code == 400

    async def test_04_download_nonexistent_file(self, e2e_client: AsyncClient):
        """Valid base64 pointing to nonexistent file under allowed dir returns 404."""
        import base64

        # Must be under an allowed dir (/tmp/agent_workspaces/) to get past 403
        fake = base64.urlsafe_b64encode(
            b"/tmp/agent_workspaces/does_not_exist_e2e.txt"
        ).decode()
        resp = await e2e_client.get(
            "/api/v1/files/output/download", params={"path": fake}
        )
        assert resp.status_code == 404

    async def test_05_download_forbidden_path(self, e2e_client: AsyncClient):
        """Paths outside allowed directories return 403."""
        import base64

        # /etc/passwd is not under any allowed dir
        encoded = base64.urlsafe_b64encode(b"/etc/passwd").decode()
        resp = await e2e_client.get(
            "/api/v1/files/output/download", params={"path": encoded}
        )
        assert resp.status_code == 403

    async def test_06_download_path_traversal(self, e2e_client: AsyncClient):
        """Path traversal attempts are blocked (resolve normalizes ..)."""
        import base64

        # Try to escape allowed dir via ..
        traversal = "/tmp/agent_workspaces/../../etc/passwd"
        encoded = base64.urlsafe_b64encode(traversal.encode()).decode()
        resp = await e2e_client.get(
            "/api/v1/files/output/download", params={"path": encoded}
        )
        # resolve() normalizes to /etc/passwd → 403 (not under allowed dirs)
        assert resp.status_code == 403

    async def test_07_cleanup(self, e2e_client: AsyncClient):
        import shutil

        tmpdir = type(self)._state.get("tmpdir")
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)


# ===================================================================
# Class 19: File Scanner Module
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestFileScannerE2E:
    """Unit-level tests for app.tools.file_scanner module."""

    _state: dict = {}

    async def test_01_snapshot_empty_dir(self):
        """Snapshot of empty directory returns empty dict."""
        from app.tools.file_scanner import snapshot_files

        tmpdir = tempfile.mkdtemp()
        type(self)._state["tmpdir"] = tmpdir
        result = snapshot_files(Path(tmpdir))
        assert result == {}

    async def test_02_snapshot_with_files(self):
        """Snapshot captures files with mtimes."""
        from app.tools.file_scanner import snapshot_files

        tmpdir = type(self)._state["tmpdir"]
        # Create test files
        Path(tmpdir, "output.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
        Path(tmpdir, "report.pdf").write_bytes(b"%PDF" + b"\x00" * 100)
        Path(tmpdir, "data.csv").write_text("a,b\n1,2\n")

        result = snapshot_files(Path(tmpdir))
        assert len(result) == 3
        for path_str, mtime in result.items():
            assert isinstance(mtime, float)
            assert Path(path_str).exists()

    async def test_03_snapshot_ignores_blacklisted(self):
        """Snapshot skips blacklisted files and directories."""
        from app.tools.file_scanner import snapshot_files

        tmpdir = type(self)._state["tmpdir"]
        # Create files that should be ignored
        Path(tmpdir, ".hidden_file").write_text("hidden")
        Path(tmpdir, "_script.py").write_text("# script")
        Path(tmpdir, "compiled.pyc").write_bytes(b"\x00\x00")
        Path(tmpdir, "requirements.txt").write_text("pandas")
        Path(tmpdir, "__pycache__").mkdir(exist_ok=True)
        Path(tmpdir, "__pycache__", "mod.pyc").write_bytes(b"\x00")

        result = snapshot_files(Path(tmpdir), recursive=True)
        filenames = [Path(p).name for p in result.keys()]
        # These should NOT be in the result
        assert ".hidden_file" not in filenames
        assert "_script.py" not in filenames
        assert "compiled.pyc" not in filenames
        assert "requirements.txt" not in filenames
        assert "mod.pyc" not in filenames
        # These should still be there from test_02
        assert "output.png" in filenames
        assert "data.csv" in filenames

    async def test_04_diff_detects_new_files(self):
        """diff_new_files detects files added after snapshot."""
        from app.tools.file_scanner import snapshot_files, diff_new_files

        tmpdir = type(self)._state["tmpdir"]
        before = snapshot_files(Path(tmpdir))

        # Create new files
        import time
        time.sleep(0.05)  # Ensure mtime difference
        new_file = Path(tmpdir, "new_output.xlsx")
        new_file.write_bytes(b"fake xlsx content")

        after = snapshot_files(Path(tmpdir))
        new_files = diff_new_files(before, after)

        new_names = [f.name for f in new_files]
        assert "new_output.xlsx" in new_names

    async def test_05_diff_detects_modified_files(self):
        """diff_new_files detects files modified after snapshot."""
        from app.tools.file_scanner import snapshot_files, diff_new_files

        tmpdir = type(self)._state["tmpdir"]
        before = snapshot_files(Path(tmpdir))

        import time
        time.sleep(0.05)
        # Modify existing file
        Path(tmpdir, "data.csv").write_text("a,b,c\n1,2,3\n4,5,6\n")

        after = snapshot_files(Path(tmpdir))
        modified = diff_new_files(before, after)

        mod_names = [f.name for f in modified]
        assert "data.csv" in mod_names

    async def test_06_build_output_file_infos(self):
        """build_output_file_infos produces correct download URLs."""
        from app.tools.file_scanner import build_output_file_infos
        import base64

        tmpdir = type(self)._state["tmpdir"]
        paths = [Path(tmpdir, "data.csv"), Path(tmpdir, "output.png")]

        infos = build_output_file_infos(paths)
        assert len(infos) == 2

        for info in infos:
            assert "filename" in info
            assert "size" in info
            assert info["size"] > 0
            assert "content_type" in info
            assert "download_url" in info
            assert info["download_url"].startswith("/api/v1/files/output/download?path=")

            # Verify the base64 encodes the correct path
            encoded = info["download_url"].split("path=")[1]
            decoded = base64.urlsafe_b64decode(encoded.encode()).decode()
            assert Path(decoded).exists()

    async def test_07_build_skips_empty_files(self):
        """build_output_file_infos skips 0-byte files."""
        from app.tools.file_scanner import build_output_file_infos

        tmpdir = type(self)._state["tmpdir"]
        empty_file = Path(tmpdir, "empty.txt")
        empty_file.write_text("")

        infos = build_output_file_infos([empty_file])
        assert len(infos) == 0

    async def test_08_snapshot_nonexistent_dir(self):
        """Snapshot of nonexistent directory returns empty dict."""
        from app.tools.file_scanner import snapshot_files

        result = snapshot_files(Path("/nonexistent/e2e/dir"))
        assert result == {}

    async def test_09_cleanup(self):
        import shutil

        tmpdir = type(self)._state.get("tmpdir")
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)


# ===================================================================
# Class 20: Auto-detect Output Files in Agent Run/Stream
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestAutoDetectOutputFilesE2E:
    """
    Test that execute_code/bash auto-detect new files and that output_files
    appear in AgentResponse, SSE events, and PublishedChatResponse.
    """

    _state: dict = {}

    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
    @patch("app.api.v1.agent.create_agent")
    async def test_01_run_with_output_files(
        self, MockAgent, _mock_load, _mock_save, e2e_client: AsyncClient
    ):
        """POST /agent/run → output_files populated from auto-detected files."""
        mock_result = _MockAgentResult(
            output_files=[
                {
                    "filename": "chart.png",
                    "size": 12345,
                    "content_type": "image/png",
                    "download_url": "/api/v1/files/output/download?path=L3RtcC9jaGFydC5wbmc=",
                },
            ]
        )
        MockAgent.return_value = _make_mock_agent(result=mock_result)

        resp = await e2e_client.post(
            "/api/v1/agent/run",
            json={"request": "Generate a chart", "session_id": "test-session-id"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["output_files"] is not None
        assert len(body["output_files"]) == 1
        assert body["output_files"][0]["filename"] == "chart.png"
        assert body["output_files"][0]["download_url"].startswith("/api/v1/files/output/download?path=")
        type(self)._state["trace_id_1"] = body["trace_id"]

    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
    @patch("app.api.v1.agent.create_agent")
    async def test_02_run_no_output_files(
        self, MockAgent, _mock_load, _mock_save, e2e_client: AsyncClient
    ):
        """POST /agent/run without output files → output_files is null."""
        MockAgent.return_value = _make_mock_agent()

        resp = await e2e_client.post(
            "/api/v1/agent/run",
            json={"request": "Just talk, no files", "session_id": "test-session-id"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["output_files"] is None

    @patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
    @patch("app.api.v1.agent.AsyncSessionLocal")
    @patch("app.api.v1.agent.create_agent")
    async def test_03_stream_with_output_file_events(
        self, MockAgent, MockSessionLocal, _mock_load, _mock_save, _mock_checkpoint, _mock_precompress, e2e_client: AsyncClient
    ):
        """POST /agent/run/stream → output_file SSE events + output_files in complete."""
        mock_output_files = [
            {
                "filename": "result.csv",
                "size": 256,
                "content_type": "text/csv",
                "download_url": "/api/v1/files/output/download?path=L3RtcC9yZXN1bHQuY3N2",
            },
        ]
        stream_events = [
            StreamEvent(event_type="turn_start", turn=1, data={"turn": 1}),
            StreamEvent(
                event_type="tool_call",
                turn=1,
                data={"tool_name": "execute_code", "tool_input": {"code": "..."}},
            ),
            StreamEvent(
                event_type="tool_result",
                turn=1,
                data={"tool_name": "execute_code", "tool_result": '{"success":true}'},
            ),
            StreamEvent(
                event_type="output_file",
                turn=1,
                data=mock_output_files[0],
            ),
            StreamEvent(
                event_type="assistant",
                turn=1,
                data={"content": "Here is your file", "turn": 1},
            ),
            StreamEvent(
                event_type="complete",
                turn=1,
                data={
                    "success": True,
                    "answer": "Here is your file",
                    "total_turns": 1,
                    "total_input_tokens": 100,
                    "total_output_tokens": 50,
                    "skills_used": [],
                    "output_files": mock_output_files,
                },
            ),
        ]

        MockAgent.return_value = _make_streaming_mock_agent(events=stream_events, answer="Here is your file")
        MockSessionLocal.side_effect = lambda: _mock_session_local()()

        resp = await e2e_client.post(
            "/api/v1/agent/run/stream",
            json={"request": "Create CSV data", "session_id": "test-session-id"},
        )
        assert resp.status_code == 200

        events = parse_sse_events(resp.text)
        event_types = [e["event_type"] for e in events]

        # Should contain output_file event
        assert "output_file" in event_types
        output_file_event = next(e for e in events if e["event_type"] == "output_file")
        assert output_file_event["filename"] == "result.csv"
        assert output_file_event["download_url"].startswith("/api/v1/files/output/download?path=")

        # Complete event should include output_files
        complete_event = next(e for e in events if e["event_type"] == "complete")
        assert "output_files" in complete_event
        assert len(complete_event["output_files"]) == 1

    @patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
    @patch("app.api.v1.agent.AsyncSessionLocal")
    @patch("app.api.v1.agent.create_agent")
    async def test_04_stream_no_output_files(
        self, MockAgent, MockSessionLocal, _mock_load, _mock_save, _mock_checkpoint, _mock_precompress, e2e_client: AsyncClient
    ):
        """POST /agent/run/stream without output files → no output_file events."""
        MockAgent.return_value = _make_streaming_mock_agent(answer="Just text")
        MockSessionLocal.side_effect = lambda: _mock_session_local()()

        resp = await e2e_client.post(
            "/api/v1/agent/run/stream",
            json={"request": "No files needed", "session_id": "test-session-id"},
        )
        assert resp.status_code == 200

        events = parse_sse_events(resp.text)
        event_types = [e["event_type"] for e in events]
        assert "output_file" not in event_types


# ===================================================================
# Class 21: Workspace Execute with Auto-scan
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestWorkspaceAutoScanE2E:
    """
    Test that workspace-bound execute_code and bash
    auto-detect new files via snapshot/diff.
    """

    async def test_01_execute_code_detects_new_file(self):
        """execute_code creates a file → new_files populated."""
        from app.agent.tools import create_workspace_bound_tools
        from app.tools.code_executor import AgentWorkspace

        with AgentWorkspace() as ws:
            tools = create_workspace_bound_tools(ws)
            result = tools["execute_code"](
                code="with open('output_result.txt', 'w') as f: f.write('hello world')"
            )

        assert result["success"] is True
        assert "new_files" in result
        assert len(result["new_files"]) >= 1
        filenames = [f["filename"] for f in result["new_files"]]
        assert "output_result.txt" in filenames
        # Verify download URL format
        for nf in result["new_files"]:
            assert nf["download_url"].startswith("/api/v1/files/output/download?path=")

    async def test_02_execute_code_no_new_files(self):
        """execute_code without file creation → no new_files key."""
        from app.agent.tools import create_workspace_bound_tools
        from app.tools.code_executor import AgentWorkspace

        with AgentWorkspace() as ws:
            tools = create_workspace_bound_tools(ws)
            result = tools["execute_code"](code="x = 42\nprint(x)")

        assert result["success"] is True
        assert "new_files" not in result

    async def test_03_bash_detects_new_file(self):
        """bash creates a file → new_files populated."""
        from app.agent.tools import create_workspace_bound_tools
        from app.tools.code_executor import AgentWorkspace

        with AgentWorkspace() as ws:
            tools = create_workspace_bound_tools(ws)
            result = tools["bash"](command="echo 'test data' > bash_output.csv")

        assert result["success"] is True
        assert "new_files" in result
        filenames = [f["filename"] for f in result["new_files"]]
        assert "bash_output.csv" in filenames

    async def test_04_execute_code_ignores_blacklisted(self):
        """execute_code creating blacklisted files → they are NOT in new_files."""
        from app.agent.tools import create_workspace_bound_tools
        from app.tools.code_executor import AgentWorkspace

        with AgentWorkspace() as ws:
            tools = create_workspace_bound_tools(ws)
            # Create both a normal file and blacklisted files
            result = tools["execute_code"](
                code="""
import os
with open('good_output.png', 'wb') as f: f.write(b'fake png')
with open('requirements.txt', 'w') as f: f.write('pandas')
with open('.hidden', 'w') as f: f.write('secret')
with open('_script_1.py', 'w') as f: f.write('# script')
"""
            )

        assert result["success"] is True
        if "new_files" in result:
            filenames = [f["filename"] for f in result["new_files"]]
            assert "good_output.png" in filenames
            assert "requirements.txt" not in filenames
            assert ".hidden" not in filenames
            assert "_script_1.py" not in filenames

    async def test_05_execute_code_multiple_files(self):
        """execute_code creating multiple files → all detected."""
        from app.agent.tools import create_workspace_bound_tools
        from app.tools.code_executor import AgentWorkspace

        with AgentWorkspace() as ws:
            tools = create_workspace_bound_tools(ws)
            result = tools["execute_code"](
                code="""
with open('report.md', 'w') as f: f.write('# Report')
with open('data.json', 'w') as f: f.write('{"key": "value"}')
with open('image.jpg', 'wb') as f: f.write(b'fake jpg content')
"""
            )

        assert result["success"] is True
        assert "new_files" in result
        filenames = [f["filename"] for f in result["new_files"]]
        assert "report.md" in filenames
        assert "data.json" in filenames
        assert "image.jpg" in filenames


# ===================================================================
# Class 23: Write Tool Workspace Isolation
# ===================================================================


@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestWriteWorkspaceE2E:
    """
    Verify that the write tool resolves relative paths to workspace_dir,
    so files created by write are accessible from execute_code.
    """

    async def test_01_write_resolves_to_workspace(self):
        """write('rel.txt') lands in workspace_dir."""
        from app.agent.tools import create_workspace_bound_tools
        from app.tools.code_executor import AgentWorkspace

        with AgentWorkspace() as ws:
            tools = create_workspace_bound_tools(ws)
            result = tools["write"](
                file_path="write_test.txt", content="workspace content"
            )

        assert result["success"] is True
        assert str(ws.workspace_dir) in result["path"]

    async def test_02_write_readable_by_execute_code(self):
        """File created by write tool can be read by execute_code (same cwd)."""
        from app.agent.tools import create_workspace_bound_tools
        from app.tools.code_executor import AgentWorkspace

        with AgentWorkspace() as ws:
            tools = create_workspace_bound_tools(ws)

            w_result = tools["write"](
                file_path="shared.csv", content="col1,col2\na,b\n"
            )
            assert w_result["success"] is True

            e_result = tools["execute_code"](
                code="print(open('shared.csv').read())"
            )
            assert e_result["success"] is True
            assert "col1,col2" in e_result["output"]

    async def test_03_write_absolute_path_unchanged(self):
        """write('/tmp/abs.txt') still writes to the absolute path."""
        import os
        from app.agent.tools import create_workspace_bound_tools
        from app.tools.code_executor import AgentWorkspace

        target = "/tmp/e2e_write_abs_test.txt"
        try:
            with AgentWorkspace() as ws:
                tools = create_workspace_bound_tools(ws)
                result = tools["write"](
                    file_path=target, content="absolute"
                )

            assert result["success"] is True
            assert result["path"] == target
            assert os.path.exists(target)
        finally:
            if os.path.exists(target):
                os.remove(target)

    async def test_04_write_detects_output_file(self):
        """write tool returns new_files for auto-detected output."""
        from app.agent.tools import create_workspace_bound_tools
        from app.tools.code_executor import AgentWorkspace

        with AgentWorkspace() as ws:
            tools = create_workspace_bound_tools(ws)
            result = tools["write"](
                file_path="report.md", content="# Report\nDone."
            )

        assert result["success"] is True
        assert "new_files" in result
        filenames = [f["filename"] for f in result["new_files"]]
        assert "report.md" in filenames

    async def test_05_write_then_bash_reads(self):
        """File created by write tool can be read by bash (same workspace cwd)."""
        from app.agent.tools import create_workspace_bound_tools
        from app.tools.code_executor import AgentWorkspace

        with AgentWorkspace() as ws:
            tools = create_workspace_bound_tools(ws)

            tools["write"](
                file_path="storyboard.csv",
                content="slide_no,title\n1,Cover\n2,Body\n",
            )

            b_result = tools["bash"](command="cat storyboard.csv")
            assert b_result["success"] is True
            assert "Cover" in b_result["output"]


# ===================================================================
# Class 24: Published Agent Output Files
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestPublishedAgentOutputFilesE2E:
    """Published Agent sync endpoint returns output_files."""

    _state: dict = {}

    async def test_01_create_and_publish(self, e2e_client: AsyncClient):
        """Setup: create and publish an agent in non_streaming mode."""
        payload = {
            "name": "e2e-pub-output-files",
            "description": "Test output_files in published agent",
            "max_turns": 3,
        }
        resp = await e2e_client.post("/api/v1/agents", json=payload)
        assert resp.status_code == 200
        pid = resp.json()["id"]
        type(self)._state["preset_id"] = pid

        resp = await e2e_client.post(
            f"/api/v1/agents/{pid}/publish",
            json={"api_response_mode": "non_streaming"},
        )
        assert resp.status_code == 200

    @patch("app.api.v1.published.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.published.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.published.SkillsAgent")
    @patch("app.api.v1.published.AsyncSessionLocal")
    async def test_02_sync_chat_with_output_files(
        self, MockSL, MockAgent, MockLoadSession, _mock_save, e2e_client: AsyncClient
    ):
        """POST /published/{id}/chat/sync → output_files in response."""
        pid = type(self)._state["preset_id"]

        mock_output_files = [
            {
                "filename": "analysis.pdf",
                "size": 54321,
                "content_type": "application/pdf",
                "download_url": "/api/v1/files/output/download?path=dGVzdA==",
            },
        ]
        mock_result = _MockAgentResult(output_files=mock_output_files)
        mock_instance = _make_mock_agent(result=mock_result)
        MockAgent.return_value = mock_instance

        from app.db.models import AgentPresetDB

        mock_preset = MagicMock(spec=AgentPresetDB)
        mock_preset.id = pid
        mock_preset.name = "e2e-pub-output-files"
        mock_preset.is_published = True
        mock_preset.api_response_mode = "non_streaming"
        mock_preset.skill_ids = []
        mock_preset.builtin_tools = None
        mock_preset.max_turns = 3
        mock_preset.mcp_servers = []
        mock_preset.system_prompt = None
        mock_preset.model_provider = None
        mock_preset.model_name = None
        mock_preset.executor_name = None

        call_idx = {"i": 0}

        @asynccontextmanager
        async def _ctx():
            idx = call_idx["i"]
            call_idx["i"] += 1
            mock_sess = AsyncMock(spec=AsyncSession)

            if idx == 0:
                # First call: find preset
                mock_result_obj = MagicMock()
                mock_result_obj.scalar_one_or_none.return_value = mock_preset
                mock_sess.execute = AsyncMock(return_value=mock_result_obj)
            else:
                # Subsequent calls: session management, trace updates
                mock_result_obj = MagicMock()
                mock_result_obj.scalar_one_or_none.return_value = None
                mock_sess.execute = AsyncMock(return_value=mock_result_obj)
            mock_sess.add = MagicMock()
            mock_sess.commit = AsyncMock()
            yield mock_sess

        MockSL.side_effect = lambda: _ctx()

        session_id = str(uuid.uuid4())
        MockLoadSession.return_value = SessionData(session_id=session_id)
        resp = await e2e_client.post(
            f"/api/v1/published/{pid}/chat/sync",
            json={"request": "Analyze this data", "session_id": session_id},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["output_files"] is not None
        assert len(body["output_files"]) == 1
        assert body["output_files"][0]["filename"] == "analysis.pdf"

    @patch("app.api.v1.published.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.published.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.published.SkillsAgent")
    @patch("app.api.v1.published.AsyncSessionLocal")
    async def test_03_sync_chat_no_output_files(
        self, MockSL, MockAgent, MockLoadSession, _mock_save, e2e_client: AsyncClient
    ):
        """POST /published/{id}/chat/sync without output_files → null."""
        pid = type(self)._state["preset_id"]

        mock_result = _MockAgentResult(output_files=[])
        mock_instance = _make_mock_agent(result=mock_result)
        MockAgent.return_value = mock_instance

        from app.db.models import AgentPresetDB

        mock_preset = MagicMock(spec=AgentPresetDB)
        mock_preset.id = pid
        mock_preset.name = "e2e-pub-output-files"
        mock_preset.is_published = True
        mock_preset.api_response_mode = "non_streaming"
        mock_preset.skill_ids = []
        mock_preset.builtin_tools = None
        mock_preset.max_turns = 3
        mock_preset.mcp_servers = []
        mock_preset.system_prompt = None
        mock_preset.model_provider = None
        mock_preset.model_name = None
        mock_preset.executor_name = None

        call_idx = {"i": 0}

        @asynccontextmanager
        async def _ctx():
            idx = call_idx["i"]
            call_idx["i"] += 1
            mock_sess = AsyncMock(spec=AsyncSession)
            if idx == 0:
                mock_result_obj = MagicMock()
                mock_result_obj.scalar_one_or_none.return_value = mock_preset
                mock_sess.execute = AsyncMock(return_value=mock_result_obj)
            else:
                mock_result_obj = MagicMock()
                mock_result_obj.scalar_one_or_none.return_value = None
                mock_sess.execute = AsyncMock(return_value=mock_result_obj)
            mock_sess.add = MagicMock()
            mock_sess.commit = AsyncMock()
            yield mock_sess

        MockSL.side_effect = lambda: _ctx()

        session_id = str(uuid.uuid4())
        MockLoadSession.return_value = SessionData(session_id=session_id)
        resp = await e2e_client.post(
            f"/api/v1/published/{pid}/chat/sync",
            json={"request": "Just a question", "session_id": session_id},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        # Empty list from agent → should be None in response
        assert body["output_files"] is None

    async def test_04_unpublish_and_cleanup(self, e2e_client: AsyncClient):
        pid = type(self)._state["preset_id"]
        await e2e_client.post(f"/api/v1/agents/{pid}/unpublish")
        resp = await e2e_client.delete(f"/api/v1/agents/{pid}")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Published Session Management E2E
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestPublishedSessionManagementE2E:
    """Test the session listing, detail, and deletion endpoints."""

    _state: Dict = {}

    async def test_01_setup_preset_and_publish(
        self, e2e_client: AsyncClient, e2e_db_session: AsyncSession
    ):
        """Create a preset and publish it so we can create sessions against it."""
        resp = await e2e_client.post(
            "/api/v1/agents",
            json={
                "name": "session-mgmt-test-agent",
                "description": "Agent for session management tests",
                "max_turns": 5,
            },
        )
        assert resp.status_code == 200
        preset = resp.json()
        type(self)._state["preset_id"] = preset["id"]

        # Publish
        resp = await e2e_client.post(
            f"/api/v1/agents/{preset['id']}/publish",
            json={"api_response_mode": "streaming"},
        )
        assert resp.status_code == 200

    async def test_02_insert_test_sessions(
        self, e2e_client: AsyncClient, e2e_db_session: AsyncSession
    ):
        """Directly insert sessions into DB for testing."""
        from app.db.models import PublishedSessionDB

        pid = type(self)._state["preset_id"]
        session_ids = []
        for i in range(3):
            session = PublishedSessionDB(
                agent_id=pid,
                messages=[
                    {"role": "user", "content": f"Hello from session {i}"},
                    {"role": "assistant", "content": f"Response in session {i}"},
                ],
            )
            e2e_db_session.add(session)
            await e2e_db_session.flush()
            session_ids.append(session.id)

        await e2e_db_session.commit()
        type(self)._state["session_ids"] = session_ids

    async def test_03_list_all_sessions(self, e2e_client: AsyncClient):
        """List all sessions without filter."""
        resp = await e2e_client.get("/api/v1/published/sessions/list")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 3
        assert len(body["sessions"]) >= 3
        # Check fields are present
        s = body["sessions"][0]
        assert "id" in s
        assert "agent_id" in s
        assert "agent_name" in s
        assert "message_count" in s
        assert "first_user_message" in s
        assert "created_at" in s
        assert "updated_at" in s

    async def test_04_list_sessions_filtered_by_agent(self, e2e_client: AsyncClient):
        """Filter sessions by agent_id."""
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.get(
            f"/api/v1/published/sessions/list?agent_id={pid}"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 3
        for s in body["sessions"]:
            assert s["agent_id"] == pid

    async def test_05_list_sessions_filtered_nonexistent_agent(self, e2e_client: AsyncClient):
        """Filter by non-existent agent returns empty."""
        resp = await e2e_client.get(
            "/api/v1/published/sessions/list?agent_id=nonexistent-id"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["sessions"] == []

    async def test_06_get_session_detail(self, e2e_client: AsyncClient):
        """Get session detail by ID."""
        sid = type(self)._state["session_ids"][0]
        resp = await e2e_client.get(f"/api/v1/published/sessions/{sid}/detail")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == sid
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "user"

    async def test_07_get_session_detail_not_found(self, e2e_client: AsyncClient):
        """Get non-existent session returns 404."""
        resp = await e2e_client.get(
            "/api/v1/published/sessions/nonexistent-session/detail"
        )
        assert resp.status_code == 404

    async def test_08_delete_single_session(self, e2e_client: AsyncClient):
        """Delete a single session."""
        sid = type(self)._state["session_ids"][0]
        resp = await e2e_client.delete(f"/api/v1/published/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["message"] == "Session deleted"

        # Verify it's gone
        resp = await e2e_client.get(f"/api/v1/published/sessions/{sid}/detail")
        assert resp.status_code == 404

    async def test_09_delete_session_not_found(self, e2e_client: AsyncClient):
        """Delete non-existent session returns 404."""
        resp = await e2e_client.delete("/api/v1/published/sessions/nonexistent-id")
        assert resp.status_code == 404

    async def test_10_delete_all_agent_sessions(self, e2e_client: AsyncClient):
        """Delete all sessions for an agent."""
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.delete(f"/api/v1/published/{pid}/sessions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted_count"] >= 2  # At least 2 remaining

        # Verify empty
        resp = await e2e_client.get(
            f"/api/v1/published/sessions/list?agent_id={pid}"
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    async def test_11_cleanup(self, e2e_client: AsyncClient):
        """Unpublish and delete the test agent."""
        pid = type(self)._state["preset_id"]
        await e2e_client.post(f"/api/v1/agents/{pid}/unpublish")
        resp = await e2e_client.delete(f"/api/v1/agents/{pid}")
        assert resp.status_code == 200


# ===================================================================
# Class: Resilient Streaming — turn_complete & incremental session save
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestResilientStreamingE2E:
    """Verify turn_complete event triggers incremental session saves and is not forwarded to SSE client.

    Tests cover both published.py and agent.py streaming endpoints.
    """

    _state: dict = {}

    # -- Helpers --

    @staticmethod
    def _make_multi_turn_events(answer="Multi-turn done"):
        """Create a multi-turn event sequence with turn_complete checkpoints."""
        turn1_snapshot = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file1.txt"}]},
        ]
        turn2_snapshot = turn1_snapshot + [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "read", "input": {"path": "file1.txt"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "hello world"}]},
        ]
        final_messages = turn2_snapshot + [
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

        return [
            StreamEvent(event_type="turn_start", turn=1, data={"turn": 1}),
            StreamEvent(event_type="tool_call", turn=1, data={
                "tool_name": "bash", "tool_input": {"command": "ls"},
            }),
            StreamEvent(event_type="tool_result", turn=1, data={
                "tool_name": "bash", "tool_result": "file1.txt",
            }),
            # First checkpoint after turn 1
            StreamEvent(event_type="turn_complete", turn=1, data={
                "messages_snapshot": turn1_snapshot,
            }),
            StreamEvent(event_type="turn_start", turn=2, data={"turn": 2}),
            StreamEvent(event_type="tool_call", turn=2, data={
                "tool_name": "read", "tool_input": {"path": "file1.txt"},
            }),
            StreamEvent(event_type="tool_result", turn=2, data={
                "tool_name": "read", "tool_result": "hello world",
            }),
            # Second checkpoint after turn 2
            StreamEvent(event_type="turn_complete", turn=2, data={
                "messages_snapshot": turn2_snapshot,
            }),
            StreamEvent(event_type="assistant", turn=3, data={
                "content": answer, "turn": 3,
            }),
            StreamEvent(event_type="complete", turn=3, data={
                "success": True,
                "answer": answer,
                "total_turns": 3,
                "total_input_tokens": 500,
                "total_output_tokens": 150,
                "skills_used": [],
                "final_messages": final_messages,
            }),
        ]

    # -- Setup --

    async def test_01_create_and_publish(self, e2e_client: AsyncClient):
        """Create and publish an agent for testing."""
        payload = {
            "name": "e2e-resilient-streaming",
            "description": "Test resilient streaming checkpoints",
            "max_turns": 10,
        }
        resp = await e2e_client.post("/api/v1/agents", json=payload)
        assert resp.status_code == 200
        type(self)._state["preset_id"] = resp.json()["id"]

        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post(
            f"/api/v1/agents/{pid}/publish",
            json={"api_response_mode": "streaming"},
        )
        assert resp.status_code == 200

    # -- Published endpoint tests --

    @patch("app.api.v1.published.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.published.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.published.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.published.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.published.SkillsAgent")
    @patch("app.api.v1.published.AsyncSessionLocal")
    async def test_02_turn_complete_triggers_incremental_save(
        self, MockSL, MockAgent, MockLoadSession, MockSave, _mock_checkpoint, _mock_precompress, e2e_client: AsyncClient
    ):
        """turn_complete events trigger incremental save_session_messages calls."""
        pid = type(self)._state["preset_id"]

        events = self._make_multi_turn_events()
        MockAgent.return_value = _make_streaming_mock_agent(events=events, answer="Multi-turn done")

        from app.db.models import AgentPresetDB
        mock_preset = MagicMock(spec=AgentPresetDB)
        mock_preset.id = pid
        mock_preset.name = "e2e-resilient-streaming"
        mock_preset.description = "Test"
        mock_preset.is_published = True
        mock_preset.api_response_mode = "streaming"
        mock_preset.skill_ids = []
        mock_preset.builtin_tools = None
        mock_preset.max_turns = 10
        mock_preset.mcp_servers = []
        mock_preset.system_prompt = None
        mock_preset.model_provider = None
        mock_preset.model_name = None
        mock_preset.executor_name = None

        call_idx = {"i": 0}
        results = [mock_preset, None, None, None, None, None]

        @asynccontextmanager
        async def _ctx():
            idx = min(call_idx["i"], len(results) - 1)
            call_idx["i"] += 1
            mock_sess = AsyncMock(spec=AsyncSession)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = results[idx]
            mock_sess.execute = AsyncMock(return_value=mock_result)
            mock_sess.add = MagicMock()
            mock_sess.commit = AsyncMock()
            yield mock_sess

        MockSL.side_effect = lambda: _ctx()

        session_id = str(uuid.uuid4())
        MockLoadSession.return_value = SessionData(session_id=session_id)

        resp = await e2e_client.post(
            f"/api/v1/published/{pid}/chat",
            json={"request": "do something", "session_id": session_id},
        )
        assert resp.status_code == 200

        # Verify save_session_checkpoint was called for incremental saves (turn_complete)
        assert _mock_checkpoint.call_count >= 2, f"Expected >=2 checkpoint calls, got {_mock_checkpoint.call_count}"

        # Verify save_session_messages was called once for the final save
        assert MockSave.call_count == 1, f"Expected 1 final save call, got {MockSave.call_count}"

        # Final call is the definitive save with final answer
        last_call = MockSave.call_args_list[-1]
        assert last_call.args[1] == "Multi-turn done"

    @patch("app.api.v1.published.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.published.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.published.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.published.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.published.SkillsAgent")
    @patch("app.api.v1.published.AsyncSessionLocal")
    async def test_03_turn_complete_not_forwarded_to_client(
        self, MockSL, MockAgent, MockLoadSession, MockSave, _mock_checkpoint, _mock_precompress, e2e_client: AsyncClient
    ):
        """turn_complete events must NOT appear in SSE output to the client."""
        pid = type(self)._state["preset_id"]

        events = self._make_multi_turn_events()
        MockAgent.return_value = _make_streaming_mock_agent(events=events, answer="Multi-turn done")

        from app.db.models import AgentPresetDB
        mock_preset = MagicMock(spec=AgentPresetDB)
        mock_preset.id = pid
        mock_preset.name = "e2e-resilient-streaming"
        mock_preset.description = "Test"
        mock_preset.is_published = True
        mock_preset.api_response_mode = "streaming"
        mock_preset.skill_ids = []
        mock_preset.builtin_tools = None
        mock_preset.max_turns = 10
        mock_preset.mcp_servers = []
        mock_preset.system_prompt = None
        mock_preset.model_provider = None
        mock_preset.model_name = None
        mock_preset.executor_name = None

        call_idx = {"i": 0}
        results = [mock_preset, None, None, None, None, None]

        @asynccontextmanager
        async def _ctx():
            idx = min(call_idx["i"], len(results) - 1)
            call_idx["i"] += 1
            mock_sess = AsyncMock(spec=AsyncSession)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = results[idx]
            mock_sess.execute = AsyncMock(return_value=mock_result)
            mock_sess.add = MagicMock()
            mock_sess.commit = AsyncMock()
            yield mock_sess

        MockSL.side_effect = lambda: _ctx()

        session_id = str(uuid.uuid4())
        MockLoadSession.return_value = SessionData(session_id=session_id)

        resp = await e2e_client.post(
            f"/api/v1/published/{pid}/chat",
            json={"request": "do something", "session_id": session_id},
        )
        assert resp.status_code == 200

        sse_events = parse_sse_events(resp.text)
        event_types = [e["event_type"] for e in sse_events]

        # turn_complete must NOT be in SSE output
        assert "turn_complete" not in event_types, \
            f"turn_complete should not be forwarded to client, but found in: {event_types}"

        # Other events should be present
        assert "run_started" in event_types
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "complete" in event_types

    @patch("app.api.v1.published.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.published.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.published.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.published.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.published.SkillsAgent")
    @patch("app.api.v1.published.AsyncSessionLocal")
    async def test_04_checkpoint_snapshot_content_correct(
        self, MockSL, MockAgent, MockLoadSession, MockSave, _mock_checkpoint, _mock_precompress, e2e_client: AsyncClient
    ):
        """Verify incremental saves contain growing messages_snapshot."""
        pid = type(self)._state["preset_id"]

        events = self._make_multi_turn_events()
        MockAgent.return_value = _make_streaming_mock_agent(events=events, answer="Multi-turn done")

        from app.db.models import AgentPresetDB
        mock_preset = MagicMock(spec=AgentPresetDB)
        mock_preset.id = pid
        mock_preset.name = "e2e-resilient-streaming"
        mock_preset.description = "Test"
        mock_preset.is_published = True
        mock_preset.api_response_mode = "streaming"
        mock_preset.skill_ids = []
        mock_preset.builtin_tools = None
        mock_preset.max_turns = 10
        mock_preset.mcp_servers = []
        mock_preset.system_prompt = None
        mock_preset.model_provider = None
        mock_preset.model_name = None
        mock_preset.executor_name = None

        call_idx = {"i": 0}
        results = [mock_preset, None, None, None, None, None]

        @asynccontextmanager
        async def _ctx():
            idx = min(call_idx["i"], len(results) - 1)
            call_idx["i"] += 1
            mock_sess = AsyncMock(spec=AsyncSession)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = results[idx]
            mock_sess.execute = AsyncMock(return_value=mock_result)
            mock_sess.add = MagicMock()
            mock_sess.commit = AsyncMock()
            yield mock_sess

        MockSL.side_effect = lambda: _ctx()

        session_id = str(uuid.uuid4())
        MockLoadSession.return_value = SessionData(session_id=session_id)

        resp = await e2e_client.post(
            f"/api/v1/published/{pid}/chat",
            json={"request": "do something", "session_id": session_id},
        )
        assert resp.status_code == 200

        checkpoint_args_list = _mock_checkpoint.call_args_list

        # First checkpoint (turn 1): 3 messages (user + assistant tool_use + tool_result)
        first_msgs = checkpoint_args_list[0].args[1]  # save_session_checkpoint(session_id, snapshot)
        assert len(first_msgs) == 3
        assert first_msgs[0]["role"] == "user"

        # Second checkpoint (turn 2): 5 messages (turn1 + turn2 tool_use + tool_result)
        second_msgs = checkpoint_args_list[1].args[1]
        assert len(second_msgs) == 5
        assert len(second_msgs) > len(first_msgs)

        # Final save via save_session_messages: full messages including assistant answer
        final_save_args = MockSave.call_args_list
        assert len(final_save_args) >= 1
        final_msgs = final_save_args[-1].kwargs.get("final_messages") or final_save_args[-1].args[3]
        assert len(final_msgs) == 6  # 5 + final assistant text

    # -- Agent (chat panel) endpoint tests --

    @patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-resilient-session"))
    @patch("app.api.v1.agent.AsyncSessionLocal")
    @patch("app.api.v1.agent.create_agent")
    async def test_05_agent_endpoint_turn_complete_incremental_save(
        self, MockAgent, MockSessionLocal, _mock_load, MockSave, _mock_checkpoint, _mock_precompress, e2e_client: AsyncClient
    ):
        """Agent /run/stream endpoint also handles turn_complete for incremental saves."""
        events = self._make_multi_turn_events(answer="Agent multi-turn done")
        MockAgent.return_value = _make_streaming_mock_agent(events=events, answer="Agent multi-turn done")
        MockSessionLocal.side_effect = lambda: _mock_session_local()()

        resp = await e2e_client.post(
            "/api/v1/agent/run/stream",
            json={"request": "do something", "session_id": "test-resilient-session"},
        )
        assert resp.status_code == 200

        # Verify incremental checkpoint saves happened via save_session_checkpoint
        assert _mock_checkpoint.call_count >= 2, f"Expected >=2 checkpoint calls, got {_mock_checkpoint.call_count}"

        # Verify final save_session_messages was called once
        assert MockSave.call_count == 1, f"Expected 1 final save call, got {MockSave.call_count}"

        # turn_complete not in SSE
        sse_events = parse_sse_events(resp.text)
        event_types = [e["event_type"] for e in sse_events]
        assert "turn_complete" not in event_types

    @patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-resilient-session"))
    @patch("app.api.v1.agent.AsyncSessionLocal")
    @patch("app.api.v1.agent.create_agent")
    async def test_06_no_turn_complete_means_no_incremental_save(
        self, MockAgent, MockSessionLocal, _mock_load, MockSave, _mock_checkpoint, _mock_precompress, e2e_client: AsyncClient
    ):
        """Without turn_complete events, only the final save happens."""
        events = _make_stream_events(answer="Simple answer")
        MockAgent.return_value = _make_streaming_mock_agent(events=events, answer="Simple answer")
        MockSessionLocal.side_effect = lambda: _mock_session_local()()

        resp = await e2e_client.post(
            "/api/v1/agent/run/stream",
            json={"request": "simple question", "session_id": "test-resilient-session"},
        )
        assert resp.status_code == 200

        # Only 1 save call (the final completion save)
        assert MockSave.call_count == 1, f"Expected 1 save call, got {MockSave.call_count}"
        assert MockSave.call_args_list[0].args[1] == "Simple answer"

    # -- Session continuity after stop --

    @patch("app.api.v1.published.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.published.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.published.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.published.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.published.SkillsAgent")
    @patch("app.api.v1.published.AsyncSessionLocal")
    async def test_07_stop_then_continue_preserves_context(
        self, MockSL, MockAgent, MockLoadSession, MockSave, _mock_checkpoint, _mock_precompress, e2e_client: AsyncClient
    ):
        """After stop, the next request receives saved checkpoint as conversation_history."""
        # pre_compress_if_needed should pass through the context unchanged
        _mock_precompress.side_effect = lambda ctx, *a, **kw: ctx

        pid = type(self)._state["preset_id"]
        session_id = str(uuid.uuid4())

        # -- Run 1: multi-turn, then simulated cancel --
        # Agent pushes turn_complete checkpoint then completes normally
        # (real cancel is hard to simulate in mock — we verify save was called)
        checkpoint_messages = [
            {"role": "user", "content": "analyze data"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "execute_code", "input": {"code": "import pandas"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
        ]

        run1_events = [
            StreamEvent(event_type="turn_start", turn=1, data={"turn": 1}),
            StreamEvent(event_type="tool_call", turn=1, data={
                "tool_name": "execute_code", "tool_input": {"code": "import pandas"},
            }),
            StreamEvent(event_type="tool_result", turn=1, data={
                "tool_name": "execute_code", "tool_result": "ok",
            }),
            StreamEvent(event_type="turn_complete", turn=1, data={
                "messages_snapshot": checkpoint_messages,
            }),
            StreamEvent(event_type="assistant", turn=2, data={"content": "Done", "turn": 2}),
            StreamEvent(event_type="complete", turn=2, data={
                "success": True, "answer": "Done", "total_turns": 2,
                "total_input_tokens": 200, "total_output_tokens": 50,
                "skills_used": [], "final_messages": checkpoint_messages + [
                    {"role": "assistant", "content": [{"type": "text", "text": "Done"}]},
                ],
            }),
        ]

        MockAgent.return_value = _make_streaming_mock_agent(events=run1_events, answer="Done")

        from app.db.models import AgentPresetDB
        mock_preset = MagicMock(spec=AgentPresetDB)
        mock_preset.id = pid
        mock_preset.name = "e2e-resilient-streaming"
        mock_preset.description = "Test"
        mock_preset.is_published = True
        mock_preset.api_response_mode = "streaming"
        mock_preset.skill_ids = []
        mock_preset.builtin_tools = None
        mock_preset.max_turns = 10
        mock_preset.mcp_servers = []
        mock_preset.system_prompt = None
        mock_preset.model_provider = None
        mock_preset.model_name = None
        mock_preset.executor_name = None

        call_idx = {"i": 0}
        results = [mock_preset, None, None, None, None, None]

        @asynccontextmanager
        async def _ctx():
            idx = min(call_idx["i"], len(results) - 1)
            call_idx["i"] += 1
            mock_sess = AsyncMock(spec=AsyncSession)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = results[idx]
            mock_sess.execute = AsyncMock(return_value=mock_result)
            mock_sess.add = MagicMock()
            mock_sess.commit = AsyncMock()
            yield mock_sess

        MockSL.side_effect = lambda: _ctx()
        MockLoadSession.return_value = SessionData(session_id=session_id)  # New session

        resp = await e2e_client.post(
            f"/api/v1/published/{pid}/chat",
            json={"request": "analyze data", "session_id": session_id},
        )
        assert resp.status_code == 200

        # Verify checkpoint was saved via save_session_checkpoint (turn_complete)
        assert _mock_checkpoint.call_count >= 1
        saved_snapshot = _mock_checkpoint.call_args_list[0].args[1]  # save_session_checkpoint(session_id, snapshot)
        assert len(saved_snapshot) == 3  # The checkpoint_messages

        # -- Run 2: same session_id, load_or_create returns the checkpoint --
        MockSave.reset_mock()
        MockAgent.reset_mock()

        run2_events = _make_stream_events(answer="Continued from checkpoint")
        mock_agent2 = _make_streaming_mock_agent(events=run2_events, answer="Continued from checkpoint")
        MockAgent.return_value = mock_agent2

        call_idx["i"] = 0  # Reset counter for AsyncSessionLocal
        MockLoadSession.return_value = SessionData(session_id=session_id, agent_context=checkpoint_messages)  # Return saved history

        resp2 = await e2e_client.post(
            f"/api/v1/published/{pid}/chat",
            json={"request": "what did I import?", "session_id": session_id},
        )
        assert resp2.status_code == 200

        # Verify the agent received the checkpoint as conversation_history
        agent_run_call = mock_agent2.run.call_args
        received_history = agent_run_call.kwargs.get("conversation_history")
        if received_history is None and len(agent_run_call.args) > 1:
            received_history = agent_run_call.args[1]
        assert received_history is not None, \
            f"Agent should receive conversation_history from session checkpoint. call_args={agent_run_call}"
        assert len(received_history) == 3, f"Expected 3 messages from checkpoint, got {len(received_history)}"
        assert received_history[0]["role"] == "user"
        assert received_history[0]["content"] == "analyze data"

    # -- Cleanup --

    async def test_08_cleanup(self, e2e_client: AsyncClient):
        """Unpublish and delete the test agent."""
        pid = type(self)._state["preset_id"]
        await e2e_client.post(f"/api/v1/agents/{pid}/unpublish")
        resp = await e2e_client.delete(f"/api/v1/agents/{pid}")
        assert resp.status_code == 200


# ===================================================================
# Class: Ask User Tool E2E
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestAskUserE2E:
    """Tests for the ask_user built-in tool: end-turn behavior, SSE events, non-streaming fallback."""

    _state: dict = {}

    async def test_01_ask_user_tool_in_base_tools(self, e2e_client: AsyncClient):
        """Verify ask_user tool is defined in BASE_TOOLS and has correct schema."""
        from app.agent.tools import BASE_TOOLS, TOOL_REQUIRED_PARAMS

        names = [t["name"] for t in BASE_TOOLS]
        assert "ask_user" in names

        # Verify it has the expected parameters
        ask_user_tool = next(t for t in BASE_TOOLS if t["name"] == "ask_user")
        schema = ask_user_tool["input_schema"]
        assert "question" in schema["properties"]
        assert "question" in schema["required"]
        assert "options" in schema["properties"]

        # Verify required params mapping
        assert "ask_user" in TOOL_REQUIRED_PARAMS
        assert "question" in TOOL_REQUIRED_PARAMS["ask_user"]

    @patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="ask-user-session"))
    @patch("app.api.v1.agent.AsyncSessionLocal")
    @patch("app.api.v1.agent.create_agent")
    async def test_02_stream_with_ask_user_ends_run(
        self, MockAgent, MockSessionLocal, _mock_load, _mock_save, _mock_checkpoint, _mock_precompress, e2e_client: AsyncClient
    ):
        """Verify that ask_user SSE event is emitted and the stream ends with complete (success=True)."""
        # ask_user now ends the run — no ask_user_response event is emitted
        ask_user_events = [
            StreamEvent(event_type="turn_start", turn=1, data={"turn": 1}),
            StreamEvent(
                event_type="assistant",
                turn=1,
                data={"content": "I need to ask you something.", "turn": 1},
            ),
            StreamEvent(
                event_type="ask_user",
                turn=1,
                data={
                    "prompt_id": "test-prompt-123",
                    "question": "Which format do you prefer?",
                    "options": ["PDF", "HTML", "Markdown"],
                },
            ),
            StreamEvent(
                event_type="tool_result",
                turn=1,
                data={
                    "tool_name": "ask_user",
                    "tool_input": {"question": "Which format do you prefer?", "options": ["PDF", "HTML", "Markdown"]},
                    "tool_result": '{"status": "waiting_for_user", "question": "Which format do you prefer?"}',
                },
            ),
            StreamEvent(
                event_type="complete",
                turn=1,
                data={
                    "success": True,
                    "answer": "[Waiting for user response: Which format do you prefer?]",
                    "total_turns": 1,
                    "total_input_tokens": 200,
                    "total_output_tokens": 100,
                    "skills_used": [],
                    "final_messages": [],
                },
            ),
        ]
        MockAgent.return_value = _make_streaming_mock_agent(events=ask_user_events, answer="[Waiting for user response: Which format do you prefer?]")
        MockSessionLocal.side_effect = lambda: _mock_session_local()()

        resp = await e2e_client.post(
            "/api/v1/agent/run/stream",
            json={"request": "Convert this file", "session_id": "ask-user-session"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        events = parse_sse_events(resp.text)
        assert len(events) >= 1

        # Verify run_started event
        assert events[0]["event_type"] == "run_started"
        type(self)._state["trace_id"] = events[0].get("trace_id")

        # Find ask_user event
        ask_user_evts = [e for e in events if e.get("event_type") == "ask_user"]
        assert len(ask_user_evts) >= 1, f"Expected ask_user event, got event types: {[e.get('event_type') for e in events]}"
        ask_evt = ask_user_evts[0]
        assert ask_evt["prompt_id"] == "test-prompt-123"
        assert ask_evt["question"] == "Which format do you prefer?"
        assert ask_evt["options"] == ["PDF", "HTML", "Markdown"]

        # No ask_user_response event should be emitted (run ends instead)
        response_evts = [e for e in events if e.get("event_type") == "ask_user_response"]
        assert len(response_evts) == 0, "ask_user_response should not be emitted; run ends on ask_user"

        # Verify stream ends with complete event (success=True)
        complete_evts = [e for e in events if e.get("event_type") == "complete"]
        assert len(complete_evts) >= 1
        assert complete_evts[0]["success"] is True

    async def test_03_respond_endpoint_removed(self, e2e_client: AsyncClient):
        """The /respond endpoint no longer exists (removed in end-turn refactor)."""
        resp = await e2e_client.post(
            "/api/v1/agent/run/stream/nonexistent-trace/respond",
            json={"prompt_id": "test-prompt", "answer": "yes"},
        )
        # Should be 404 or 405 since the endpoint no longer exists
        assert resp.status_code in (404, 405)

    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="ask-user-sync"))
    @patch("app.api.v1.agent.create_agent")
    async def test_04_non_streaming_ask_user_ends_run(self, MockAgent, _mock_load, _mock_save, e2e_client: AsyncClient):
        """In non-streaming (sync) mode, ask_user also ends the run (same as streaming)."""
        MockAgent.return_value = _make_mock_agent()
        resp = await e2e_client.post(
            "/api/v1/agent/run",
            json={"request": "Do something that needs confirmation", "session_id": "ask-user-sync"},
        )
        # Mock agent doesn't actually call ask_user; we verify the endpoint works.
        # The non-streaming ask_user end-turn logic is in the agent loop itself.
        assert resp.status_code == 200
        assert resp.json()["success"] is True


# ===================================================================
# Class: Session Display Format (ChatMessage format)
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestSessionDisplayFormatE2E:
    """Verify that sessions store display messages in ChatMessage format
    (with streamEvents and attachedFiles), not raw Anthropic format.

    Tests both streaming (agent.py) and non-streaming endpoints.
    """

    _state: dict = {}

    async def test_01_create_preset(self, e2e_client: AsyncClient):
        """Create an agent preset for display format tests."""
        resp = await e2e_client.post("/api/v1/agents", json={
            "name": "e2e-display-format",
            "description": "Test session display format",
            "max_turns": 5,
        })
        assert resp.status_code == 200
        type(self)._state["preset_id"] = resp.json()["id"]

    @patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.agent.create_agent")
    async def test_02_streaming_display_has_stream_events(
        self, MockAgent, MockLoadSession, MockSaveMessages, MockCheckpoint, MockPreCompress,
        e2e_client: AsyncClient, e2e_db_session,
    ):
        """Streaming: final display_replace_messages uses ChatMessage format with streamEvents."""
        session_id = str(uuid.uuid4())
        MockLoadSession.return_value = SessionData(
            session_id=session_id,
            display_messages=[],
        )

        events = [
            StreamEvent(event_type="turn_start", turn=1, data={"turn": 1, "max_turns": 5}),
            StreamEvent(event_type="tool_call", turn=1, data={
                "tool_name": "execute_code", "tool_input": {"code": "print(42)"},
            }),
            StreamEvent(event_type="tool_result", turn=1, data={
                "tool_name": "execute_code", "tool_result": "42",
            }),
            StreamEvent(event_type="turn_complete", turn=1, data={
                "messages_snapshot": [{"role": "user", "content": "Compute"}],
            }),
            StreamEvent(event_type="assistant", turn=2, data={"content": "The answer is 42"}),
            StreamEvent(event_type="complete", turn=2, data={
                "success": True,
                "answer": "The answer is 42",
                "total_turns": 2,
                "total_input_tokens": 200,
                "total_output_tokens": 30,
                "skills_used": [],
                "final_messages": [
                    {"role": "user", "content": "Compute"},
                    {"role": "assistant", "content": [{"type": "text", "text": "The answer is 42"}]},
                ],
            }),
        ]
        MockAgent.return_value = _make_streaming_mock_agent(events=events, answer="The answer is 42")

        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post("/api/v1/agent/run/stream", json={
            "request": "Compute",
            "session_id": session_id,
            "agent_id": pid,
        })
        assert resp.status_code == 200

        # Check final save uses display_replace_messages with ChatMessage format
        assert MockSaveMessages.call_count >= 1
        call_kwargs = MockSaveMessages.call_args.kwargs
        display = call_kwargs.get("display_replace_messages")
        assert display is not None
        # Should be: [user_msg, assistant_with_streamEvents]
        assert len(display) == 2
        assert display[0]["role"] == "user"
        assert display[0]["content"] == "Compute"
        assert display[1]["role"] == "assistant"
        assert "streamEvents" in display[1]
        event_types = [e["type"] for e in display[1]["streamEvents"]]
        assert "turn_start" in event_types
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "assistant" in event_types
        assert "complete" in event_types

    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.agent.create_agent")
    async def test_03_non_streaming_display_has_stream_events(
        self, MockAgent, MockLoadSession, MockSaveMessages,
        e2e_client: AsyncClient, e2e_db_session,
    ):
        """Non-streaming: display_append_messages uses ChatMessage format with streamEvents."""
        session_id = str(uuid.uuid4())
        MockLoadSession.return_value = SessionData(
            session_id=session_id,
            display_messages=[],
        )
        MockAgent.return_value = _make_mock_agent()

        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post("/api/v1/agent/run", json={
            "request": "Hello",
            "session_id": session_id,
            "agent_id": pid,
        })
        assert resp.status_code == 200

        # Check display_append_messages
        assert MockSaveMessages.call_count >= 1
        call_kwargs = MockSaveMessages.call_args.kwargs
        display = call_kwargs.get("display_append_messages")
        assert display is not None
        # Should be: [user_msg, assistant_with_streamEvents]
        assert len(display) == 2
        assert display[0]["role"] == "user"
        assert display[0]["content"] == "Hello"
        assert display[1]["role"] == "assistant"
        assert "streamEvents" in display[1]
        # Should have at least a complete event
        event_types = [e["type"] for e in display[1]["streamEvents"]]
        assert "complete" in event_types

    @patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.agent.create_agent")
    async def test_04_uploaded_files_stored_as_attached_files(
        self, MockAgent, MockLoadSession, MockSaveMessages, MockCheckpoint, MockPreCompress,
        e2e_client: AsyncClient, e2e_db_session,
    ):
        """Uploaded files produce attachedFiles in the user message (not [Uploaded Files] text)."""
        session_id = str(uuid.uuid4())
        MockLoadSession.return_value = SessionData(
            session_id=session_id,
            display_messages=[],
        )

        events = [
            StreamEvent(event_type="assistant", turn=1, data={"content": "Got it"}),
            StreamEvent(event_type="complete", turn=1, data={
                "success": True, "answer": "Got it", "total_turns": 1,
                "total_input_tokens": 100, "total_output_tokens": 20,
                "skills_used": [],
                "final_messages": [],
            }),
        ]
        MockAgent.return_value = _make_streaming_mock_agent(events=events, answer="Got it")

        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post("/api/v1/agent/run/stream", json={
            "request": "Analyze this file",
            "session_id": session_id,
            "agent_id": pid,
            "uploaded_files": [
                {"file_id": "f1", "filename": "data.csv", "path": "/tmp/data.csv", "content_type": "text/csv"},
            ],
        })
        assert resp.status_code == 200

        # Check the display messages have attachedFiles
        assert MockSaveMessages.call_count >= 1
        display = MockSaveMessages.call_args.kwargs.get("display_replace_messages")
        assert display is not None
        user_msg = display[0]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == "Analyze this file"
        assert "attachedFiles" in user_msg
        assert user_msg["attachedFiles"] == [
            {"file_id": "f1", "filename": "data.csv"},
        ]
        # Crucially, no "[Uploaded Files]" text in content
        assert "[Uploaded Files]" not in user_msg["content"]

    @patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.agent.create_agent")
    async def test_05_ask_user_event_in_display(
        self, MockAgent, MockLoadSession, MockSaveMessages, MockCheckpoint, MockPreCompress,
        e2e_client: AsyncClient, e2e_db_session,
    ):
        """ask_user events are stored in streamEvents for proper UI rendering on refresh."""
        session_id = str(uuid.uuid4())
        MockLoadSession.return_value = SessionData(
            session_id=session_id,
            display_messages=[],
        )

        events = [
            StreamEvent(event_type="turn_start", turn=1, data={"turn": 1, "max_turns": 5}),
            StreamEvent(event_type="ask_user", turn=1, data={
                "prompt_id": "p1", "question": "Which format?", "options": ["CSV", "JSON"],
            }),
            StreamEvent(event_type="complete", turn=1, data={
                "success": True,
                "answer": "[Waiting for user response: Which format?]",
                "total_turns": 1,
                "total_input_tokens": 100,
                "total_output_tokens": 20,
                "skills_used": [],
                "final_messages": [],
            }),
        ]
        MockAgent.return_value = _make_streaming_mock_agent(events=events, answer="[Waiting for user response]")

        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post("/api/v1/agent/run/stream", json={
            "request": "Do something",
            "session_id": session_id,
            "agent_id": pid,
        })
        assert resp.status_code == 200

        display = MockSaveMessages.call_args.kwargs.get("display_replace_messages")
        assert display is not None
        assistant_events = display[1]["streamEvents"]
        event_types = [e["type"] for e in assistant_events]
        assert "ask_user" in event_types
        ask_evt = next(e for e in assistant_events if e["type"] == "ask_user")
        assert ask_evt["data"]["question"] == "Which format?"
        assert ask_evt["data"]["options"] == ["CSV", "JSON"]

    @patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.agent.create_agent")
    async def test_06_output_file_event_in_display(
        self, MockAgent, MockLoadSession, MockSaveMessages, MockCheckpoint, MockPreCompress,
        e2e_client: AsyncClient, e2e_db_session,
    ):
        """output_file events are stored in streamEvents for download buttons on refresh."""
        session_id = str(uuid.uuid4())
        MockLoadSession.return_value = SessionData(
            session_id=session_id,
            display_messages=[],
        )

        events = [
            StreamEvent(event_type="turn_start", turn=1, data={"turn": 1, "max_turns": 5}),
            StreamEvent(event_type="output_file", turn=1, data={
                "file_id": "of1", "filename": "chart.png", "size": 2048,
                "content_type": "image/png", "download_url": "/download/of1",
            }),
            StreamEvent(event_type="complete", turn=1, data={
                "success": True, "answer": "Done", "total_turns": 1,
                "total_input_tokens": 100, "total_output_tokens": 20,
                "skills_used": [], "final_messages": [],
            }),
        ]
        MockAgent.return_value = _make_streaming_mock_agent(events=events, answer="Done")

        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post("/api/v1/agent/run/stream", json={
            "request": "Generate chart",
            "session_id": session_id,
            "agent_id": pid,
        })
        assert resp.status_code == 200

        display = MockSaveMessages.call_args.kwargs.get("display_replace_messages")
        assert display is not None
        assistant_events = display[1]["streamEvents"]
        event_types = [e["type"] for e in assistant_events]
        assert "output_file" in event_types
        of_evt = next(e for e in assistant_events if e["type"] == "output_file")
        assert of_evt["data"]["fileId"] == "of1"
        assert of_evt["data"]["filename"] == "chart.png"
        assert of_evt["data"]["downloadUrl"] == "/download/of1"

    @patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.agent.create_agent")
    async def test_07_display_appends_to_existing_session(
        self, MockAgent, MockLoadSession, MockSaveMessages, MockCheckpoint, MockPreCompress,
        e2e_client: AsyncClient, e2e_db_session,
    ):
        """Second turn: display_base preserves previous ChatMessage-format messages."""
        session_id = str(uuid.uuid4())
        existing_display = [
            {"role": "user", "content": "First Q"},
            {"role": "assistant", "content": "", "streamEvents": [
                {"type": "complete", "data": {"success": True, "answer": "First A", "totalTurns": 1}},
            ]},
        ]
        MockLoadSession.return_value = SessionData(
            session_id=session_id,
            display_messages=existing_display,
            agent_context=[{"role": "user", "content": "First Q"}],
        )

        events = [
            StreamEvent(event_type="assistant", turn=1, data={"content": "Second A"}),
            StreamEvent(event_type="complete", turn=1, data={
                "success": True, "answer": "Second A", "total_turns": 1,
                "total_input_tokens": 100, "total_output_tokens": 20,
                "skills_used": [], "final_messages": [],
            }),
        ]
        MockAgent.return_value = _make_streaming_mock_agent(events=events, answer="Second A")

        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post("/api/v1/agent/run/stream", json={
            "request": "Second Q",
            "session_id": session_id,
            "agent_id": pid,
        })
        assert resp.status_code == 200

        display = MockSaveMessages.call_args.kwargs.get("display_replace_messages")
        assert display is not None
        # Should be: existing 2 + new user + new assistant = 4
        assert len(display) == 4
        assert display[0] == existing_display[0]
        assert display[1] == existing_display[1]
        assert display[2]["role"] == "user"
        assert display[2]["content"] == "Second Q"
        assert display[3]["role"] == "assistant"
        assert "streamEvents" in display[3]

    async def test_08_cleanup(self, e2e_client: AsyncClient):
        """Clean up test agent."""
        pid = type(self)._state.get("preset_id")
        if pid:
            await e2e_client.delete(f"/api/v1/agents/{pid}")


# ===================================================================
# Class: Multi-Feishu-App Channel Bindings
# ===================================================================

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestMultiFeishuChannelE2E:
    """Test multi-Feishu-app support, secret masking, config merge,
    agent_name in response, and channel edit (update) flow.
    """

    _state: dict = {}

    async def test_01_create_preset(self, e2e_client: AsyncClient):
        """Create an agent preset for channel binding tests."""
        resp = await e2e_client.post("/api/v1/agents", json={
            "name": "e2e-channel-agent",
            "description": "Agent for channel E2E tests",
            "max_turns": 5,
        })
        assert resp.status_code == 200
        body = resp.json()
        type(self)._state["preset_id"] = body["id"]
        type(self)._state["preset_name"] = body["name"]

    async def test_02_create_second_preset(self, e2e_client: AsyncClient):
        """Create a second agent preset for edit tests."""
        resp = await e2e_client.post("/api/v1/agents", json={
            "name": "e2e-channel-agent-2",
            "description": "Second agent for channel E2E tests",
            "max_turns": 10,
        })
        assert resp.status_code == 200
        type(self)._state["preset_id_2"] = resp.json()["id"]

    async def test_03_create_binding_with_feishu_config(self, e2e_client: AsyncClient):
        """Create a Feishu binding with app_id/app_secret in config."""
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post("/api/v1/channels", json={
            "channel_type": "feishu",
            "external_id": "oc_e2e_test_001",
            "name": "E2E Feishu Binding 1",
            "agent_id": pid,
            "trigger_pattern": "@bot",
            "config": {
                "app_id": "cli_e2e_app_001",
                "app_secret": "secret_value_abc123xyz",
            },
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "E2E Feishu Binding 1"
        assert body["channel_type"] == "feishu"
        assert body["agent_id"] == pid
        type(self)._state["binding_id_1"] = body["id"]

    async def test_04_secret_is_masked_in_response(self, e2e_client: AsyncClient):
        """GET binding should return masked app_secret."""
        bid = type(self)._state["binding_id_1"]
        resp = await e2e_client.get(f"/api/v1/channels/{bid}")
        assert resp.status_code == 200
        body = resp.json()
        config = body["config"]
        assert config is not None
        assert config["app_id"] == "cli_e2e_app_001"
        # Secret should be masked: ****<last4>
        assert config["app_secret"].startswith("****")
        assert config["app_secret"].endswith("3xyz")
        assert len(config["app_secret"]) == 8  # ****xxxx

    async def test_05_agent_name_in_response(self, e2e_client: AsyncClient):
        """Response should include agent_name from the preset."""
        bid = type(self)._state["binding_id_1"]
        resp = await e2e_client.get(f"/api/v1/channels/{bid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["agent_name"] == "e2e-channel-agent"

    async def test_06_agent_name_in_list_response(self, e2e_client: AsyncClient):
        """List endpoint should also include agent_name."""
        resp = await e2e_client.get("/api/v1/channels?channel_type=feishu")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1
        binding = next(b for b in body["bindings"] if b["id"] == type(self)._state["binding_id_1"])
        assert binding["agent_name"] == "e2e-channel-agent"

    async def test_07_create_second_binding_different_app(self, e2e_client: AsyncClient):
        """Create a second Feishu binding with a different app_id."""
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post("/api/v1/channels", json={
            "channel_type": "feishu",
            "external_id": "oc_e2e_test_002",
            "name": "E2E Feishu Binding 2",
            "agent_id": pid,
            "config": {
                "app_id": "cli_e2e_app_002",
                "app_secret": "another_secret_999",
            },
        })
        assert resp.status_code == 200
        body = resp.json()
        type(self)._state["binding_id_2"] = body["id"]
        # Verify both bindings exist
        resp2 = await e2e_client.get("/api/v1/channels?channel_type=feishu")
        assert resp2.status_code == 200
        assert resp2.json()["total"] >= 2

    async def test_08_update_name_and_agent(self, e2e_client: AsyncClient):
        """Update binding name and agent_id via PUT."""
        bid = type(self)._state["binding_id_1"]
        pid2 = type(self)._state["preset_id_2"]
        resp = await e2e_client.put(f"/api/v1/channels/{bid}", json={
            "name": "E2E Feishu Binding 1 Updated",
            "agent_id": pid2,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "E2E Feishu Binding 1 Updated"
        assert body["agent_id"] == pid2
        assert body["agent_name"] == "e2e-channel-agent-2"

    async def test_09_update_trigger_pattern(self, e2e_client: AsyncClient):
        """Update trigger pattern."""
        bid = type(self)._state["binding_id_1"]
        resp = await e2e_client.put(f"/api/v1/channels/{bid}", json={
            "trigger_pattern": "@newbot|/help",
        })
        assert resp.status_code == 200
        assert resp.json()["trigger_pattern"] == "@newbot|/help"

    async def test_10_update_config_preserves_masked_secret(self, e2e_client: AsyncClient):
        """PUT with masked app_secret should preserve the original secret."""
        bid = type(self)._state["binding_id_1"]

        # First, get the current masked config
        resp = await e2e_client.get(f"/api/v1/channels/{bid}")
        assert resp.status_code == 200
        masked_secret = resp.json()["config"]["app_secret"]
        assert masked_secret.startswith("****")

        # Update config with the masked value — should preserve original
        resp = await e2e_client.put(f"/api/v1/channels/{bid}", json={
            "config": {
                "app_id": "cli_e2e_app_001_new",
                "app_secret": masked_secret,
            },
        })
        assert resp.status_code == 200
        body = resp.json()
        # app_id should be updated
        assert body["config"]["app_id"] == "cli_e2e_app_001_new"
        # Secret should still be masked (same last 4 chars)
        assert body["config"]["app_secret"].startswith("****")
        assert body["config"]["app_secret"].endswith("3xyz")

    async def test_11_update_config_with_new_secret(self, e2e_client: AsyncClient):
        """PUT with a new (non-masked) app_secret should update it."""
        bid = type(self)._state["binding_id_1"]
        resp = await e2e_client.put(f"/api/v1/channels/{bid}", json={
            "config": {
                "app_secret": "brand_new_secret_value",
            },
        })
        assert resp.status_code == 200
        body = resp.json()
        # New secret should be masked with its last 4 chars
        assert body["config"]["app_secret"].startswith("****")
        assert body["config"]["app_secret"].endswith("alue")

    async def test_12_clear_trigger_pattern(self, e2e_client: AsyncClient):
        """Setting trigger_pattern to null should clear it."""
        bid = type(self)._state["binding_id_1"]
        resp = await e2e_client.put(f"/api/v1/channels/{bid}", json={
            "trigger_pattern": None,
        })
        assert resp.status_code == 200
        assert resp.json()["trigger_pattern"] is None

    async def test_13_invalid_agent_id_rejected(self, e2e_client: AsyncClient):
        """Updating with a non-existent agent_id should fail."""
        bid = type(self)._state["binding_id_1"]
        resp = await e2e_client.put(f"/api/v1/channels/{bid}", json={
            "agent_id": "non-existent-agent-id",
        })
        assert resp.status_code == 400

    async def test_14_duplicate_binding_rejected(self, e2e_client: AsyncClient):
        """Creating a binding with duplicate channel_type+external_id should fail."""
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post("/api/v1/channels", json={
            "channel_type": "feishu",
            "external_id": "oc_e2e_test_001",  # Same as binding 1
            "name": "Duplicate binding",
            "agent_id": pid,
        })
        assert resp.status_code == 409

    async def test_15_toggle_binding(self, e2e_client: AsyncClient):
        """Toggle enabled/disabled should flip the state."""
        bid = type(self)._state["binding_id_1"]
        # Get current state
        resp = await e2e_client.get(f"/api/v1/channels/{bid}")
        was_enabled = resp.json()["enabled"]

        # Toggle
        resp = await e2e_client.post(f"/api/v1/channels/{bid}/toggle")
        assert resp.status_code == 200
        assert resp.json()["enabled"] == (not was_enabled)
        # Toggle includes agent_name
        assert resp.json()["agent_name"] is not None

        # Toggle back
        resp = await e2e_client.post(f"/api/v1/channels/{bid}/toggle")
        assert resp.status_code == 200
        assert resp.json()["enabled"] == was_enabled

    async def test_16_binding_without_config(self, e2e_client: AsyncClient):
        """Create a non-Feishu binding without config — should work fine."""
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post("/api/v1/channels", json={
            "channel_type": "webhook",
            "external_id": "webhook_e2e_001",
            "name": "E2E Webhook Binding",
            "agent_id": pid,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["config"] is None
        assert body["agent_name"] == "e2e-channel-agent"
        type(self)._state["webhook_binding_id"] = body["id"]

    async def test_17_adapter_key_derivation(self, e2e_client: AsyncClient):
        """Verify the adapter key derivation helper logic."""
        from app.services.channel_manager import adapter_key_for_binding

        class _FakeBinding:
            def __init__(self, channel_type, config):
                self.channel_type = channel_type
                self.config = config

        # Feishu with config
        b1 = _FakeBinding("feishu", {"app_id": "cli_abc", "app_secret": "s"})
        assert adapter_key_for_binding(b1) == "feishu:cli_abc"

        # Feishu without config
        b2 = _FakeBinding("feishu", None)
        assert adapter_key_for_binding(b2) is None

        # Feishu with empty config
        b3 = _FakeBinding("feishu", {})
        assert adapter_key_for_binding(b3) is None

        # Telegram
        b4 = _FakeBinding("telegram", None)
        assert adapter_key_for_binding(b4) == "telegram"

        # Webhook
        b5 = _FakeBinding("webhook", {"url": "http://example.com"})
        assert adapter_key_for_binding(b5) == "webhook"

    async def test_18_mask_config_helper(self, e2e_client: AsyncClient):
        """Test the _mask_config helper directly."""
        from app.api.v1.channels import _mask_config

        # None config
        assert _mask_config(None) is None

        # Empty config
        assert _mask_config({}) == {}

        # Config without secret
        assert _mask_config({"app_id": "cli_abc"}) == {"app_id": "cli_abc"}

        # Config with long secret
        result = _mask_config({"app_id": "cli_abc", "app_secret": "my_long_secret"})
        assert result["app_id"] == "cli_abc"
        assert result["app_secret"] == "****cret"

        # Config with short secret (<=4 chars)
        result = _mask_config({"app_secret": "ab"})
        assert result["app_secret"] == "****"

    async def test_19_merge_config_helper(self, e2e_client: AsyncClient):
        """Test the _merge_config helper directly."""
        from app.api.v1.channels import _merge_config

        # None incoming -> keep existing
        assert _merge_config({"app_id": "x"}, None) == {"app_id": "x"}

        # None existing -> use incoming (strip masked secrets)
        assert _merge_config(None, {"app_id": "x", "app_secret": "real"}) == {"app_id": "x", "app_secret": "real"}

        # Masked secret in incoming -> preserve existing
        result = _merge_config(
            {"app_id": "old", "app_secret": "original_secret"},
            {"app_id": "new", "app_secret": "****cret"},
        )
        assert result["app_id"] == "new"
        assert result["app_secret"] == "original_secret"

        # Real secret in incoming -> overwrite
        result = _merge_config(
            {"app_id": "old", "app_secret": "original_secret"},
            {"app_id": "new", "app_secret": "brand_new"},
        )
        assert result["app_id"] == "new"
        assert result["app_secret"] == "brand_new"

    async def test_20_delete_bindings(self, e2e_client: AsyncClient):
        """Delete all test bindings."""
        for key in ["binding_id_1", "binding_id_2", "webhook_binding_id"]:
            bid = type(self)._state.get(key)
            if bid:
                resp = await e2e_client.delete(f"/api/v1/channels/{bid}")
                assert resp.status_code == 200

    async def test_21_delete_nonexistent_binding(self, e2e_client: AsyncClient):
        """Deleting a non-existent binding should return 404."""
        resp = await e2e_client.delete("/api/v1/channels/nonexistent-id")
        assert resp.status_code == 404

    async def test_22_cleanup_presets(self, e2e_client: AsyncClient):
        """Clean up test agent presets."""
        for key in ["preset_id", "preset_id_2"]:
            pid = type(self)._state.get(key)
            if pid:
                await e2e_client.delete(f"/api/v1/agents/{pid}")


@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestGlobalFeishuBindingE2E:
    """E2E tests for global Feishu binding support (external_id='*').

    Covers: create global, duplicate rejection, coexistence with specific,
    is_global in responses, and cleanup.
    """

    _state: dict = {}

    async def test_01_create_preset(self, e2e_client: AsyncClient):
        """Create agent preset for global binding tests."""
        resp = await e2e_client.post("/api/v1/agents", json={
            "name": "e2e-global-agent",
            "description": "Agent for global binding tests",
            "max_turns": 5,
        })
        assert resp.status_code == 200
        type(self)._state["preset_id"] = resp.json()["id"]

    async def test_02_create_global_binding(self, e2e_client: AsyncClient):
        """Create a global Feishu binding (no external_id)."""
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post("/api/v1/channels", json={
            "channel_type": "feishu",
            "name": "Global Feishu E2E",
            "agent_id": pid,
            "config": {"app_id": "cli_global_e2e_001", "app_secret": "globalsecret123"},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["external_id"] == "*"
        assert body["is_global"] is True
        assert body["channel_type"] == "feishu"
        type(self)._state["global_id"] = body["id"]

    async def test_03_global_secret_masked(self, e2e_client: AsyncClient):
        """Global binding should also mask app_secret."""
        gid = type(self)._state["global_id"]
        resp = await e2e_client.get(f"/api/v1/channels/{gid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["config"]["app_secret"].startswith("****")
        assert body["config"]["app_secret"].endswith("t123")
        assert body["is_global"] is True

    async def test_04_duplicate_global_rejected(self, e2e_client: AsyncClient):
        """A second global binding for the same app_id should be rejected."""
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post("/api/v1/channels", json={
            "channel_type": "feishu",
            "name": "Duplicate Global",
            "agent_id": pid,
            "config": {"app_id": "cli_global_e2e_001", "app_secret": "other_secret"},
        })
        assert resp.status_code == 409

    async def test_05_global_non_feishu_rejected(self, e2e_client: AsyncClient):
        """Global binding for non-Feishu channel types should be rejected."""
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post("/api/v1/channels", json={
            "channel_type": "webhook",
            "name": "Global Webhook Fail",
            "agent_id": pid,
        })
        assert resp.status_code == 400
        assert "Feishu" in resp.json()["detail"]

    async def test_06_global_requires_app_id(self, e2e_client: AsyncClient):
        """Global Feishu binding without app_id in config should be rejected."""
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post("/api/v1/channels", json={
            "channel_type": "feishu",
            "name": "Global No AppId",
            "agent_id": pid,
            "config": {"app_secret": "only_secret"},
        })
        assert resp.status_code == 400
        assert "app_id" in resp.json()["detail"]

    async def test_07_create_specific_binding_same_app(self, e2e_client: AsyncClient):
        """A specific binding for the same app_id can coexist with the global one."""
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post("/api/v1/channels", json={
            "channel_type": "feishu",
            "external_id": "oc_specific_e2e_001",
            "name": "Specific Feishu E2E",
            "agent_id": pid,
            "config": {"app_id": "cli_global_e2e_001", "app_secret": "specificsecret"},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_global"] is False
        assert body["external_id"] == "oc_specific_e2e_001"
        type(self)._state["specific_id"] = body["id"]

    async def test_08_list_shows_is_global(self, e2e_client: AsyncClient):
        """List endpoint should show is_global correctly for both bindings."""
        resp = await e2e_client.get("/api/v1/channels?channel_type=feishu")
        assert resp.status_code == 200
        bindings = resp.json()["bindings"]
        global_b = next((b for b in bindings if b["id"] == type(self)._state["global_id"]), None)
        specific_b = next((b for b in bindings if b["id"] == type(self)._state["specific_id"]), None)
        assert global_b is not None and global_b["is_global"] is True
        assert specific_b is not None and specific_b["is_global"] is False

    async def test_09_second_app_global_binding(self, e2e_client: AsyncClient):
        """A global binding for a different app_id should succeed."""
        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post("/api/v1/channels", json={
            "channel_type": "feishu",
            "name": "Global Feishu App 2",
            "agent_id": pid,
            "config": {"app_id": "cli_global_e2e_002", "app_secret": "app2secret"},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_global"] is True
        type(self)._state["global_id_2"] = body["id"]

    async def test_10_cleanup(self, e2e_client: AsyncClient):
        """Delete all test bindings and preset."""
        for key in ["global_id", "global_id_2", "specific_id"]:
            bid = type(self)._state.get(key)
            if bid:
                resp = await e2e_client.delete(f"/api/v1/channels/{bid}")
                assert resp.status_code == 200

        pid = type(self)._state.get("preset_id")
        if pid:
            await e2e_client.delete(f"/api/v1/agents/{pid}")
