"""
Tests for the Agent run endpoint.

Covers:
- POST /api/v1/agent/run (synchronous agent execution)
- Various request parameters: skills, max_turns, session_id, files, mcp_servers
- Success and failure scenarios
- Trace saving after execution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from app.api.v1.sessions import SessionData


# ---------------------------------------------------------------------------
# Mock dataclasses matching app.agent.agent (AgentStep, LLMCall, AgentResult)
# ---------------------------------------------------------------------------


@dataclass
class MockStep:
    role: str = "assistant"
    content: str = "Mock answer"
    tool_name: Optional[str] = None
    tool_input: Optional[Dict] = None
    tool_result: Optional[str] = None


@dataclass
class MockLLMCall:
    turn: int = 1
    timestamp: str = "2024-01-01T00:00:00"
    model: str = "kimi-k2.5"
    request_messages: List[Dict] = field(default_factory=list)
    response_content: List[Dict] = field(default_factory=list)
    stop_reason: str = "end_turn"
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class MockAgentResult:
    success: bool = True
    answer: str = "Mock answer"
    total_turns: int = 1
    total_input_tokens: int = 100
    total_output_tokens: int = 50
    steps: List = field(default_factory=lambda: [MockStep()])
    llm_calls: List = field(default_factory=lambda: [MockLLMCall()])
    error: Optional[str] = None
    log_file: Optional[str] = None
    output_files: List = field(default_factory=list)
    skills_used: List = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper to build a mock SkillsAgent instance
# ---------------------------------------------------------------------------


def _make_mock_agent(result: Optional[MockAgentResult] = None):
    """Return a MagicMock that behaves like SkillsAgent."""
    instance = MagicMock()
    instance.run = AsyncMock(return_value=result or MockAgentResult())
    instance.model = "kimi-k2.5"
    instance.model_provider = "kimi"
    instance.cleanup = MagicMock()
    return instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
@patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
@patch("app.api.v1.agent.create_agent")
async def test_agent_run_simple(mock_create, _mock_load, _mock_save, client: AsyncClient):
    """POST /agent/run with a simple request returns 200 with success=True."""
    mock_create.return_value = _make_mock_agent()

    response = await client.post(
        "/api/v1/agent/run",
        json={"request": "Hello, what can you do?", "session_id": "test-session-id"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["answer"] == "Mock answer"
    assert body["total_turns"] == 1
    assert len(body["steps"]) >= 1


@patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
@patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
@patch("app.api.v1.agent.create_agent")
async def test_agent_run_with_skills(mock_create, _mock_load, _mock_save, client: AsyncClient):
    """POST /agent/run with skills parameter passes skills to agent via AgentConfig."""
    mock_create.return_value = _make_mock_agent()

    response = await client.post(
        "/api/v1/agent/run",
        json={"request": "Analyze this", "skills": ["test-skill"], "session_id": "test-session-id"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    # Verify create_agent received config with correct skills
    mock_create.assert_called_once()
    config = mock_create.call_args[0][0]
    assert config.skills == ["test-skill"]


@patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
@patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
@patch("app.api.v1.agent.create_agent")
async def test_agent_run_with_max_turns(mock_create, _mock_load, _mock_save, client: AsyncClient):
    """POST /agent/run with max_turns passes the value to agent."""
    mock_create.return_value = _make_mock_agent()

    response = await client.post(
        "/api/v1/agent/run",
        json={"request": "Quick task", "max_turns": 5, "session_id": "test-session-id"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    config = mock_create.call_args[0][0]
    assert config.max_turns == 5


@patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
@patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
@patch("app.api.v1.agent.create_agent")
async def test_agent_run_with_session_id(mock_create, _mock_load, _mock_save, client: AsyncClient):
    """POST /agent/run with session_id accepts and processes the request."""
    mock_create.return_value = _make_mock_agent()

    response = await client.post(
        "/api/v1/agent/run",
        json={"request": "Continue our chat", "session_id": "test-session-id"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True


@patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
@patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
@patch("app.api.v1.agent.create_agent")
async def test_agent_run_with_files(mock_create, _mock_load, _mock_save, client: AsyncClient):
    """POST /agent/run with uploaded_files appends file info to request."""
    mock_agent = _make_mock_agent()
    mock_create.return_value = mock_agent

    files = [
        {
            "file_id": "abc123",
            "filename": "report.pdf",
            "path": "/tmp/uploads/report.pdf",
            "content_type": "application/pdf",
        }
    ]

    response = await client.post(
        "/api/v1/agent/run",
        json={"request": "Summarize this file", "uploaded_files": files, "session_id": "test-session-id"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    # The actual_request passed to agent.run should contain the file info
    actual_request = mock_agent.run.call_args[0][0]
    assert "report.pdf" in actual_request
    assert "/tmp/uploads/report.pdf" in actual_request


@patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
@patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
@patch("app.api.v1.agent.create_agent")
async def test_agent_run_failure(mock_create, _mock_load, _mock_save, client: AsyncClient):
    """POST /agent/run when agent fails returns 200 with success=False."""
    failed_result = MockAgentResult(
        success=False,
        answer="",
        error="Something went wrong",
        steps=[MockStep(role="assistant", content="Error occurred")],
    )
    mock_create.return_value = _make_mock_agent(failed_result)

    response = await client.post(
        "/api/v1/agent/run",
        json={"request": "Do something impossible", "session_id": "test-session-id"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["error"] == "Something went wrong"


@patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
@patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
@patch("app.api.v1.agent.create_agent")
async def test_agent_run_saves_trace(mock_create, _mock_load, _mock_save, client: AsyncClient):
    """POST /agent/run saves an execution trace and returns trace_id."""
    mock_create.return_value = _make_mock_agent()

    response = await client.post(
        "/api/v1/agent/run",
        json={"request": "Test trace saving", "session_id": "test-session-id"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trace_id"] is not None
    assert isinstance(body["trace_id"], str)
    assert len(body["trace_id"]) > 0


@patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
@patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
@patch("app.api.v1.agent.create_agent")
async def test_agent_run_with_mcp_servers(mock_create, _mock_load, _mock_save, client: AsyncClient):
    """POST /agent/run with equipped_mcp_servers passes them to agent."""
    mock_create.return_value = _make_mock_agent()

    response = await client.post(
        "/api/v1/agent/run",
        json={
            "request": "Fetch a webpage",
            "equipped_mcp_servers": ["fetch"],
            "session_id": "test-session-id",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    config = mock_create.call_args[0][0]
    assert config.equipped_mcp_servers == ["fetch"]
