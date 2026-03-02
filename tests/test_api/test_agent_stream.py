"""
Tests for Agent streaming endpoint: POST /api/v1/agent/run/stream

Uses mocked SkillsAgent (via create_agent) to avoid real LLM calls.
The stream endpoint uses AsyncSessionLocal directly (not get_db),
so we also need to mock that for trace saving.

Also tests:
- POST /api/v1/agent/run/stream/{trace_id}/steer
"""
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.agent import StreamEvent
from app.agent.event_stream import EventStream
from app.api.v1.sessions import SessionData


@dataclass
class MockStep:
    role: str = "assistant"
    content: str = "Mock answer"
    tool_name: str = None
    tool_input: dict = None
    tool_result: str = None


@dataclass
class MockLLMCall:
    turn: int = 1
    timestamp: str = "2024-01-01T00:00:00"
    model: str = "claude-sonnet-4-5-20250929"
    input_tokens: int = 100
    output_tokens: int = 50
    stop_reason: str = "end_turn"
    messages: list = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []


def _make_stream_events():
    """Create a list of StreamEvent objects for mock agent."""
    return [
        StreamEvent(event_type="turn_start", turn=1, data={"turn": 1}),
        StreamEvent(
            event_type="assistant",
            turn=1,
            data={"content": "Hello from mock", "turn": 1},
        ),
        StreamEvent(
            event_type="complete",
            turn=1,
            data={
                "success": True,
                "answer": "Done",
                "total_turns": 1,
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "skills_used": [],
                "output_files": [],
                "final_messages": [],
            },
        ),
    ]


def _make_mock_agent_instance(events=None):
    """Create a mock SkillsAgent whose run() pushes events to event_stream."""
    from app.agent.agent import AgentResult

    if events is None:
        events = _make_stream_events()

    # Extract result from complete event
    complete_event = next((e for e in events if e.event_type == "complete"), None)
    result = AgentResult(
        success=complete_event.data.get("success", True) if complete_event else True,
        answer=complete_event.data.get("answer", "Done") if complete_event else "Done",
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


def _mock_session_local(db_session: AsyncSession):
    """Create a mock AsyncSessionLocal that returns a context manager
    wrapping the test db_session but with no-op commit/close."""

    @asynccontextmanager
    async def _mock_ctx():
        # Create a lightweight mock that delegates execute/add to a no-op
        # since we don't need trace saving to actually work in stream tests
        mock_sess = AsyncMock(spec=AsyncSession)
        mock_sess.add = MagicMock()
        mock_sess.commit = AsyncMock()
        mock_sess.close = AsyncMock()
        mock_sess.execute = AsyncMock()
        yield mock_sess

    return _mock_ctx


@pytest.mark.asyncio
@patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
@patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
@patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
@patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
@patch("app.api.v1.agent.create_agent")
async def test_stream_returns_event_stream(mock_create, _mock_load, _mock_save, _mock_checkpoint, _mock_precompress, client):
    mock_create.return_value = _make_mock_agent_instance()

    response = await client.post(
        "/api/v1/agent/run/stream",
        json={"request": "hello", "session_id": "test-session-id"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")


@pytest.mark.asyncio
@patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
@patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
@patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
@patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
@patch("app.api.v1.agent.AsyncSessionLocal")
@patch("app.api.v1.agent.create_agent")
async def test_stream_sends_run_started(mock_create, MockSessionLocal, _mock_load, _mock_save, _mock_checkpoint, _mock_precompress, client, db_session):
    mock_create.return_value = _make_mock_agent_instance()
    MockSessionLocal.side_effect = lambda: _mock_session_local(db_session)()

    response = await client.post(
        "/api/v1/agent/run/stream",
        json={"request": "hello", "session_id": "test-session-id"},
    )
    assert response.status_code == 200
    text = response.text
    # SSE events are formatted as "data: {...}\n\n"
    lines = [l for l in text.strip().split("\n") if l.startswith("data: ")]
    assert len(lines) > 0
    first_event = json.loads(lines[0].replace("data: ", ""))
    assert first_event["event_type"] == "run_started"
    assert "trace_id" in first_event


@pytest.mark.asyncio
@patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
@patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
@patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
@patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
@patch("app.api.v1.agent.AsyncSessionLocal")
@patch("app.api.v1.agent.create_agent")
async def test_stream_sends_trace_saved(mock_create, MockSessionLocal, _mock_load, _mock_save, _mock_checkpoint, _mock_precompress, client, db_session):
    mock_create.return_value = _make_mock_agent_instance()
    MockSessionLocal.side_effect = lambda: _mock_session_local(db_session)()

    response = await client.post(
        "/api/v1/agent/run/stream",
        json={"request": "hello", "session_id": "test-session-id"},
    )
    text = response.text
    lines = [l for l in text.strip().split("\n") if l.startswith("data: ")]
    last_event = json.loads(lines[-1].replace("data: ", ""))
    assert last_event["event_type"] == "trace_saved"
    assert "trace_id" in last_event


@pytest.mark.asyncio
@patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
@patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
@patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
@patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
@patch("app.api.v1.agent.AsyncSessionLocal")
@patch("app.api.v1.agent.create_agent")
async def test_stream_with_skills(mock_create, MockSessionLocal, _mock_load, _mock_save, _mock_checkpoint, _mock_precompress, client, db_session):
    mock_create.return_value = _make_mock_agent_instance()
    MockSessionLocal.side_effect = lambda: _mock_session_local(db_session)()

    response = await client.post(
        "/api/v1/agent/run/stream",
        json={"request": "hello", "skills": ["test-skill"], "session_id": "test-session-id"},
    )
    assert response.status_code == 200
    # create_agent is called twice in stream (once for initial trace config read, once for actual agent)
    # The actual agent creation uses a stream_config with verbose=False
    assert mock_create.call_count >= 1
    # Check the config passed to create_agent has the correct skills
    config = mock_create.call_args[0][0]
    assert config.skills == ["test-skill"]


@pytest.mark.asyncio
@patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
@patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
@patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
@patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
@patch("app.api.v1.agent.AsyncSessionLocal")
@patch("app.api.v1.agent.create_agent")
async def test_stream_with_mcp_servers(mock_create, MockSessionLocal, _mock_load, _mock_save, _mock_checkpoint, _mock_precompress, client, db_session):
    mock_create.return_value = _make_mock_agent_instance()
    MockSessionLocal.side_effect = lambda: _mock_session_local(db_session)()

    response = await client.post(
        "/api/v1/agent/run/stream",
        json={"request": "hello", "equipped_mcp_servers": ["fetch"], "session_id": "test-session-id"},
    )
    assert response.status_code == 200
    assert mock_create.call_count >= 1
    config = mock_create.call_args[0][0]
    assert config.equipped_mcp_servers == ["fetch"]


@pytest.mark.asyncio
@patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
@patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
@patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
@patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock, return_value=SessionData(session_id="test-session-id"))
@patch("app.api.v1.agent.AsyncSessionLocal")
@patch("app.api.v1.agent.create_agent")
async def test_stream_error_handling(mock_create, MockSessionLocal, _mock_load, _mock_save, _mock_checkpoint, _mock_precompress, client, db_session):
    """When agent encounters error, it pushes error complete event and stream includes it."""
    # In the async architecture, agent.run() catches errors internally
    # and pushes a complete event with success=False
    error_events = [
        StreamEvent(event_type="turn_start", turn=1, data={"turn": 1}),
        StreamEvent(
            event_type="complete",
            turn=1,
            data={
                "success": False,
                "answer": "Error: Mock error",
                "error": "Mock error",
                "total_turns": 1,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "skills_used": [],
                "output_files": [],
                "final_messages": [],
            },
        ),
    ]
    mock_create.return_value = _make_mock_agent_instance(events=error_events)
    MockSessionLocal.side_effect = lambda: _mock_session_local(db_session)()

    response = await client.post(
        "/api/v1/agent/run/stream",
        json={"request": "hello", "session_id": "test-session-id"},
    )
    assert response.status_code == 200
    text = response.text
    lines = [l for l in text.strip().split("\n") if l.startswith("data: ")]
    # Should have run_started + events + trace_saved
    assert len(lines) >= 2
    # Check for error complete event
    events = [json.loads(l.replace("data: ", "")) for l in lines]
    complete_events = [e for e in events if e.get("event_type") == "complete"]
    assert len(complete_events) == 1
    assert complete_events[0]["success"] is False


# ---------------------------------------------------------------------------
# Steer endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_steer_no_active_run_404(client):
    """Steering a non-existent trace returns 404."""
    resp = await client.post(
        "/api/v1/agent/run/stream/nonexistent-trace-id/steer",
        json={"message": "do something else"},
    )
    assert resp.status_code == 404
    assert "No active run" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_steer_completed_run_409(client):
    """Steering a completed (closed) stream returns 409."""
    from app.api.v1.agent import _active_streams

    # Register a closed event stream
    es = EventStream()
    await es.close()
    _active_streams["test-trace-closed"] = es

    try:
        resp = await client.post(
            "/api/v1/agent/run/stream/test-trace-closed/steer",
            json={"message": "too late"},
        )
        assert resp.status_code == 409
        assert "already completed" in resp.json()["detail"]
    finally:
        _active_streams.pop("test-trace-closed", None)


@pytest.mark.asyncio
async def test_steer_endpoint_injects_message(client):
    """Steering an active stream injects the message into its injection queue."""
    from app.api.v1.agent import _active_streams

    es = EventStream()
    _active_streams["test-trace-active"] = es

    try:
        resp = await client.post(
            "/api/v1/agent/run/stream/test-trace-active/steer",
            json={"message": "focus on validation"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "injected"
        assert body["trace_id"] == "test-trace-active"

        # Verify the message is in the injection queue
        assert es.has_injection() is True
        msg = es.get_injection_nowait()
        assert msg == "focus on validation"
    finally:
        _active_streams.pop("test-trace-active", None)


@pytest.mark.asyncio
async def test_steer_empty_message_rejected(client):
    """Empty steering message is rejected by validation."""
    resp = await client.post(
        "/api/v1/agent/run/stream/some-trace/steer",
        json={"message": ""},
    )
    assert resp.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_steer_cross_worker_via_filesystem(client, db_session):
    """Cross-worker steering: trace in DB + filesystem queue (no _active_streams entry)."""
    from app.db.models import AgentTraceDB
    from app.agent.steering import STEERING_DIR, cleanup_steering_dir

    # Create a running trace in the DB (simulating another worker)
    trace = AgentTraceDB(
        request="test",
        skills_used=[],
        model="test",
        model_provider="test",
        status="running",
        success=False,
        answer="",
        total_turns=0,
        total_input_tokens=0,
        total_output_tokens=0,
        steps=[],
        llm_calls=[],
        duration_ms=0,
    )
    db_session.add(trace)
    await db_session.commit()

    try:
        # Steer should hit the cross-worker path (no _active_streams entry)
        resp = await client.post(
            f"/api/v1/agent/run/stream/{trace.id}/steer",
            json={"message": "cross-worker steer"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "injected"

        # Verify the message was written to filesystem
        trace_dir = STEERING_DIR / trace.id
        msg_files = list(trace_dir.glob("*.msg"))
        assert len(msg_files) == 1
        assert msg_files[0].read_text(encoding="utf-8") == "cross-worker steer"
    finally:
        cleanup_steering_dir(trace.id)


@pytest.mark.asyncio
async def test_steer_cross_worker_completed_trace_409(client, db_session):
    """Cross-worker steering: completed trace returns 409."""
    from app.db.models import AgentTraceDB

    # Create a completed trace
    trace = AgentTraceDB(
        request="test",
        skills_used=[],
        model="test",
        model_provider="test",
        status="completed",
        success=True,
        answer="done",
        total_turns=1,
        total_input_tokens=100,
        total_output_tokens=50,
        steps=[],
        llm_calls=[],
        duration_ms=1000,
    )
    db_session.add(trace)
    await db_session.commit()

    resp = await client.post(
        f"/api/v1/agent/run/stream/{trace.id}/steer",
        json={"message": "too late"},
    )
    assert resp.status_code == 409
    assert "already completed" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_make_steering_event_unique_ids():
    """_make_steering_event produces events with unique steering_ids."""
    from app.agent.agent import _make_steering_event

    e1 = _make_steering_event(1, "steer message 1")
    e2 = _make_steering_event(1, "steer message 2")
    e3 = _make_steering_event(2, "steer message 1")  # same text, different turn

    # All have steering_id
    assert e1.data["steering_id"].startswith("steer-")
    assert e2.data["steering_id"].startswith("steer-")
    assert e3.data["steering_id"].startswith("steer-")

    # All IDs are unique (even with same message text)
    ids = {e1.data["steering_id"], e2.data["steering_id"], e3.data["steering_id"]}
    assert len(ids) == 3

    # Messages and turns preserved
    assert e1.data["message"] == "steer message 1"
    assert e2.data["message"] == "steer message 2"
    assert e1.event_type == "steering_received"
    assert e1.turn == 1
    assert e3.turn == 2


@pytest.mark.asyncio
async def test_steer_409_response_body(client):
    """409 response includes 'already completed' detail for client-side detection."""
    from app.api.v1.agent import _active_streams

    # Register a closed event stream
    es = EventStream()
    await es.close()
    _active_streams["test-trace-409-body"] = es

    try:
        resp = await client.post(
            "/api/v1/agent/run/stream/test-trace-409-body/steer",
            json={"message": "too late"},
        )
        assert resp.status_code == 409
        body = resp.json()
        assert "detail" in body
        assert "already completed" in body["detail"]
    finally:
        _active_streams.pop("test-trace-409-body", None)
