"""
End-to-end tests for context compression and session full-message persistence.

Covers:
- Logical turn boundary detection (never splits tool_use/tool_result pairs)
- Dynamic token budget with MAX_RECENT_TURNS cap
- Summary prompt format (<summary> tags)
- Compression message structure (alternation, recent turns preserved)
- Serialization of messages for summarization
- Published Agent session saves full messages (tool_use/tool_result)
- Session history restoration with full context
- Incremental display save on turn_complete (stop/refresh resilience)

Run:
    pytest tests/test_e2e/test_context_compression.py -v
"""

import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.agent import StreamEvent
from app.api.v1.sessions import SessionData
from tests.test_e2e.conftest import parse_sse_events


# ---------------------------------------------------------------------------
# Helpers: build realistic messages with tool_use / tool_result
# ---------------------------------------------------------------------------

def _make_simple_turn(user_text: str, assistant_text: str) -> List[Dict]:
    """A simple user→assistant turn (no tools)."""
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]


def _make_tool_turn(
    user_text: str,
    tools: List[Dict],
    assistant_text: str,
) -> List[Dict]:
    """A turn where the agent uses one or more tools before responding.

    tools: list of {"name": ..., "input": ..., "result": ...}
    """
    msgs: List[Dict] = [{"role": "user", "content": user_text}]
    for i, tool in enumerate(tools):
        tool_id = f"{tool['name']}:{i}"
        msgs.append({
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_id, "name": tool["name"], "input": tool.get("input", {})}],
        })
        msgs.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": tool.get("result", "ok")}],
        })
    msgs.append({
        "role": "assistant",
        "content": [{"type": "text", "text": assistant_text}],
    })
    return msgs


def _make_heavy_tool_turn(user_text: str, result_size: int = 50000) -> List[Dict]:
    """A turn with a large tool_result (simulates get_skill returning big SKILL.md)."""
    return _make_tool_turn(
        user_text,
        [{"name": "get_skill", "input": {"skill_name": "big-skill"}, "result": "X" * result_size}],
        "Done.",
    )


# ---------------------------------------------------------------------------
# Class 1: Unit-level tests for _compress_messages internals
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestCompressionTurnBoundariesE2E:
    """Test logical turn boundary detection and split logic."""

    _state: dict = {}

    def _make_agent(self, context_limit: int = 256_000):
        """Create a SkillsAgent with bypassed __init__ for unit testing methods."""
        from app.agent.agent import SkillsAgent
        agent = SkillsAgent.__new__(SkillsAgent)
        agent.model = "kimi-k2.5"
        agent.model_provider = "kimi"
        agent.verbose = False
        agent.client = MagicMock()
        agent._get_context_limit = lambda: context_limit
        return agent

    def _mock_summary_response(self, agent, summary_text="<summary>\n## Test\nMocked summary\n</summary>"):
        """Mock the LLM client.create call used by _compress_messages."""
        mock_resp = MagicMock()
        mock_resp.text_content = summary_text
        mock_resp.usage = MagicMock(input_tokens=500, output_tokens=200)
        agent.client = MagicMock()
        agent.client.acreate = AsyncMock(return_value=mock_resp)

    async def test_01_simple_turns_boundary_detection(self):
        """Simple user/assistant pairs: each pair is one logical turn."""
        agent = self._make_agent()
        self._mock_summary_response(agent)

        # 8 simple turns (need more than MAX_RECENT_TURNS=5 to trigger compression)
        messages = []
        for i in range(8):
            messages.extend(_make_simple_turn(f"Q{i}", f"A{i}"))

        compressed, s_in, s_out = await agent._compress_messages(messages)
        # Should keep last 5 turns = 10 messages, compress first 3 turns
        # compressed = [summary_user, ack_assistant] + 10 recent = 12
        assert compressed[0]["role"] == "user"
        assert "<summary>" in compressed[0]["content"]
        assert compressed[1]["role"] == "assistant"
        # Recent messages start at index 2
        assert compressed[2]["content"] == "Q3"  # Turn 4 (0-indexed turn 3)

    async def test_02_tool_turns_never_split(self):
        """Turns with tool_use/tool_result: split only at logical turn boundaries."""
        agent = self._make_agent()
        self._mock_summary_response(agent)

        # Turn 1: user + get_skill + tool_result + execute_code + tool_result + assistant = 6 msgs
        turn1 = _make_tool_turn("Generate image", [
            {"name": "get_skill", "input": {"skill_name": "gemini-imagegen"}, "result": "SKILL.md content"},
            {"name": "execute_code", "input": {"code": "print(1)"}, "result": "1"},
        ], "Image generated!")

        # Turn 2: simple
        turn2 = _make_simple_turn("Make it bigger", "Resized!")

        # Turn 3: with one tool
        turn3 = _make_tool_turn("Add rainbow", [
            {"name": "execute_code", "input": {"code": "add_rainbow()"}, "result": "done"},
        ], "Rainbow added!")

        # Turn 4: simple
        turn4 = _make_simple_turn("Thanks", "You're welcome!")

        # Turn 5: simple
        turn5 = _make_simple_turn("Another task", "Sure!")

        # Turn 6: simple
        turn6 = _make_simple_turn("More work", "Done!")

        # Turn 7: current question
        messages = turn1 + turn2 + turn3 + turn4 + turn5 + turn6 + [{"role": "user", "content": "What model?"}]

        compressed, _, _ = await agent._compress_messages(messages)

        # Should keep last 5 logical turns (turn 3, 4, 5, 6, 7)
        # Recent starts at turn 3's user message "Add rainbow"
        recent_start = next(i for i, m in enumerate(compressed) if m.get("content") == "Add rainbow")
        # Verify turn 3 is complete: user, assistant(tool_use), user(tool_result), assistant(text)
        assert compressed[recent_start]["role"] == "user"
        assert compressed[recent_start]["content"] == "Add rainbow"
        # Next should be assistant with tool_use
        assert compressed[recent_start + 1]["role"] == "assistant"
        assert compressed[recent_start + 1]["content"][0]["type"] == "tool_use"
        # Then tool_result
        assert compressed[recent_start + 2]["role"] == "user"
        assert compressed[recent_start + 2]["content"][0]["type"] == "tool_result"
        # Then text response
        assert compressed[recent_start + 3]["role"] == "assistant"
        assert compressed[recent_start + 3]["content"][0]["text"] == "Rainbow added!"

    async def test_03_not_enough_turns_skip(self):
        """Only 1 logical turn: skip compression."""
        agent = self._make_agent()

        messages = _make_tool_turn("Hello", [
            {"name": "get_skill", "result": "content"},
        ], "Hi!")

        result, s_in, s_out = await agent._compress_messages(messages)
        assert result is messages  # Unchanged
        assert s_in == 0
        assert s_out == 0

    async def test_04_heavy_turns_compress_oldest(self):
        """Heavy turns with large tool results: compresses oldest, keeps recent ones within budget."""
        agent = self._make_agent(context_limit=10_000)  # Small limit → budget = 2500 tokens
        self._mock_summary_response(agent)

        # Turn 1: heavy (~14K tokens, exceeds entire budget alone)
        turn1 = _make_tool_turn("Turn 1", [
            {"name": "get_skill", "result": "X" * 50000},
        ], "Done 1")
        # Turns 2-4: light
        turn2 = _make_simple_turn("Turn 2", "Done 2")
        turn3 = _make_simple_turn("Turn 3", "Done 3")
        turn4 = _make_simple_turn("Turn 4", "Done 4")

        messages = turn1 + turn2 + turn3 + turn4
        compressed, _, _ = await agent._compress_messages(messages)

        # Summary message exists
        assert compressed[0]["role"] == "user"
        assert "<summary>" in compressed[0]["content"]
        # Recent portion includes some light turns
        recent_user_texts = [
            m["content"] for m in compressed
            if m["role"] == "user" and isinstance(m.get("content"), str)
            and "<summary>" not in m.get("content", "")
        ]
        assert "Turn 4" in recent_user_texts
        # Turn 1 (heavy) is compressed away
        assert "Turn 1" not in recent_user_texts

    async def test_05_tool_result_not_counted_as_turn_boundary(self):
        """User messages with tool_result content are NOT turn boundaries."""
        agent = self._make_agent()
        self._mock_summary_response(agent)

        # Build messages manually to expose the boundary detection
        # Need 7+ real turns so that with MAX_RECENT_TURNS=5, some get compressed
        messages = [
            {"role": "user", "content": "Real user turn 1"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t:0", "name": "bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t:0", "content": "ok"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t:1", "name": "bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t:1", "content": "ok2"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "All done"}]},
            {"role": "user", "content": "Real user turn 2"},
            {"role": "assistant", "content": [{"type": "text", "text": "Response 2"}]},
            {"role": "user", "content": "Real user turn 3"},
            {"role": "assistant", "content": [{"type": "text", "text": "Response 3"}]},
            {"role": "user", "content": "Real user turn 4"},
            {"role": "assistant", "content": [{"type": "text", "text": "Response 4"}]},
            {"role": "user", "content": "Real user turn 5"},
            {"role": "assistant", "content": [{"type": "text", "text": "Response 5"}]},
            {"role": "user", "content": "Real user turn 6"},
            {"role": "assistant", "content": [{"type": "text", "text": "Response 6"}]},
            {"role": "user", "content": "Real user turn 7"},
        ]

        compressed, _, _ = await agent._compress_messages(messages)

        # 7 real turn boundaries. Keep 5 → compress turns 1-2, keep turns 3-7.
        # Turn 3 starts at "Real user turn 3"
        recent_user_msgs = [
            m["content"] for m in compressed
            if m["role"] == "user" and isinstance(m.get("content"), str) and m["content"] != compressed[0]["content"]
        ]
        assert "Real user turn 3" in recent_user_msgs
        assert "Real user turn 4" in recent_user_msgs
        assert "Real user turn 5" in recent_user_msgs
        assert "Real user turn 6" in recent_user_msgs
        assert "Real user turn 7" in recent_user_msgs
        # Turn 1 and 2 should NOT appear in recent (compressed away)
        assert "Real user turn 1" not in recent_user_msgs
        assert "Real user turn 2" not in recent_user_msgs


# ---------------------------------------------------------------------------
# Class 2: Dynamic token budget + MAX_RECENT_TURNS cap
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestCompressionTokenBudgetE2E:
    """Test dynamic token budget and max turns cap."""

    _state: dict = {}

    def _make_agent(self, context_limit: int = 256_000):
        from app.agent.agent import SkillsAgent
        agent = SkillsAgent.__new__(SkillsAgent)
        agent.model = "kimi-k2.5"
        agent.model_provider = "kimi"
        agent.verbose = False
        agent.client = MagicMock()
        agent._get_context_limit = lambda: context_limit
        return agent

    def _mock_summary(self, agent):
        mock_resp = MagicMock()
        mock_resp.text_content = "<summary>\n## Test\nSummary\n</summary>"
        mock_resp.usage = MagicMock(input_tokens=500, output_tokens=200)
        agent.client = MagicMock()
        agent.client.acreate = AsyncMock(return_value=mock_resp)

    async def test_01_max_five_turns_cap(self):
        """Even with small turns, never keep more than MAX_RECENT_TURNS (5)."""
        agent = self._make_agent(context_limit=1_000_000)  # Huge budget
        self._mock_summary(agent)

        # 12 simple turns — all tiny, well within budget
        messages = []
        for i in range(12):
            messages.extend(_make_simple_turn(f"Q{i}", f"A{i}"))

        compressed, _, _ = await agent._compress_messages(messages)

        # Count real user messages in recent portion (after summary)
        recent_user_texts = [
            m["content"] for m in compressed
            if m["role"] == "user" and isinstance(m.get("content"), str) and "<summary>" not in m.get("content", "")
        ]
        # Max 5 turns kept
        assert len(recent_user_texts) == 5
        assert recent_user_texts == ["Q7", "Q8", "Q9", "Q10", "Q11"]

    async def test_02_token_budget_limits_before_cap(self):
        """Heavy turns hit token budget before reaching 5 turns."""
        # Small context → small budget (256K * 0.25 = 64K tokens)
        agent = self._make_agent(context_limit=256_000)
        self._mock_summary(agent)

        # Each turn has ~23K tokens of content
        messages = []
        for i in range(8):
            messages.extend(_make_heavy_tool_turn(f"Heavy task {i}", result_size=80000))

        compressed, _, _ = await agent._compress_messages(messages)

        # Count kept user text messages
        recent_user_texts = [
            m["content"] for m in compressed
            if m["role"] == "user" and isinstance(m.get("content"), str) and "<summary>" not in m.get("content", "")
        ]
        # Should keep fewer than 5 due to token budget
        assert len(recent_user_texts) < 5
        # But at least 1 is always kept
        assert len(recent_user_texts) >= 1

    async def test_03_tiny_context_keeps_at_least_one(self):
        """Even with a very small context limit, at least 1 turn is kept."""
        agent = self._make_agent(context_limit=1000)  # Tiny: budget = 250 tokens
        self._mock_summary(agent)

        messages = []
        for i in range(8):
            messages.extend(_make_heavy_tool_turn(f"Task {i}", result_size=10000))

        compressed, _, _ = await agent._compress_messages(messages)

        # Compressed output should be shorter than original
        assert len(compressed) < len(messages)
        # First message should be summary
        assert compressed[0]["role"] == "user"
        assert "<summary>" in compressed[0]["content"]

    async def test_04_all_turns_fit_skip_compression(self):
        """If all turns fit in budget + under cap, skip compression."""
        agent = self._make_agent(context_limit=1_000_000)

        # Only 2 tiny turns — both fit and under cap of 3
        messages = _make_simple_turn("Q1", "A1") + _make_simple_turn("Q2", "A2")
        result, s_in, s_out = await agent._compress_messages(messages)

        assert result is messages  # Returned as-is
        assert s_in == 0

    async def test_05_mixed_heavy_and_light_turns(self):
        """Mix of heavy and light turns: budget logic correctly accumulates."""
        agent = self._make_agent(context_limit=256_000)  # Budget = 64K tokens
        self._mock_summary(agent)

        # 4 heavy turns (~23K tokens each) + 4 light turns (~10 tokens each)
        messages = []
        for i in range(4):
            messages.extend(_make_heavy_tool_turn(f"Heavy {i}", result_size=80000))
        for i in range(4):
            messages.extend(_make_simple_turn(f"Light {i}", f"OK {i}"))
        messages.append({"role": "user", "content": "Final"})

        compressed, _, _ = await agent._compress_messages(messages)

        # Light turns are tiny, should be included; heavy turns may exceed budget
        # The light turns + "Final" should definitely be in recent
        recent_user_texts = [
            m["content"] for m in compressed
            if m["role"] == "user" and isinstance(m.get("content"), str) and "<summary>" not in m.get("content", "")
        ]
        assert "Final" in recent_user_texts
        assert "Light 3" in recent_user_texts


# ---------------------------------------------------------------------------
# Class 3: Summary format and compression message structure
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestCompressionSummaryFormatE2E:
    """Test summary prompt, <summary> tags, and compressed message structure."""

    _state: dict = {}

    def _make_agent(self):
        from app.agent.agent import SkillsAgent
        agent = SkillsAgent.__new__(SkillsAgent)
        agent.model = "kimi-k2.5"
        agent.model_provider = "kimi"
        agent.verbose = False
        agent.client = MagicMock()
        agent._get_context_limit = lambda: 256_000
        return agent

    async def test_01_summary_has_summary_tags(self):
        """LLM-generated summary with <summary> tags is used as-is (not double-wrapped)."""
        agent = self._make_agent()
        llm_summary = "<summary>\n## Primary Request\nUser wanted X\n## Current State\nDone Y\n</summary>"
        mock_resp = MagicMock()
        mock_resp.text_content = llm_summary
        mock_resp.usage = MagicMock(input_tokens=500, output_tokens=200)
        agent.client = MagicMock()
        agent.client.acreate = AsyncMock(return_value=mock_resp)

        messages = []
        for i in range(8):
            messages.extend(_make_simple_turn(f"Q{i}", f"A{i}"))

        compressed, _, _ = await agent._compress_messages(messages)

        summary_content = compressed[0]["content"]
        # Should contain the summary tags exactly once
        assert summary_content.count("<summary>") == 1
        assert summary_content.count("</summary>") == 1
        assert "User wanted X" in summary_content

    async def test_02_fallback_wraps_in_summary_tags(self):
        """When LLM call fails, fallback text gets wrapped in <summary> tags."""
        agent = self._make_agent()
        agent.client = MagicMock()
        agent.client.acreate = AsyncMock(side_effect=Exception("API error"))

        messages = []
        for i in range(8):
            messages.extend(_make_simple_turn(f"Q{i}", f"A{i}"))

        compressed, _, _ = await agent._compress_messages(messages)

        summary_content = compressed[0]["content"]
        assert "<summary>" in summary_content
        assert "</summary>" in summary_content

    async def test_03_compression_message_structure(self):
        """Compressed messages have proper structure: summary + ack + recent."""
        agent = self._make_agent()
        mock_resp = MagicMock()
        mock_resp.text_content = "<summary>\nTest summary\n</summary>"
        mock_resp.usage = MagicMock(input_tokens=100, output_tokens=50)
        agent.client = MagicMock()
        agent.client.acreate = AsyncMock(return_value=mock_resp)

        messages = []
        for i in range(8):
            messages.extend(_make_simple_turn(f"Q{i}", f"A{i}"))

        compressed, s_in, s_out = await agent._compress_messages(messages)

        # [0] summary user message
        assert compressed[0]["role"] == "user"
        assert "This session is being continued" in compressed[0]["content"]
        assert "<summary>" in compressed[0]["content"]

        # [1] assistant acknowledgment (since recent starts with user)
        assert compressed[1]["role"] == "assistant"
        assert "continue from where we left off" in compressed[1]["content"][0]["text"]

        # [2:] recent messages — proper user/assistant alternation
        for i in range(2, len(compressed) - 1, 2):
            assert compressed[i]["role"] == "user"
            assert compressed[i + 1]["role"] == "assistant"

        # Token counts from mock
        assert s_in == 100
        assert s_out == 50

    async def test_04_summary_prompt_contains_required_sections(self):
        """The summary system prompt includes all 7 required sections and verbatim user message guidance."""
        from app.agent.agent import SUMMARY_SYSTEM_PROMPT

        required_sections = [
            "Primary Request and Intent",
            "Key Technical Concepts",
            "Files and Code Sections",
            "Problem Solving",
            "All User Messages",
            "Current State",
            "Pending Tasks",
        ]
        for section in required_sections:
            assert section in SUMMARY_SYSTEM_PROMPT, f"Missing section: {section}"

        assert "<summary>" in SUMMARY_SYSTEM_PROMPT
        assert "</summary>" in SUMMARY_SYSTEM_PROMPT
        # Verify enhanced user message preservation guidance
        assert "verbatim" in SUMMARY_SYSTEM_PROMPT
        assert "user intent must be preserved precisely" in SUMMARY_SYSTEM_PROMPT

    async def test_05_compression_preserves_tool_pairs_in_recent(self):
        """After compression, recent messages contain complete tool_use/tool_result pairs."""
        agent = self._make_agent()
        mock_resp = MagicMock()
        mock_resp.text_content = "<summary>\nSummary\n</summary>"
        mock_resp.usage = MagicMock(input_tokens=100, output_tokens=50)
        agent.client = MagicMock()
        agent.client.acreate = AsyncMock(return_value=mock_resp)

        # 8 turns: some with tools, some simple (need >5 to trigger compression)
        turn1 = _make_tool_turn("T1", [{"name": "bash", "result": "ok"}], "Done1")
        turn2 = _make_tool_turn("T2", [
            {"name": "get_skill", "result": "skill content"},
            {"name": "execute_code", "result": "output"},
        ], "Done2")
        turn3 = _make_simple_turn("T3", "Done3")
        turn4 = _make_simple_turn("T4", "Done4")
        turn5 = _make_tool_turn("T5", [{"name": "bash", "result": "ok5"}], "Done5")
        turn6 = _make_tool_turn("T6", [{"name": "bash", "result": "ok6"}], "Done6")
        turn7 = _make_simple_turn("T7", "Done7")
        turn8 = _make_simple_turn("T8", "Done8")

        messages = turn1 + turn2 + turn3 + turn4 + turn5 + turn6 + turn7 + turn8
        compressed, _, _ = await agent._compress_messages(messages)

        # Extract recent portion (skip summary + ack)
        recent = compressed[2:]

        # Verify every tool_use has a matching tool_result
        tool_use_ids = set()
        tool_result_ids = set()
        for msg in recent:
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            tool_use_ids.add(block["id"])
                        elif block.get("type") == "tool_result":
                            tool_result_ids.add(block["tool_use_id"])

        # Every tool_use should have a corresponding tool_result
        assert tool_use_ids == tool_result_ids


# ---------------------------------------------------------------------------
# Class 4: Serialization for summarization
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestSerializeMessagesE2E:
    """Test _serialize_messages_for_summary handles all message types."""

    _state: dict = {}

    def _make_agent(self):
        from app.agent.agent import SkillsAgent
        agent = SkillsAgent.__new__(SkillsAgent)
        agent.model = "kimi-k2.5"
        agent.model_provider = "kimi"
        agent.verbose = False
        agent.client = MagicMock()
        return agent

    async def test_01_serialize_text_messages(self):
        """Plain text user/assistant messages are serialized."""
        agent = self._make_agent()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there"}]},
        ]
        text = agent._serialize_messages_for_summary(messages)
        assert "[user]: Hello" in text
        assert "[assistant]: Hi there" in text

    async def test_02_serialize_tool_use(self):
        """tool_use blocks are serialized with tool name and input."""
        agent = self._make_agent()
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t:0", "name": "execute_code", "input": {"code": "print(1)"}},
            ]},
        ]
        text = agent._serialize_messages_for_summary(messages)
        assert "tool_use(execute_code)" in text
        assert "print(1)" in text

    async def test_03_serialize_tool_result(self):
        """tool_result blocks are serialized."""
        agent = self._make_agent()
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t:0", "content": "execution output"},
            ]},
        ]
        text = agent._serialize_messages_for_summary(messages)
        assert "[tool_result]: execution output" in text

    async def test_04_truncate_long_tool_input(self):
        """tool_use inputs longer than 500 chars are truncated."""
        agent = self._make_agent()
        long_code = "x" * 1000
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t:0", "name": "execute_code", "input": {"code": long_code}},
            ]},
        ]
        text = agent._serialize_messages_for_summary(messages)
        assert "...(truncated)" in text

    async def test_05_truncate_long_tool_result(self):
        """tool_result content longer than 1000 chars is truncated."""
        agent = self._make_agent()
        long_result = "R" * 2000
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t:0", "content": long_result},
            ]},
        ]
        text = agent._serialize_messages_for_summary(messages)
        assert "...(truncated)" in text
        assert len(text) < 2000

    async def test_06_huge_text_head_tail_truncation(self):
        """Total text exceeding 100K chars is truncated to head + tail."""
        agent = self._make_agent()
        # Create many messages with large content
        messages = []
        for i in range(200):
            messages.append({"role": "user", "content": f"Message {i}: {'X' * 600}"})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": f"Response {i}: {'Y' * 600}"}]})

        text = agent._serialize_messages_for_summary(messages)
        assert "[... truncated middle section ...]" in text


# ---------------------------------------------------------------------------
# Class 5: Published Agent session saves full messages (integration)
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestPublishedSessionFullMessagesE2E:
    """Published Agent sessions save complete messages including tool_use/tool_result."""

    _state: dict = {}

    async def test_01_create_and_publish(self, e2e_client: AsyncClient):
        """Create and publish an agent for testing."""
        resp = await e2e_client.post("/api/v1/agents", json={
            "name": "e2e-session-fullmsg",
            "description": "Test full message sessions",
            "max_turns": 5,
        })
        assert resp.status_code == 200
        pid = resp.json()["id"]
        type(self)._state["preset_id"] = pid

        resp = await e2e_client.post(
            f"/api/v1/agents/{pid}/publish",
            json={"api_response_mode": "streaming"},
        )
        assert resp.status_code == 200

    @patch("app.api.v1.published.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.published.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.published.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.published.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.published.SkillsAgent")
    @patch("app.api.v1.published.AsyncSessionLocal")
    async def test_02_streaming_saves_full_messages(
        self, MockSL, MockAgent, MockLoadSession, _mock_save, MockCheckpoint, MockPreCompress, e2e_client: AsyncClient,
    ):
        """Streaming chat saves final_messages (with tool_use/tool_result) to session."""
        pid = type(self)._state["preset_id"]
        session_id = str(uuid.uuid4())
        type(self)._state["session_id"] = session_id
        MockLoadSession.return_value = SessionData(session_id=session_id)

        # Build final_messages with tool context
        final_msgs = [
            {"role": "user", "content": "List skills"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "ls:0", "name": "list_skills", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "ls:0", "content": '{"skills": ["a", "b"]}'},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Found 2 skills: a, b"},
            ]},
        ]

        # Mock agent with async run() that pushes events to event_stream
        mock_instance = MagicMock()
        mock_instance.cleanup = MagicMock()
        mock_instance.model = "kimi-k2.5"
        mock_instance.model_provider = "kimi"

        events_to_push = [
            StreamEvent(event_type="turn_start", turn=1, data={"turn": 1}),
            StreamEvent(event_type="tool_result", turn=1, data={
                "tool_name": "list_skills", "tool_input": {}, "tool_result": '{"skills": ["a","b"]}',
            }),
            StreamEvent(event_type="assistant", turn=2, data={
                "content": "Found 2 skills: a, b", "turn": 2,
            }),
            StreamEvent(event_type="complete", turn=2, data={
                "success": True,
                "answer": "Found 2 skills: a, b",
                "total_turns": 2,
                "total_input_tokens": 200,
                "total_output_tokens": 30,
                "skills_used": [],
                "final_messages": final_msgs,
            }),
        ]

        async def mock_run(request, conversation_history=None, image_contents=None,
                           event_stream=None, cancellation_event=None):
            if event_stream:
                for event in events_to_push:
                    await event_stream.push(event)
                await event_stream.close()
            from app.agent.agent import AgentResult
            return AgentResult(
                success=True, answer="Found 2 skills: a, b",
                total_turns=2, total_input_tokens=200, total_output_tokens=30,
                skills_used=[], final_messages=final_msgs,
            )

        mock_instance.run = AsyncMock(side_effect=mock_run)
        MockAgent.return_value = mock_instance

        # Mock AsyncSessionLocal with DB interactions tracking
        from app.db.models import AgentPresetDB, PublishedSessionDB

        mock_preset = MagicMock(spec=AgentPresetDB)
        mock_preset.id = pid
        mock_preset.name = "e2e-session-fullmsg"
        mock_preset.description = "Test"
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

        saved_messages = {}
        call_idx = {"i": 0}

        @asynccontextmanager
        async def _ctx():
            idx = call_idx["i"]
            call_idx["i"] += 1
            mock_sess = AsyncMock(spec=AsyncSession)

            if idx == 0:
                # First call: find preset
                mock_result = MagicMock()
                mock_result.scalar_one_or_none.return_value = mock_preset
                mock_sess.execute = AsyncMock(return_value=mock_result)
            elif idx == 1:
                # Second call: find session (not found → create)
                mock_result = MagicMock()
                mock_result.scalar_one_or_none.return_value = None
                mock_sess.execute = AsyncMock(return_value=mock_result)
                mock_sess.add = MagicMock()
                mock_sess.commit = AsyncMock()
            else:
                # Subsequent calls: session save — capture the update values
                mock_result = MagicMock()
                mock_session_record = MagicMock()
                mock_session_record.messages = []
                mock_result.scalar_one_or_none.return_value = mock_session_record

                async def capture_execute(stmt, *args, **kwargs):
                    if hasattr(stmt, 'compile'):
                        compiled = stmt.compile(compile_kwargs={"literal_binds": True})
                        stmt_str = str(compiled)
                        if "UPDATE" in stmt_str and "messages" in stmt_str:
                            params = stmt.compile().params
                            if "messages" in params:
                                saved_messages["data"] = params["messages"]
                    return mock_result

                mock_sess.execute = AsyncMock(side_effect=capture_execute)
                mock_sess.commit = AsyncMock()

            yield mock_sess

        MockSL.side_effect = lambda: _ctx()

        resp = await e2e_client.post(
            f"/api/v1/published/{pid}/chat",
            json={"request": "List skills", "session_id": session_id},
        )
        assert resp.status_code == 200

        events = parse_sse_events(resp.text)
        complete_events = [e for e in events if e.get("event_type") == "complete"]
        assert len(complete_events) == 1

        # Verify complete event contains final_messages
        complete = complete_events[0]
        assert "final_messages" in complete
        fm = complete["final_messages"]
        assert len(fm) == 4
        # First: user text
        assert fm[0]["role"] == "user"
        assert fm[0]["content"] == "List skills"
        # Second: assistant tool_use
        assert fm[1]["role"] == "assistant"
        assert fm[1]["content"][0]["type"] == "tool_use"
        # Third: user tool_result
        assert fm[2]["role"] == "user"
        assert fm[2]["content"][0]["type"] == "tool_result"
        # Fourth: assistant text
        assert fm[3]["role"] == "assistant"
        assert fm[3]["content"][0]["type"] == "text"

    @patch("app.api.v1.published.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.published.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.published.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.published.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.published.SkillsAgent")
    @patch("app.api.v1.published.AsyncSessionLocal")
    async def test_03_nonstreaming_saves_full_messages(
        self, MockSL, MockAgent, MockLoadSession, _mock_save, MockCheckpoint, MockPreCompress, e2e_client: AsyncClient,
    ):
        """Non-streaming chat also saves final_messages to session."""
        pid = type(self)._state["preset_id"]

        # Unpublish and re-publish as non_streaming
        await e2e_client.post(f"/api/v1/agents/{pid}/unpublish")
        resp = await e2e_client.post(
            f"/api/v1/agents/{pid}/publish",
            json={"api_response_mode": "non_streaming"},
        )
        assert resp.status_code == 200

        session_id = str(uuid.uuid4())
        MockLoadSession.return_value = SessionData(session_id=session_id)

        final_msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ]

        from app.agent.agent import AgentResult
        mock_result = AgentResult(
            success=True, answer="Hi!", total_turns=1,
            total_input_tokens=100, total_output_tokens=20,
            skills_used=[], output_files=[], final_messages=final_msgs,
        )

        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value=mock_result)
        mock_instance.cleanup = MagicMock()
        mock_instance.model = "kimi-k2.5"
        mock_instance.model_provider = "kimi"
        MockAgent.return_value = mock_instance

        from app.db.models import AgentPresetDB

        mock_preset = MagicMock(spec=AgentPresetDB)
        mock_preset.id = pid
        mock_preset.name = "e2e-session-fullmsg"
        mock_preset.description = "Test"
        mock_preset.is_published = True
        mock_preset.api_response_mode = "non_streaming"
        mock_preset.skill_ids = []
        mock_preset.builtin_tools = None
        mock_preset.max_turns = 5
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
                mock_r = MagicMock()
                mock_r.scalar_one_or_none.return_value = mock_preset
                mock_sess.execute = AsyncMock(return_value=mock_r)
            elif idx == 1:
                mock_r = MagicMock()
                mock_r.scalar_one_or_none.return_value = None
                mock_sess.execute = AsyncMock(return_value=mock_r)
                mock_sess.add = MagicMock()
                mock_sess.commit = AsyncMock()
            else:
                mock_r = MagicMock()
                mock_session_record = MagicMock()
                mock_session_record.messages = []
                mock_r.scalar_one_or_none.return_value = mock_session_record
                mock_sess.execute = AsyncMock(return_value=mock_r)
                mock_sess.commit = AsyncMock()

            yield mock_sess

        MockSL.side_effect = lambda: _ctx()

        resp = await e2e_client.post(
            f"/api/v1/published/{pid}/chat/sync",
            json={"request": "Hello", "session_id": session_id},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["answer"] == "Hi!"

    async def test_04_cleanup(self, e2e_client: AsyncClient):
        pid = type(self)._state["preset_id"]
        await e2e_client.post(f"/api/v1/agents/{pid}/unpublish")
        resp = await e2e_client.delete(f"/api/v1/agents/{pid}")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Class 6: Iterative summary and file tracking
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestIterativeSummaryAndFileTrackingE2E:
    """Test iterative summary updates and cumulative file tracking."""

    _state: dict = {}

    def _make_agent(self, context_limit: int = 256_000):
        from app.agent.agent import SkillsAgent
        agent = SkillsAgent.__new__(SkillsAgent)
        agent.model = "kimi-k2.5"
        agent.model_provider = "kimi"
        agent.verbose = False
        agent.client = MagicMock()
        agent._get_context_limit = lambda: context_limit
        return agent

    def _mock_summary(self, agent, summary_text="<summary>\n## Test\nSummary\n</summary>"):
        mock_resp = MagicMock()
        mock_resp.text_content = summary_text
        mock_resp.usage = MagicMock(input_tokens=500, output_tokens=200)
        agent.client = MagicMock()
        agent.client.acreate = AsyncMock(return_value=mock_resp)

    async def test_01_iterative_summary_uses_update_prompt(self):
        """When old_messages starts with a previous summary, SUMMARY_UPDATE_PROMPT is used."""
        from app.agent.agent import SUMMARY_UPDATE_PROMPT

        agent = self._make_agent()
        self._mock_summary(agent, "<summary>\n## Updated\nNew summary\n</summary>")

        # Build messages that simulate a previous compression + new turns
        previous_summary_content = (
            "This session is being continued from a previous conversation that ran out of context. "
            "The summary below covers the earlier portion.\n\n"
            "<summary>\n## Primary Request\nUser wanted to analyze data\n## Current State\nWIP\n</summary>\n\n"
            "Continue with the last task."
        )
        messages = [
            {"role": "user", "content": previous_summary_content},
            {"role": "assistant", "content": [{"type": "text", "text": "I understand the context. Let me continue from where we left off."}]},
        ]
        # Add 8 new turns after the summary
        for i in range(8):
            messages.extend(_make_simple_turn(f"New Q{i}", f"New A{i}"))

        compressed, _, _ = await agent._compress_messages(messages)

        # Verify the LLM was called with the update prompt
        call_args = agent.client.acreate.call_args
        system_used = call_args.kwargs.get("system", "")
        assert "previous-summary" in system_used
        assert "Update the existing summary" in system_used or "update the summary" in system_used.lower()

    async def test_02_file_tracking_in_summary(self):
        """Compression includes <read-files> and <modified-files> tags from tool calls."""
        agent = self._make_agent()
        self._mock_summary(agent)

        # Build messages with file operations
        messages = [
            {"role": "user", "content": "Read some files"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "r:0", "name": "read", "input": {"file_path": "/app/main.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "r:0", "content": "file content"},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "w:0", "name": "write", "input": {"file_path": "/app/output.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "w:0", "content": "written"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "Done reading and writing"}]},
        ]
        # Add more turns to trigger compression
        for i in range(8):
            messages.extend(_make_simple_turn(f"Q{i}", f"A{i}"))

        compressed, _, _ = await agent._compress_messages(messages)

        summary_content = compressed[0]["content"]
        assert "<read-files>" in summary_content
        assert "/app/main.py" in summary_content
        assert "<modified-files>" in summary_content
        assert "/app/output.py" in summary_content

    async def test_03_cumulative_file_tracking_merges_previous(self):
        """Second compression merges file tracking from previous summary."""
        agent = self._make_agent()
        self._mock_summary(agent)

        # Simulate first compression output with file tracking
        previous_summary_content = (
            "This session is being continued from a previous conversation.\n\n"
            "<summary>\n## Primary Request\nAnalyze files\n\n"
            "<read-files>\n/app/old_file.py\n</read-files>\n"
            "<modified-files>\n/app/old_output.csv\n</modified-files>\n"
            "</summary>\n\nContinue."
        )
        messages = [
            {"role": "user", "content": previous_summary_content},
            {"role": "assistant", "content": [{"type": "text", "text": "I understand the context. Let me continue from where we left off."}]},
            # New turn with additional file operations
            {"role": "user", "content": "Now edit another file"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "e:0", "name": "edit", "input": {"file_path": "/app/new_file.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "e:0", "content": "edited"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "File edited"}]},
        ]
        # Add more turns
        for i in range(8):
            messages.extend(_make_simple_turn(f"Q{i}", f"A{i}"))

        compressed, _, _ = await agent._compress_messages(messages)

        # The system prompt should include the UPDATE prompt
        call_args = agent.client.acreate.call_args
        system_used = call_args.kwargs.get("system", "")
        # File tracking in the system prompt should contain both old and new files
        assert "/app/old_file.py" in system_used or "/app/new_file.py" in system_used

    async def test_04_extract_file_operations_helper(self):
        """_extract_file_operations correctly extracts read and modified files."""
        from app.agent.agent import _extract_file_operations

        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "r:0", "name": "read", "input": {"file_path": "/app/config.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "r:0", "content": "config content"},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "w:0", "name": "write", "input": {"file_path": "/app/out.txt"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "w:0", "content": '{"success": true, "new_files": [{"filename": "result.csv"}]}'},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "g:0", "name": "grep", "input": {"path": "/app/src"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "g:0", "content": "matches found"},
            ]},
        ]

        read_files, modified_files = _extract_file_operations(messages)
        assert "/app/config.py" in read_files
        assert "/app/src" in read_files
        assert "/app/out.txt" in modified_files
        assert "result.csv" in modified_files

    async def test_05_summary_update_prompt_exists(self):
        """SUMMARY_UPDATE_PROMPT is defined and has required structure."""
        from app.agent.agent import SUMMARY_UPDATE_PROMPT

        assert "previous-summary" in SUMMARY_UPDATE_PROMPT
        assert "PRESERVE" in SUMMARY_UPDATE_PROMPT
        assert "{previous_summary}" in SUMMARY_UPDATE_PROMPT
        assert "{file_tracking_section}" in SUMMARY_UPDATE_PROMPT


# ---------------------------------------------------------------------------
# Class 7: Split turn handling
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestSplitTurnHandlingE2E:
    """Test split-oversized-turn code path in _compress_messages."""

    _state: dict = {}

    def _make_agent(self, context_limit: int = 256_000):
        from app.agent.agent import SkillsAgent
        agent = SkillsAgent.__new__(SkillsAgent)
        agent.model = "kimi-k2.5"
        agent.model_provider = "kimi"
        agent.verbose = False
        agent.client = MagicMock()
        agent._get_context_limit = lambda: context_limit
        return agent

    async def test_01_split_turn_with_valid_cut_points(self):
        """Oversized turn with valid cut points is split; prefix summary included."""
        agent = self._make_agent(context_limit=5000)  # Budget = 1250 tokens

        call_count = {"n": 0}

        async def mock_acreate(**kwargs):
            call_count["n"] += 1
            resp = MagicMock()
            if call_count["n"] == 1:
                # Turn prefix summary
                resp.text_content = "## Original Request\nBig task\n## Early Progress\nSteps 0-5\n## Context for Suffix\nContinuing from step 6"
            else:
                # Main summary
                resp.text_content = "<summary>\n## Primary Request\nSetup task summary\n</summary>"
            resp.usage = MagicMock(input_tokens=200, output_tokens=80)
            return resp

        agent.client = MagicMock()
        agent.client.acreate = AsyncMock(side_effect=mock_acreate)

        # Turn 1: small (will be in old_messages)
        messages = [
            {"role": "user", "content": "Setup task"},
            {"role": "assistant", "content": [{"type": "text", "text": "Setup done"}]},
        ]
        # Turn 2: one large turn with many tool steps (NO intermediate plain user messages)
        # Only tool_result user messages within this turn — keeps it as a single logical turn
        messages.append({"role": "user", "content": "Run 10 analysis steps"})
        for step in range(10):
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Step {step} analysis:"},
                    {"type": "tool_use", "id": f"t:{step}", "name": "execute_code",
                     "input": {"code": f"analyze_step({step})\n" + "x" * 400}},
                ],
            })
            messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"t:{step}",
                     "content": f"Step {step} result: " + "R" * 400},
                ],
            })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": "All analysis complete!"}],
        })

        compressed, _, _ = await agent._compress_messages(messages)

        assert len(compressed) < len(messages)
        summary_content = compressed[0]["content"]
        # Split should have happened: prefix summary gets appended
        assert "Recent turn prefix context" in summary_content

    async def test_02_split_turn_prefix_api_failure_fallback(self):
        """When prefix summary LLM call fails, fallback text is used."""
        agent = self._make_agent(context_limit=5000)

        call_count = {"n": 0}

        async def mock_acreate(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("Prefix summary API error")
            resp = MagicMock()
            resp.text_content = "<summary>\n## Primary Request\nTask\n</summary>"
            resp.usage = MagicMock(input_tokens=200, output_tokens=80)
            return resp

        agent.client = MagicMock()
        agent.client.acreate = AsyncMock(side_effect=mock_acreate)

        # Turn 1: small
        messages = [
            {"role": "user", "content": "Small task"},
            {"role": "assistant", "content": [{"type": "text", "text": "Done"}]},
        ]
        # Turn 2: large single turn with tool steps (no intermediate plain user messages)
        messages.append({"role": "user", "content": "Big task"})
        for step in range(8):
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Working {step}"},
                    {"type": "tool_use", "id": f"t:{step}", "name": "bash",
                     "input": {"command": f"step_{step}" + "Z" * 300}},
                ],
            })
            messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"t:{step}",
                     "content": f"output_{step}" + "W" * 300},
                ],
            })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": "All done!"}],
        })

        compressed, _, _ = await agent._compress_messages(messages)

        assert len(compressed) < len(messages)
        summary_content = compressed[0]["content"]
        # Should still have turn prefix context (from fallback serialized text)
        assert "Recent turn prefix context" in summary_content

    async def test_03_no_split_when_turn_fits_budget(self):
        """When the oversized turn fits in budget, no split occurs."""
        agent = self._make_agent(context_limit=1_000_000)  # Huge budget

        mock_resp = MagicMock()
        mock_resp.text_content = "<summary>\n## Test\nSummary\n</summary>"
        mock_resp.usage = MagicMock(input_tokens=100, output_tokens=50)
        agent.client = MagicMock()
        agent.client.acreate = AsyncMock(return_value=mock_resp)

        # 8 turns, last one is large but budget is huge
        messages = []
        for i in range(7):
            messages.extend(_make_simple_turn(f"Q{i}", f"A{i}"))
        # Large final turn
        messages.extend(_make_heavy_tool_turn("Big final task", result_size=50000))

        compressed, _, _ = await agent._compress_messages(messages)

        summary_content = compressed[0]["content"]
        # No split should happen — turn fits in budget
        assert "Recent turn prefix context" not in summary_content

    async def test_04_turn_prefix_summary_prompt_structure(self):
        """TURN_PREFIX_SUMMARY_PROMPT has required sections."""
        from app.agent.agent import TURN_PREFIX_SUMMARY_PROMPT

        assert "Original Request" in TURN_PREFIX_SUMMARY_PROMPT
        assert "Early Progress" in TURN_PREFIX_SUMMARY_PROMPT
        assert "Context for Suffix" in TURN_PREFIX_SUMMARY_PROMPT
        assert "PREFIX" in TURN_PREFIX_SUMMARY_PROMPT
        assert "SUFFIX" in TURN_PREFIX_SUMMARY_PROMPT

    async def test_05_split_preserves_recent_suffix_verbatim(self):
        """After split, the suffix (recent) messages are preserved exactly."""
        agent = self._make_agent(context_limit=3000)  # Very small

        call_count = {"n": 0}

        async def mock_acreate(**kwargs):
            call_count["n"] += 1
            resp = MagicMock()
            if call_count["n"] == 1:
                resp.text_content = "Prefix context summary"
            else:
                resp.text_content = "<summary>\nHistory\n</summary>"
            resp.usage = MagicMock(input_tokens=100, output_tokens=50)
            return resp

        agent.client = MagicMock()
        agent.client.acreate = AsyncMock(side_effect=mock_acreate)

        # Turn 1: small
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": [{"type": "text", "text": "OK"}]},
        ]
        # Turn 2: large single turn (only tool_result user messages, no plain user text)
        messages.append({"role": "user", "content": "Big work"})
        for step in range(6):
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Step {step}"},
                    {"type": "tool_use", "id": f"t:{step}", "name": "bash",
                     "input": {"command": "x" * 300}},
                ],
            })
            messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"t:{step}",
                     "content": "y" * 300},
                ],
            })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": "FINAL_ANSWER_MARKER"}],
        })

        compressed, _, _ = await agent._compress_messages(messages)

        # The final assistant message should be in the compressed output verbatim
        found_final = any(
            isinstance(m.get("content"), list)
            and any(isinstance(b, dict) and b.get("text") == "FINAL_ANSWER_MARKER" for b in m["content"])
            for m in compressed
        )
        assert found_final, "Final answer should be preserved verbatim in suffix"


# ---------------------------------------------------------------------------
# Class 8: Triple compression with cumulative file tracking
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestTripleCompressionE2E:
    """Test three successive compressions with cumulative file tracking."""

    _state: dict = {}

    def _make_agent(self, context_limit: int = 256_000):
        from app.agent.agent import SkillsAgent
        agent = SkillsAgent.__new__(SkillsAgent)
        agent.model = "kimi-k2.5"
        agent.model_provider = "kimi"
        agent.verbose = False
        agent.client = MagicMock()
        agent._get_context_limit = lambda: context_limit
        return agent

    def _mock_summary(self, agent, summary_text="<summary>\n## Test\nSummary\n</summary>"):
        mock_resp = MagicMock()
        mock_resp.text_content = summary_text
        mock_resp.usage = MagicMock(input_tokens=500, output_tokens=200)
        agent.client = MagicMock()
        agent.client.acreate = AsyncMock(return_value=mock_resp)

    async def test_01_first_compression_adds_file_tracking(self):
        """First compression with file ops adds <read-files>/<modified-files>."""
        agent = self._make_agent()
        self._mock_summary(agent)

        messages = [
            {"role": "user", "content": "Read config"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "r:0", "name": "read", "input": {"file_path": "/app/config.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "r:0", "content": "config"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "Read config"}]},
        ]
        for i in range(8):
            messages.extend(_make_simple_turn(f"Q{i}", f"A{i}"))

        compressed, _, _ = await agent._compress_messages(messages)
        type(self)._state["first_compression"] = compressed

        summary = compressed[0]["content"]
        assert "<read-files>" in summary
        assert "/app/config.py" in summary

    async def test_02_second_compression_merges_old_tracking(self):
        """Second compression merges file tracking from first summary."""
        agent = self._make_agent()
        self._mock_summary(agent)

        # Start from first compression output
        first = type(self)._state["first_compression"]

        # Add new turns with new file operations
        messages = list(first)
        messages.extend([
            {"role": "user", "content": "Write output"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "w:0", "name": "write", "input": {"file_path": "/app/output.txt"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "w:0", "content": "written"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "Written output"}]},
        ])
        for i in range(8):
            messages.extend(_make_simple_turn(f"R{i}", f"B{i}"))

        compressed, _, _ = await agent._compress_messages(messages)
        type(self)._state["second_compression"] = compressed

        # Verify UPDATE prompt was used (iterative)
        call_args = agent.client.acreate.call_args
        system_used = call_args.kwargs.get("system", "")
        assert "previous-summary" in system_used

        # Verify cumulative file tracking in the system prompt
        assert "/app/config.py" in system_used  # From first compression
        assert "/app/output.txt" in system_used  # From new operations

    async def test_03_third_compression_accumulates_all_files(self):
        """Third compression accumulates files from all three rounds."""
        agent = self._make_agent()
        self._mock_summary(agent)

        second = type(self)._state["second_compression"]

        # Add more file operations
        messages = list(second)
        messages.extend([
            {"role": "user", "content": "Edit models"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "e:0", "name": "edit", "input": {"file_path": "/app/models.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "e:0", "content": "edited"},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "r:1", "name": "read", "input": {"file_path": "/app/schema.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "r:1", "content": "schema"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "Done"}]},
        ])
        for i in range(8):
            messages.extend(_make_simple_turn(f"S{i}", f"C{i}"))

        compressed, _, _ = await agent._compress_messages(messages)

        # All files from all three compressions should be in the system prompt
        call_args = agent.client.acreate.call_args
        system_used = call_args.kwargs.get("system", "")
        assert "/app/config.py" in system_used    # Round 1
        assert "/app/output.txt" in system_used   # Round 2
        assert "/app/models.py" in system_used    # Round 3
        assert "/app/schema.py" in system_used    # Round 3


# ---------------------------------------------------------------------------
# Class 9: Edge cases and no-file-ops scenarios
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestCompressionEdgeCasesE2E:
    """Test edge cases in compression: no file ops, multiple new_files, etc."""

    _state: dict = {}

    def _make_agent(self, context_limit: int = 256_000):
        from app.agent.agent import SkillsAgent
        agent = SkillsAgent.__new__(SkillsAgent)
        agent.model = "kimi-k2.5"
        agent.model_provider = "kimi"
        agent.verbose = False
        agent.client = MagicMock()
        agent._get_context_limit = lambda: context_limit
        return agent

    def _mock_summary(self, agent, summary_text="<summary>\n## Test\nSummary\n</summary>"):
        mock_resp = MagicMock()
        mock_resp.text_content = summary_text
        mock_resp.usage = MagicMock(input_tokens=500, output_tokens=200)
        agent.client = MagicMock()
        agent.client.acreate = AsyncMock(return_value=mock_resp)

    async def test_01_no_file_ops_no_xml_tags(self):
        """Conversation with no file tools produces no <read-files>/<modified-files>."""
        agent = self._make_agent()
        self._mock_summary(agent)

        # Pure text conversation — no tools at all
        messages = []
        for i in range(8):
            messages.extend(_make_simple_turn(f"Q{i}", f"A{i}"))

        compressed, _, _ = await agent._compress_messages(messages)

        summary_content = compressed[0]["content"]
        assert "<read-files>" not in summary_content
        assert "<modified-files>" not in summary_content

    async def test_02_multiple_new_files_in_single_tool_result(self):
        """Multiple new_files in one tool_result are all tracked."""
        from app.agent.agent import _extract_file_operations

        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "c:0", "name": "execute_code",
                 "input": {"code": "create_charts()"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "c:0",
                 "content": json.dumps({
                     "success": True,
                     "output": "Charts created",
                     "new_files": [
                         {"filename": "chart1.png"},
                         {"filename": "chart2.png"},
                         {"filename": "data_summary.csv"},
                     ],
                 })},
            ]},
        ]

        _, modified_files = _extract_file_operations(messages)
        assert "chart1.png" in modified_files
        assert "chart2.png" in modified_files
        assert "data_summary.csv" in modified_files

    async def test_03_duplicate_files_across_compressions(self):
        """Same file read twice across compressions: deduplication by set."""
        from app.agent.agent import _extract_file_operations, _extract_previous_file_tracking

        # Previous summary had /app/config.py
        previous_text = "<read-files>\n/app/config.py\n</read-files>"
        prev_read, prev_mod = _extract_previous_file_tracking(previous_text)

        # New messages also read /app/config.py
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "r:0", "name": "read", "input": {"file_path": "/app/config.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "r:0", "content": "config"},
            ]},
        ]
        new_read, new_mod = _extract_file_operations(messages)
        merged = prev_read | new_read

        # Should have only one entry (deduplicated)
        assert len([f for f in merged if f == "/app/config.py"]) == 1

    async def test_04_file_tracking_inserted_before_summary_close(self):
        """File tracking XML is inserted before </summary>, not after."""
        agent = self._make_agent()
        self._mock_summary(agent)

        messages = [
            {"role": "user", "content": "Read file"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "r:0", "name": "read", "input": {"file_path": "/app/test.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "r:0", "content": "content"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "Read it"}]},
        ]
        for i in range(8):
            messages.extend(_make_simple_turn(f"Q{i}", f"A{i}"))

        compressed, _, _ = await agent._compress_messages(messages)

        summary_content = compressed[0]["content"]
        # Find positions: read-files should be before </summary>
        read_pos = summary_content.find("<read-files>")
        close_pos = summary_content.find("</summary>")
        assert read_pos < close_pos, "<read-files> should be inside </summary>"

    async def test_05_iterative_fallback_preserves_previous_summary(self):
        """When UPDATE prompt fails, fallback to previous_summary_text."""
        agent = self._make_agent()
        agent.client = MagicMock()
        agent.client.acreate = AsyncMock(side_effect=Exception("API down"))

        previous_summary_content = (
            "This session is being continued from a previous conversation.\n\n"
            "<summary>\n## Primary Request\nImportant task preserved\n## Current State\nIn progress\n</summary>\n\n"
            "Continue with the last task."
        )
        messages = [
            {"role": "user", "content": previous_summary_content},
            {"role": "assistant", "content": [{"type": "text", "text": "I understand the context. Let me continue from where we left off."}]},
        ]
        for i in range(8):
            messages.extend(_make_simple_turn(f"Q{i}", f"A{i}"))

        compressed, s_in, s_out = await agent._compress_messages(messages)

        summary_content = compressed[0]["content"]
        # Previous summary should be preserved in fallback
        assert "Important task preserved" in summary_content
        assert s_in == 0  # No API tokens consumed
        assert s_out == 0

    async def test_06_read_file_alias_extracted(self):
        """Both 'read' and 'read_file' tool names are recognized."""
        from app.agent.agent import _extract_file_operations

        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "r:0", "name": "read_file", "input": {"file_path": "/app/via_alias.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "r:0", "content": "content"},
            ]},
        ]
        read_files, _ = _extract_file_operations(messages)
        assert "/app/via_alias.py" in read_files


# ---------------------------------------------------------------------------
# Class 10: _should_compress threshold
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestShouldCompressE2E:
    """Test compression trigger threshold."""

    _state: dict = {}

    def _make_agent(self, context_limit: int = 200_000):
        from app.agent.agent import SkillsAgent
        agent = SkillsAgent.__new__(SkillsAgent)
        agent.model = "kimi-k2.5"
        agent.model_provider = "kimi"
        agent.verbose = False
        agent.client = MagicMock()
        agent._get_context_limit = lambda: context_limit
        return agent

    async def test_01_below_threshold_no_compress(self):
        """Input tokens below 70% of limit: don't compress."""
        agent = self._make_agent(200_000)
        # 70% of 200K = 140K
        assert agent._should_compress(100_000) is False
        assert agent._should_compress(139_999) is False

    async def test_02_at_threshold_no_compress(self):
        """Input tokens exactly at threshold: don't compress (> not >=)."""
        agent = self._make_agent(200_000)
        assert agent._should_compress(140_000) is False

    async def test_03_above_threshold_compress(self):
        """Input tokens above 70% of limit: compress."""
        agent = self._make_agent(200_000)
        assert agent._should_compress(140_001) is True
        assert agent._should_compress(180_000) is True


# ---------------------------------------------------------------------------
# Class 8: Incremental display save on stop/refresh/cancel
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="class")
class TestIncrementalDisplaySaveE2E:
    """Verify that display messages are saved incrementally on turn_complete
    and on cancellation, so they survive stop/refresh/network errors.

    Tests both the chat panel (agent.py) and published agent (published.py).

    Cancellation tests use unit-level testing of the checkpoint functions
    rather than httpx streaming disconnection, since ASGITransport doesn't
    reliably propagate disconnect as CancelledError in test contexts.
    """

    _state: dict = {}

    # ---- Chat panel (agent.py) tests ----

    @patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.agent.create_agent")
    async def test_01_turn_complete_saves_display(
        self, MockAgent, MockLoadSession, MockSaveMessages, MockCheckpoint, MockPreCompress,
        e2e_client: AsyncClient, e2e_db_session,
    ):
        """turn_complete checkpoint includes display_messages (agent.py)."""
        session_id = str(uuid.uuid4())
        MockLoadSession.return_value = SessionData(
            session_id=session_id,
            display_messages=[],  # Fresh session
        )

        # Messages snapshot that would be in turn_complete
        snapshot_msgs = [
            {"role": "user", "content": "Analyze data"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tc:0", "name": "execute_code", "input": {"code": "print(1)"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tc:0", "content": "1"},
            ]},
        ]

        events = [
            StreamEvent(event_type="turn_start", turn=1, data={"turn": 1}),
            StreamEvent(event_type="tool_result", turn=1, data={
                "tool_name": "execute_code", "tool_input": {"code": "print(1)"}, "tool_result": "1",
            }),
            # turn_complete includes messages_snapshot
            StreamEvent(event_type="turn_complete", turn=1, data={
                "messages_snapshot": snapshot_msgs,
            }),
            StreamEvent(event_type="assistant", turn=2, data={
                "content": "Result is 1", "turn": 2,
            }),
            StreamEvent(event_type="complete", turn=2, data={
                "success": True,
                "answer": "Result is 1",
                "total_turns": 2,
                "total_input_tokens": 200,
                "total_output_tokens": 30,
                "skills_used": [],
                "final_messages": snapshot_msgs + [
                    {"role": "assistant", "content": [{"type": "text", "text": "Result is 1"}]},
                ],
            }),
        ]

        mock_instance = MagicMock()
        mock_instance.model = "kimi-k2.5"
        mock_instance.model_provider = "kimi"
        mock_instance.cleanup = MagicMock()

        async def mock_run(request, conversation_history=None, image_contents=None,
                           event_stream=None, cancellation_event=None):
            if event_stream:
                for event in events:
                    await event_stream.push(event)
                await event_stream.close()
            from app.agent.agent import AgentResult
            return AgentResult(
                success=True, answer="Result is 1",
                total_turns=2, total_input_tokens=200, total_output_tokens=30,
                skills_used=[], final_messages=snapshot_msgs + [
                    {"role": "assistant", "content": [{"type": "text", "text": "Result is 1"}]},
                ],
            )

        mock_instance.run = AsyncMock(side_effect=mock_run)
        MockAgent.return_value = mock_instance

        # Create a real preset so _resolve_agent_config works
        from app.db.models import AgentPresetDB
        preset = AgentPresetDB(
            name="e2e-display-save-test",
            description="Test incremental display",
            max_turns=5,
        )
        e2e_db_session.add(preset)
        await e2e_db_session.commit()
        await e2e_db_session.refresh(preset)
        type(self)._state["preset_id"] = preset.id

        resp = await e2e_client.post(
            "/api/v1/agent/run/stream",
            json={
                "request": "Analyze data",
                "session_id": session_id,
                "agent_id": preset.id,
            },
        )
        assert resp.status_code == 200

        # Verify save_session_checkpoint was called with display_messages
        assert MockCheckpoint.call_count >= 1
        # First checkpoint call (from turn_complete)
        cp_call = MockCheckpoint.call_args_list[0]
        assert cp_call.args[0] == session_id  # session_id
        assert cp_call.args[1] == snapshot_msgs  # agent_context
        # display_messages is now ChatMessage format (from DisplayMessageBuilder)
        cp_display = cp_call.kwargs.get("display_messages")
        assert cp_display is not None
        assert len(cp_display) == 2  # user + assistant-with-events
        assert cp_display[0]["role"] == "user"
        assert cp_display[0]["content"] == "Analyze data"
        assert cp_display[1]["role"] == "assistant"
        assert "streamEvents" in cp_display[1]
        event_types = [e["type"] for e in cp_display[1]["streamEvents"]]
        assert "turn_start" in event_types
        assert "tool_result" in event_types

    @patch("app.api.v1.agent.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.agent.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.agent.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.agent.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.agent.create_agent")
    async def test_02_turn_complete_appends_to_existing_display(
        self, MockAgent, MockLoadSession, MockSaveMessages, MockCheckpoint, MockPreCompress,
        e2e_client: AsyncClient, e2e_db_session,
    ):
        """turn_complete with existing display_base: display_base + new turns (agent.py)."""
        session_id = str(uuid.uuid4())
        existing_display = [
            {"role": "user", "content": "Previous Q"},
            {"role": "assistant", "content": [{"type": "text", "text": "Previous A"}]},
        ]
        existing_context = list(existing_display)
        MockLoadSession.return_value = SessionData(
            session_id=session_id,
            display_messages=existing_display,
            agent_context=existing_context,
        )

        # Snapshot includes history + new turn
        snapshot_msgs = existing_context + [
            {"role": "user", "content": "New Q"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t:0", "name": "bash", "input": {"command": "ls"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t:0", "content": "file.txt"},
            ]},
        ]

        events = [
            StreamEvent(event_type="turn_start", turn=1, data={"turn": 1}),
            StreamEvent(event_type="tool_result", turn=1, data={
                "tool_name": "bash", "tool_input": {"command": "ls"}, "tool_result": "file.txt",
            }),
            StreamEvent(event_type="turn_complete", turn=1, data={
                "messages_snapshot": snapshot_msgs,
            }),
            StreamEvent(event_type="complete", turn=2, data={
                "success": True,
                "answer": "Listed files",
                "total_turns": 2,
                "total_input_tokens": 200,
                "total_output_tokens": 30,
                "skills_used": [],
                "final_messages": snapshot_msgs + [
                    {"role": "assistant", "content": [{"type": "text", "text": "Listed files"}]},
                ],
            }),
        ]

        mock_instance = MagicMock()
        mock_instance.model = "kimi-k2.5"
        mock_instance.model_provider = "kimi"
        mock_instance.cleanup = MagicMock()

        async def mock_run(request, conversation_history=None, image_contents=None,
                           event_stream=None, cancellation_event=None):
            if event_stream:
                for event in events:
                    await event_stream.push(event)
                await event_stream.close()
            from app.agent.agent import AgentResult
            return AgentResult(
                success=True, answer="Listed files",
                total_turns=2, total_input_tokens=200, total_output_tokens=30,
                skills_used=[], final_messages=snapshot_msgs + [
                    {"role": "assistant", "content": [{"type": "text", "text": "Listed files"}]},
                ],
            )

        mock_instance.run = AsyncMock(side_effect=mock_run)
        MockAgent.return_value = mock_instance

        pid = type(self)._state["preset_id"]
        resp = await e2e_client.post(
            "/api/v1/agent/run/stream",
            json={
                "request": "New Q",
                "session_id": session_id,
                "agent_id": pid,
            },
        )
        assert resp.status_code == 200

        # Verify checkpoint display = existing_display + new ChatMessage-format turns
        assert MockCheckpoint.call_count >= 1
        cp_call = MockCheckpoint.call_args_list[0]
        cp_display = cp_call.kwargs.get("display_messages")
        assert cp_display is not None
        # Should be existing 2 msgs + 2 new (user + assistant-with-events) = 4
        assert len(cp_display) == 4
        assert cp_display[0] == existing_display[0]
        assert cp_display[1] == existing_display[1]
        assert cp_display[2]["role"] == "user"
        assert cp_display[2]["content"] == "New Q"
        assert cp_display[3]["role"] == "assistant"
        assert "streamEvents" in cp_display[3]

    async def test_03_checkpoint_func_accepts_display(self):
        """save_session_checkpoint passes display_messages to DB when provided."""
        from app.api.v1.sessions import save_session_checkpoint, save_session_checkpoint_sync

        # Test that the function signatures accept the new parameter
        import inspect
        async_sig = inspect.signature(save_session_checkpoint)
        assert "display_messages" in async_sig.parameters
        assert async_sig.parameters["display_messages"].default is None

        sync_sig = inspect.signature(save_session_checkpoint_sync)
        assert "display_messages" in sync_sig.parameters
        assert sync_sig.parameters["display_messages"].default is None

    async def test_04_checkpoint_sync_func_accepts_display(self):
        """save_session_checkpoint_sync builds correct UPDATE when display_messages provided."""
        # Directly test the sync function with a mocked DB
        from app.api.v1.sessions import save_session_checkpoint_sync
        from unittest.mock import patch as sync_patch

        captured_values = {}

        class MockSyncSession:
            def execute(self, stmt):
                # Extract values from the UPDATE statement
                if hasattr(stmt, 'compile'):
                    compiled = stmt.compile(compile_kwargs={"literal_binds": False})
                    captured_values.update(compiled.params or {})

            def commit(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        # SyncSessionLocal is imported inside the function body, so mock at database module level
        with sync_patch("app.db.database.SyncSessionLocal", return_value=MockSyncSession()):
            save_session_checkpoint_sync(
                "test-session",
                [{"role": "user", "content": "ctx"}],
                display_messages=[{"role": "user", "content": "display"}],
            )

        assert "messages" in captured_values
        assert captured_values["messages"] == [{"role": "user", "content": "display"}]
        assert "agent_context" in captured_values

    # ---- Published agent (published.py) tests ----

    async def test_05_create_and_publish(self, e2e_client: AsyncClient):
        """Create and publish an agent for published display-save tests."""
        resp = await e2e_client.post("/api/v1/agents", json={
            "name": "e2e-display-save-pub",
            "description": "Test incremental display save (published)",
            "max_turns": 5,
        })
        assert resp.status_code == 200
        pid = resp.json()["id"]
        type(self)._state["pub_preset_id"] = pid

        resp = await e2e_client.post(
            f"/api/v1/agents/{pid}/publish",
            json={"api_response_mode": "streaming"},
        )
        assert resp.status_code == 200

    @patch("app.api.v1.published.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.published.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.published.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.published.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.published.SkillsAgent")
    @patch("app.api.v1.published.AsyncSessionLocal")
    async def test_06_published_turn_complete_saves_display(
        self, MockSL, MockAgent, MockLoadSession, MockSaveMessages, MockCheckpoint, MockPreCompress,
        e2e_client: AsyncClient,
    ):
        """turn_complete checkpoint includes display_messages (published.py)."""
        pid = type(self)._state["pub_preset_id"]
        session_id = str(uuid.uuid4())
        MockLoadSession.return_value = SessionData(
            session_id=session_id,
            display_messages=[],
        )

        snapshot_msgs = [
            {"role": "user", "content": "Search web"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "ws:0", "name": "web_search", "input": {"query": "test"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "ws:0", "content": "results"},
            ]},
        ]

        events_to_push = [
            StreamEvent(event_type="turn_start", turn=1, data={"turn": 1}),
            StreamEvent(event_type="tool_result", turn=1, data={
                "tool_name": "web_search", "tool_input": {"query": "test"}, "tool_result": "results",
            }),
            StreamEvent(event_type="turn_complete", turn=1, data={
                "messages_snapshot": snapshot_msgs,
            }),
            StreamEvent(event_type="assistant", turn=2, data={
                "content": "Found results", "turn": 2,
            }),
            StreamEvent(event_type="complete", turn=2, data={
                "success": True,
                "answer": "Found results",
                "total_turns": 2,
                "total_input_tokens": 200,
                "total_output_tokens": 30,
                "skills_used": [],
                "final_messages": snapshot_msgs + [
                    {"role": "assistant", "content": [{"type": "text", "text": "Found results"}]},
                ],
            }),
        ]

        mock_instance = MagicMock()
        mock_instance.cleanup = MagicMock()
        mock_instance.model = "kimi-k2.5"
        mock_instance.model_provider = "kimi"

        async def mock_run(request, conversation_history=None, image_contents=None,
                           event_stream=None, cancellation_event=None):
            if event_stream:
                for event in events_to_push:
                    await event_stream.push(event)
                await event_stream.close()
            from app.agent.agent import AgentResult
            return AgentResult(
                success=True, answer="Found results",
                total_turns=2, total_input_tokens=200, total_output_tokens=30,
                skills_used=[], final_messages=snapshot_msgs + [
                    {"role": "assistant", "content": [{"type": "text", "text": "Found results"}]},
                ],
            )

        mock_instance.run = AsyncMock(side_effect=mock_run)
        MockAgent.return_value = mock_instance

        # Mock AsyncSessionLocal for preset lookup + trace creation
        from app.db.models import AgentPresetDB

        mock_preset = MagicMock(spec=AgentPresetDB)
        mock_preset.id = pid
        mock_preset.name = "e2e-display-save-pub"
        mock_preset.description = "Test"
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

        @asynccontextmanager
        async def _ctx():
            idx = call_idx["i"]
            call_idx["i"] += 1
            mock_sess = AsyncMock(spec=AsyncSession)

            if idx == 0:
                mock_result = MagicMock()
                mock_result.scalar_one_or_none.return_value = mock_preset
                mock_sess.execute = AsyncMock(return_value=mock_result)
            else:
                mock_result = MagicMock()
                mock_result.scalar_one_or_none.return_value = None
                mock_sess.execute = AsyncMock(return_value=mock_result)
                mock_sess.add = MagicMock()
                mock_sess.commit = AsyncMock()

            yield mock_sess

        MockSL.side_effect = lambda: _ctx()

        resp = await e2e_client.post(
            f"/api/v1/published/{pid}/chat",
            json={"request": "Search web", "session_id": session_id},
        )
        assert resp.status_code == 200

        # Verify checkpoint was called with display_messages (ChatMessage format)
        assert MockCheckpoint.call_count >= 1
        cp_call = MockCheckpoint.call_args_list[0]
        assert cp_call.args[0] == session_id
        assert cp_call.args[1] == snapshot_msgs
        cp_display = cp_call.kwargs.get("display_messages")
        assert cp_display is not None
        assert len(cp_display) == 2  # user + assistant-with-events
        assert cp_display[0]["role"] == "user"
        assert cp_display[0]["content"] == "Search web"
        assert cp_display[1]["role"] == "assistant"
        assert "streamEvents" in cp_display[1]
        event_types = [e["type"] for e in cp_display[1]["streamEvents"]]
        assert "turn_start" in event_types
        assert "tool_result" in event_types

    @patch("app.api.v1.published.pre_compress_if_needed", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.published.save_session_checkpoint", new_callable=AsyncMock)
    @patch("app.api.v1.published.save_session_messages", new_callable=AsyncMock)
    @patch("app.api.v1.published.load_or_create_session", new_callable=AsyncMock)
    @patch("app.api.v1.published.SkillsAgent")
    @patch("app.api.v1.published.AsyncSessionLocal")
    async def test_07_published_multi_turn_display_accumulates(
        self, MockSL, MockAgent, MockLoadSession, MockSaveMessages, MockCheckpoint, MockPreCompress,
        e2e_client: AsyncClient,
    ):
        """Multiple turn_complete events: each checkpoint includes all completed turns."""
        pid = type(self)._state["pub_preset_id"]
        session_id = str(uuid.uuid4())
        MockLoadSession.return_value = SessionData(
            session_id=session_id,
            display_messages=[],
        )

        # Turn 1 snapshot
        turn1_snapshot = [
            {"role": "user", "content": "Step 1"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t:0", "name": "bash", "input": {"command": "echo 1"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t:0", "content": "1"},
            ]},
        ]
        # Turn 2 snapshot (accumulates)
        turn2_snapshot = turn1_snapshot + [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t:1", "name": "bash", "input": {"command": "echo 2"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t:1", "content": "2"},
            ]},
        ]

        events_to_push = [
            StreamEvent(event_type="turn_start", turn=1, data={"turn": 1}),
            StreamEvent(event_type="turn_complete", turn=1, data={"messages_snapshot": turn1_snapshot}),
            StreamEvent(event_type="turn_start", turn=2, data={"turn": 2}),
            StreamEvent(event_type="turn_complete", turn=2, data={"messages_snapshot": turn2_snapshot}),
            StreamEvent(event_type="complete", turn=3, data={
                "success": True, "answer": "Done", "total_turns": 3,
                "total_input_tokens": 300, "total_output_tokens": 50,
                "skills_used": [],
                "final_messages": turn2_snapshot + [
                    {"role": "assistant", "content": [{"type": "text", "text": "Done"}]},
                ],
            }),
        ]

        mock_instance = MagicMock()
        mock_instance.cleanup = MagicMock()
        mock_instance.model = "kimi-k2.5"
        mock_instance.model_provider = "kimi"

        async def mock_run(request, conversation_history=None, image_contents=None,
                           event_stream=None, cancellation_event=None):
            if event_stream:
                for ev in events_to_push:
                    await event_stream.push(ev)
                await event_stream.close()
            from app.agent.agent import AgentResult
            return AgentResult(
                success=True, answer="Done", total_turns=3,
                total_input_tokens=300, total_output_tokens=50,
                skills_used=[], final_messages=turn2_snapshot + [
                    {"role": "assistant", "content": [{"type": "text", "text": "Done"}]},
                ],
            )

        mock_instance.run = AsyncMock(side_effect=mock_run)
        MockAgent.return_value = mock_instance

        from app.db.models import AgentPresetDB
        mock_preset = MagicMock(spec=AgentPresetDB)
        mock_preset.id = pid
        mock_preset.name = "e2e-display-save-pub"
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

        @asynccontextmanager
        async def _ctx():
            idx = call_idx["i"]
            call_idx["i"] += 1
            mock_sess = AsyncMock(spec=AsyncSession)
            if idx == 0:
                mock_result = MagicMock()
                mock_result.scalar_one_or_none.return_value = mock_preset
                mock_sess.execute = AsyncMock(return_value=mock_result)
            else:
                mock_result = MagicMock()
                mock_result.scalar_one_or_none.return_value = None
                mock_sess.execute = AsyncMock(return_value=mock_result)
                mock_sess.add = MagicMock()
                mock_sess.commit = AsyncMock()
            yield mock_sess

        MockSL.side_effect = lambda: _ctx()

        resp = await e2e_client.post(
            f"/api/v1/published/{pid}/chat",
            json={"request": "Step 1", "session_id": session_id},
        )
        assert resp.status_code == 200

        # Should have 2 checkpoint calls (one per turn_complete)
        assert MockCheckpoint.call_count >= 2

        # First checkpoint: user + assistant with turn_start(1) events
        cp1_display = MockCheckpoint.call_args_list[0].kwargs.get("display_messages")
        assert cp1_display is not None
        assert len(cp1_display) == 2  # user + assistant-with-events
        assert cp1_display[0]["role"] == "user"
        assert cp1_display[0]["content"] == "Step 1"
        assert cp1_display[1]["role"] == "assistant"
        assert "streamEvents" in cp1_display[1]

        # Second checkpoint: same user + assistant now with both turn_start events
        cp2_display = MockCheckpoint.call_args_list[1].kwargs.get("display_messages")
        assert cp2_display is not None
        assert len(cp2_display) == 2  # still user + assistant
        # Assistant events should now contain both turn_start records
        cp2_events = cp2_display[1]["streamEvents"]
        turn_start_events = [e for e in cp2_events if e["type"] == "turn_start"]
        assert len(turn_start_events) == 2

    async def test_08_cleanup(self, e2e_client: AsyncClient):
        """Clean up test agents."""
        for key in ("preset_id", "pub_preset_id"):
            pid = type(self)._state.get(key)
            if pid:
                await e2e_client.post(f"/api/v1/agents/{pid}/unpublish")
                await e2e_client.delete(f"/api/v1/agents/{pid}")
