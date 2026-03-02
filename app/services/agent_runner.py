"""Unified agent execution service layer.

Provides shared agent construction, trace creation, and config resolution
used by all 4 execution paths:
  - /agent/run  (non-streaming)
  - /agent/run/stream  (streaming SSE)
  - scheduler  (periodic background tasks)
  - channel_manager  (Feishu / Telegram inbound messages)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List, TYPE_CHECKING

from app.db.models import AgentTraceDB, AgentPresetDB

if TYPE_CHECKING:
    from app.agent import SkillsAgent, AgentResult


@dataclass
class AgentConfig:
    """Normalized agent configuration.

    Replaces the dict returned by _resolve_agent_config and inline preset
    field-mapping scattered across scheduler / channel_manager.
    """
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    max_turns: int = 60
    skills: Optional[List[str]] = None
    allowed_tools: Optional[List[str]] = None
    equipped_mcp_servers: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    executor_name: Optional[str] = None
    agent_id: Optional[str] = None
    is_meta_agent: bool = False
    verbose: bool = True


def config_from_preset(preset: AgentPresetDB) -> AgentConfig:
    """Map AgentPresetDB fields to AgentConfig."""
    return AgentConfig(
        model_provider=preset.model_provider,
        model_name=preset.model_name,
        max_turns=preset.max_turns or 60,
        skills=preset.skill_ids,
        allowed_tools=preset.builtin_tools,
        equipped_mcp_servers=preset.mcp_servers,
        system_prompt=preset.system_prompt,
        executor_name=preset.executor_name or None,
        agent_id=preset.id,
        is_meta_agent=preset.is_system,
    )


def create_agent(config: AgentConfig, workspace_id: Optional[str] = None) -> SkillsAgent:
    """Create a SkillsAgent from an AgentConfig.

    Single constructor replacing 4 duplicate SkillsAgent(...) calls.
    """
    from app.agent import SkillsAgent

    return SkillsAgent(
        model=config.model_name,
        model_provider=config.model_provider,
        max_turns=config.max_turns,
        verbose=config.verbose,
        allowed_skills=config.skills,
        allowed_tools=config.allowed_tools,
        equipped_mcp_servers=config.equipped_mcp_servers,
        custom_system_prompt=config.system_prompt,
        executor_name=config.executor_name,
        workspace_id=workspace_id,
        is_meta_agent=config.is_meta_agent,
    )


def build_completed_trace(
    request_text: str,
    result: AgentResult,
    agent: SkillsAgent,
    duration_ms: int,
    executor_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> AgentTraceDB:
    """Build an unsaved AgentTraceDB from a completed AgentResult.

    Reads agent.model_provider / agent.model which have defaults resolved.
    Caller adds the returned object to their own DB session.
    """
    return AgentTraceDB(
        request=request_text,
        skills_used=list(result.skills_used) if result.skills_used else [],
        model_provider=agent.model_provider,
        model=agent.model,
        status="completed" if result.success else "failed",
        success=result.success,
        answer=result.answer,
        error=result.error,
        total_turns=result.total_turns,
        total_input_tokens=result.total_input_tokens,
        total_output_tokens=result.total_output_tokens,
        steps=[asdict(s) for s in result.steps],
        llm_calls=[asdict(c) for c in result.llm_calls],
        duration_ms=duration_ms,
        executor_name=executor_name,
        session_id=session_id,
    )


def build_initial_trace(
    request_text: str,
    model_provider: str,
    model_name: str,
    executor_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> AgentTraceDB:
    """Build an unsaved "running" trace for the streaming path.

    Caller adds to their own DB session and commits to get the trace ID.
    """
    return AgentTraceDB(
        request=request_text,
        skills_used=[],
        model_provider=model_provider,
        model=model_name,
        status="running",
        success=False,
        answer="",
        error=None,
        total_turns=0,
        total_input_tokens=0,
        total_output_tokens=0,
        steps=[],
        llm_calls=[],
        duration_ms=0,
        executor_name=executor_name,
        session_id=session_id,
    )
