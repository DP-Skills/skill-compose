"""Agent API - Run the skills agent"""
import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import EventStream, write_steering_message, poll_steering_messages, cleanup_steering_dir
from app.api.v1.display_builder import DisplayMessageBuilder
from app.api.v1.sessions import load_or_create_session, save_session_messages, save_session_checkpoint, save_session_checkpoint_sync, pre_compress_if_needed, CHAT_SENTINEL_AGENT_ID
from app.db.database import get_db, AsyncSessionLocal, SyncSessionLocal
from app.db.models import AgentTraceDB, AgentPresetDB
from app.services.agent_runner import AgentConfig, config_from_preset, create_agent, build_completed_trace, build_initial_trace

logger = logging.getLogger("skills_api")

router = APIRouter(prefix="/agent", tags=["Agent"])

# Module-level registry of active streaming runs (trace_id → EventStream)
_active_streams: Dict[str, EventStream] = {}


def _update_trace_sync(trace_id: str, values: dict):
    """Update trace record via sync session — immune to async cancellation."""
    with SyncSessionLocal() as sync_db:
        sync_db.execute(
            update(AgentTraceDB)
            .where(AgentTraceDB.id == trace_id)
            .values(**values)
        )
        sync_db.commit()


async def _finalize_trace(trace_id: str, values: dict):
    """Update trace with final status. Uses sync DB to be resilient to task cancellation.

    SSE generator finally-blocks run in a context where ASGI may cancel async operations
    at any time. Using sync DB (psycopg2) avoids orphaned asyncpg connections that
    asyncio.shield() would create.
    """
    try:
        _update_trace_sync(trace_id, values)
    except Exception as e:
        logger.error(f"Trace update failed for {trace_id}: {e}")


class UploadedFile(BaseModel):
    """Info about an uploaded file"""
    file_id: str
    filename: str
    path: str  # Full path for agent to access
    content_type: str


class AgentRequest(BaseModel):
    """Agent run request.

    If agent_id is provided, the preset's config is used and individual
    config fields (skills, allowed_tools, etc.) are ignored.
    """
    request: str = Field(..., description="The task/request for the agent")
    agent_id: Optional[str] = Field(None, description="Agent preset ID. When set, uses preset config and ignores individual config fields.")
    model_provider: Optional[str] = Field(None, description="LLM provider: anthropic, openrouter, openai, google")
    model_name: Optional[str] = Field(None, description="Model name/ID for the provider")
    skills: Optional[List[str]] = Field(None, description="List of skill names to activate (None = all available)")
    allowed_tools: Optional[List[str]] = Field(None, description="List of tool names to enable (None = all available)")
    max_turns: int = Field(60, description="Maximum turns before stopping", ge=1, le=60000)
    session_id: str = Field(..., description="Session ID for server-side session management. "
                                             "History is loaded from DB automatically.")
    uploaded_files: Optional[List[UploadedFile]] = Field(
        None, description="List of uploaded files available to the agent"
    )
    equipped_mcp_servers: Optional[List[str]] = Field(
        None, description="List of MCP server names to enable (None = all available)"
    )
    system_prompt: Optional[str] = Field(
        None, description="Custom system prompt to append to the base prompt"
    )
    executor_name: Optional[str] = Field(
        None, description="Executor name for code execution (custom mode only, ignored when agent_id is set)"
    )


class StepInfo(BaseModel):
    """Info about a single execution step"""
    role: str
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None


class AgentResponse(BaseModel):
    """Agent run response"""
    success: bool
    answer: str
    total_turns: int
    steps: List[StepInfo]
    error: Optional[str] = None
    trace_id: Optional[str] = None  # ID of saved trace for later reference
    output_files: Optional[List[dict]] = None


class SteerRequest(BaseModel):
    """Request to inject a steering message into a running agent."""
    message: str = Field(..., description="Steering message to inject", min_length=1)


@router.post("/run/stream/{trace_id}/steer")
async def steer_agent(trace_id: str, body: SteerRequest, db: AsyncSession = Depends(get_db)):
    """Inject a steering message into a running agent stream.

    Uses a hybrid approach for multi-worker support:
    1. Fast path: if the stream is in this worker's memory, inject directly
    2. Cross-worker path: verify trace is running in DB, write to filesystem queue
       (a polling task in the streaming worker picks it up)
    """
    # Fast path: same worker
    event_stream = _active_streams.get(trace_id)
    if event_stream:
        if event_stream.closed:
            raise HTTPException(status_code=409, detail="Agent has already completed")
        await event_stream.inject(body.message)
        return {"status": "injected", "trace_id": trace_id}

    # Cross-worker path: check DB for running trace, then write to filesystem
    from sqlalchemy import select
    result = await db.execute(
        select(AgentTraceDB).where(AgentTraceDB.id == trace_id)
    )
    trace = result.scalar_one_or_none()
    if not trace:
        raise HTTPException(status_code=404, detail="No active run found for this trace_id")
    if trace.status != "running":
        raise HTTPException(status_code=409, detail="Agent has already completed")

    write_steering_message(trace_id, body.message)
    return {"status": "injected", "trace_id": trace_id}


async def _resolve_agent_config(request: AgentRequest, db: AsyncSession) -> AgentConfig:
    """Resolve effective agent config. If agent_id is set, load preset and use its config."""
    from sqlalchemy import select

    if not request.agent_id:
        # Custom mode: use executor_name directly
        executor_name = request.executor_name or None
        if executor_name:
            from app.api.v1.executors import _get_executor_status
            status = await _get_executor_status(executor_name)
            if status != "online":
                raise HTTPException(
                    status_code=503,
                    detail=f"Executor '{executor_name}' is offline. Cannot run agent."
                )

        return AgentConfig(
            skills=request.skills,
            allowed_tools=request.allowed_tools,
            max_turns=request.max_turns,
            equipped_mcp_servers=request.equipped_mcp_servers,
            system_prompt=request.system_prompt,
            model_provider=request.model_provider,
            model_name=request.model_name,
            agent_id=None,
            executor_name=executor_name,
            is_meta_agent=False,
        )

    result = await db.execute(
        select(AgentPresetDB).where(AgentPresetDB.id == request.agent_id)
    )
    preset = result.scalar_one_or_none()
    if not preset:
        raise HTTPException(status_code=404, detail=f"Agent preset '{request.agent_id}' not found")

    # Use executor_name directly from preset
    executor_name = preset.executor_name or None
    if executor_name:
        from app.api.v1.executors import _get_executor_status
        status = await _get_executor_status(executor_name)
        if status != "online":
            raise HTTPException(
                status_code=503,
                detail=f"Executor '{executor_name}' is offline. Cannot run agent."
            )

    cfg = config_from_preset(preset)
    # Override model from request if preset doesn't specify
    cfg.model_provider = preset.model_provider or request.model_provider
    cfg.model_name = preset.model_name or request.model_name
    return cfg


def _build_request_with_files(
    request_text: str,
    uploaded_files: Optional[List[UploadedFile]],
    model_provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Tuple[str, Optional[List[dict]]]:
    """Build the actual request text and image content blocks from uploaded files.

    Separates image files (for vision-capable models) from non-image files.
    Images are encoded as base64 Anthropic-format image blocks.
    Non-image files are appended as text paths in the request.

    Returns:
        (actual_request, image_contents) where image_contents is None if no images
        or the model doesn't support vision.
    """
    if not uploaded_files:
        return request_text, None

    from app.llm.models import supports_vision

    # Check if model supports vision
    provider = model_provider or "kimi"
    model = model_name or "kimi-k2.5"
    has_vision = supports_vision(provider, model)

    # Separate image files from non-image files
    image_files = []
    non_image_files = []
    for f in uploaded_files:
        if has_vision and f.content_type and f.content_type.startswith("image/"):
            image_files.append(f)
        else:
            non_image_files.append(f)

    # Build text request with non-image file paths
    actual_request = request_text
    if non_image_files:
        files_info = "\n".join([
            f"- {f.filename}: {Path(f.path).resolve()} (type: {f.content_type})"
            for f in non_image_files
        ])
        actual_request = f"""{request_text}

[Uploaded Files]
The user has uploaded the following files that you can access:
{files_info}

IMPORTANT: Use the absolute file paths shown above when reading or processing files."""

    # Build image content blocks
    image_contents = None
    if image_files:
        image_contents = []
        for f in image_files:
            try:
                file_path = Path(f.path).resolve()
                with open(file_path, "rb") as fp:
                    data = base64.standard_b64encode(fp.read()).decode("utf-8")
                image_contents.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f.content_type,
                        "data": data,
                    }
                })
            except Exception as e:
                logger.warning(f"Failed to read image file {f.filename}: {e}")
                # Fall back to text path for this file
                actual_request += f"\n- {f.filename}: {Path(f.path).resolve()} (type: {f.content_type})"

        if not image_contents:
            image_contents = None

    return actual_request, image_contents


def _validate_api_key(config: AgentConfig):
    """Validate that the API key for the selected provider is configured.

    Raises HTTPException(400) if the key is missing or empty.
    """
    from app.config import get_settings, read_env_value
    from app.llm.provider import PROVIDER_API_KEY_MAP

    settings_obj = get_settings()
    provider = config.model_provider or settings_obj.default_model_provider
    env_var = PROVIDER_API_KEY_MAP.get(provider, f"{provider.upper()}_API_KEY")
    key_value = read_env_value(env_var)
    if not key_value or not key_value.strip():
        raise HTTPException(
            status_code=400,
            detail=f"API key for provider '{provider}' is not configured. "
                   f"Please set {env_var} in Settings > Environment.",
        )


@router.post("/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest, db: AsyncSession = Depends(get_db)):
    """
    Run the skills agent on a task (non-streaming).

    The agent will:
    1. List available skills
    2. Read relevant skill documentation
    3. Write and execute code
    4. Return the final result

    Execution traces are automatically saved to the database.
    """
    # Resolve config from agent_id or individual fields
    config = await _resolve_agent_config(request, db)
    _validate_api_key(config)

    start_time = time.time()

    # Load session history from DB (dual-store)
    agent_id_for_session = config.agent_id or CHAT_SENTINEL_AGENT_ID
    session_data = await load_or_create_session(request.session_id, agent_id_for_session)
    session_id = session_data.session_id
    history = session_data.agent_context  # Use agent_context for the agent

    # Create agent with session_id as workspace_id for deterministic mapping
    agent = create_agent(config, workspace_id=session_id)

    try:
        # Pre-compress if context exceeds threshold
        from app.config import settings as app_settings_run
        effective_provider = config.model_provider or app_settings_run.default_model_provider
        effective_model = config.model_name or app_settings_run.default_model_name
        if history:
            history = await pre_compress_if_needed(history, effective_provider, effective_model)

        # Build the actual request with file info and image blocks
        actual_request, image_contents = _build_request_with_files(
            request.request,
            request.uploaded_files,
            model_provider=config.model_provider,
            model_name=config.model_name,
        )

        result = await agent.run(actual_request, conversation_history=history, image_contents=image_contents)

        duration_ms = int((time.time() - start_time) * 1000)

        # Save trace to database
        trace = build_completed_trace(
            request_text=request.request,
            result=result,
            agent=agent,
            duration_ms=duration_ms,
            executor_name=config.executor_name,
            session_id=session_id,
        )
        db.add(trace)
        await db.commit()

        # Build display messages using DisplayMessageBuilder
        final_msgs = getattr(result, "final_messages", None)
        display_builder = DisplayMessageBuilder.from_agent_result(result, request.request, request.uploaded_files)
        new_display = display_builder.get_messages()

        # Save session (dual-store)
        await save_session_messages(
            session_id,
            result.answer,
            request.request,
            final_messages=final_msgs,
            display_append_messages=new_display,
        )

        return AgentResponse(
            success=result.success,
            answer=result.answer,
            total_turns=result.total_turns,
            steps=[
                StepInfo(
                    role=step.role,
                    content=step.content[:1000] if step.content else "",
                    tool_name=step.tool_name,
                    tool_input=step.tool_input,
                )
                for step in result.steps
            ],
            error=result.error,
            trace_id=trace.id,
            output_files=result.output_files or None,
        )
    finally:
        # Always cleanup workspace
        agent.cleanup()


@router.post("/run/stream")
async def run_agent_stream(request: AgentRequest, db: AsyncSession = Depends(get_db)):
    """
    Run the skills agent with streaming output (SSE).

    Returns Server-Sent Events with each turn's progress.
    Event types:
    - turn_start: New turn started
    - text_delta: Incremental text chunk from LLM
    - assistant: Assistant text response (legacy, may be skipped when text_delta is used)
    - tool_call: Tool being called
    - tool_result: Tool execution result
    - complete: Agent finished successfully
    - error: Agent encountered an error
    """
    # Resolve config from agent_id or individual fields
    config = await _resolve_agent_config(request, db)
    _validate_api_key(config)

    # Build the actual request with file info and image blocks
    actual_request, image_contents = _build_request_with_files(
        request.request,
        request.uploaded_files,
        model_provider=config.model_provider,
        model_name=config.model_name,
    )

    # Load session history from DB (dual-store)
    agent_id_for_session = config.agent_id or CHAT_SENTINEL_AGENT_ID
    session_data = await load_or_create_session(request.session_id, agent_id_for_session)
    session_id = session_data.session_id
    history = session_data.agent_context  # Use agent_context for the agent
    display_base = session_data.display_messages or []  # Snapshot of display history for incremental saves

    # Pre-compress if context exceeds threshold
    from app.config import settings as app_settings_pre
    pre_provider = config.model_provider or app_settings_pre.default_model_provider
    pre_model = config.model_name or app_settings_pre.default_model_name
    if history:
        history = await pre_compress_if_needed(history, pre_provider, pre_model)

    async def event_generator():
        start_time = time.time()

        # Create trace record at the start (with running status)
        trace_id = None
        from app.config import settings as app_settings
        effective_provider = config.model_provider or app_settings.default_model_provider
        effective_model = config.model_name or app_settings.default_model_name
        async with AsyncSessionLocal() as trace_db:
            trace = build_initial_trace(
                request_text=request.request,
                model_provider=effective_provider,
                model_name=effective_model,
                executor_name=config.executor_name,
                session_id=session_id,
            )
            trace_db.add(trace)
            await trace_db.commit()
            trace_id = trace.id

        # Send run_started event with trace_id and session_id immediately
        yield f"data: {json.dumps({'event_type': 'run_started', 'turn': 0, 'trace_id': trace_id, 'session_id': session_id})}\n\n"

        # Create agent, event stream, and cancellation event
        config.verbose = False
        agent = create_agent(config, workspace_id=session_id)

        event_stream = EventStream()
        cancel_event = asyncio.Event()

        # Register stream for steering (same-worker fast path + cross-worker polling)
        if trace_id:
            _active_streams[trace_id] = event_stream
        steering_task = asyncio.create_task(
            poll_steering_messages(trace_id, event_stream)
        ) if trace_id else None

        # Run agent in a background task
        agent_task = asyncio.create_task(
            agent.run(
                actual_request,
                conversation_history=history,
                image_contents=image_contents,
                event_stream=event_stream,
                cancellation_event=cancel_event,
            )
        )

        last_complete_event = None
        last_messages_snapshot = None  # Incremental checkpoint for resilient save
        collected_steps = []  # Collect steps during streaming
        current_text_buffer = ""  # Accumulate text_delta chunks
        was_cancelled = False
        display_builder = DisplayMessageBuilder()
        display_builder.add_user_message(request.request, request.uploaded_files)

        try:
            async for event in event_stream:
                # Heartbeat — SSE comment to keep connection alive through proxies
                if event.event_type == "heartbeat":
                    yield ": heartbeat\n\n"
                    continue

                # Feed to display builder (before turn_complete check)
                display_builder.add_event(event.event_type, event.turn, event.data)

                # Intercept turn_complete for incremental checkpoint (not forwarded to client)
                if event.event_type == "turn_complete":
                    snapshot = event.data.get("messages_snapshot")
                    last_messages_snapshot = snapshot
                    # Checkpoint: agent_context (snapshot) + display (builder snapshot)
                    if session_id and snapshot:
                        try:
                            display_to_save = display_base + display_builder.get_snapshot()
                            await save_session_checkpoint(session_id, snapshot, display_messages=display_to_save)
                        except Exception:
                            pass  # fire-and-forget
                    continue

                event_data = {
                    "event_type": event.event_type,
                    "turn": event.turn,
                    **event.data
                }
                yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"

                # Collect steps from events
                event_type = event.event_type
                if event_type == "text_delta":
                    current_text_buffer += event.data.get("text", "")
                elif event_type == "assistant" and event.data.get("content"):
                    collected_steps.append({
                        "role": "assistant",
                        "content": event.data.get("content", "")[:2000],
                        "tool_name": None,
                        "tool_input": None,
                    })
                elif event_type in ("tool_call", "tool_result", "turn_start", "complete"):
                    # Flush buffered text as assistant step
                    if current_text_buffer:
                        collected_steps.append({
                            "role": "assistant",
                            "content": current_text_buffer[:2000],
                            "tool_name": None,
                            "tool_input": None,
                        })
                        current_text_buffer = ""
                    if event_type == "tool_result":
                        collected_steps.append({
                            "role": "tool",
                            "content": event.data.get("tool_result", "")[:5000],
                            "tool_name": event.data.get("tool_name"),
                            "tool_input": event.data.get("tool_input"),
                        })
                    elif event_type == "complete":
                        last_complete_event = event.data

            # Wait for agent task to complete (it should already be done after stream closes)
            await agent_task

        except (asyncio.CancelledError, GeneratorExit):
            # Request was cancelled (e.g., user clicked Stop)
            was_cancelled = True
            cancel_event.set()       # Signal agent to stop
            agent_task.cancel()      # Cancel agent task
            try:
                await agent_task
            except asyncio.CancelledError:
                pass

        finally:
            # Stop steering poll and clean up
            if steering_task:
                steering_task.cancel()
                try:
                    await steering_task
                except asyncio.CancelledError:
                    pass
            if trace_id:
                cleanup_steering_dir(trace_id)
                _active_streams.pop(trace_id, None)

            # Always cleanup workspace
            agent.cleanup()

            # Always update trace record with final results
            duration_ms = int((time.time() - start_time) * 1000)

            # Determine final status
            if was_cancelled:
                final_status = "cancelled"
                is_success = False
                error_msg = "Request was cancelled by user"
            elif last_complete_event:
                is_success = last_complete_event.get("success", False)
                final_status = "completed" if is_success else "failed"
                error_msg = last_complete_event.get("error")
            else:
                final_status = "failed"
                is_success = False
                error_msg = "Agent did not complete"

            await _finalize_trace(trace_id, {
                "status": final_status,
                "success": is_success,
                "answer": last_complete_event.get("answer", "") if last_complete_event else "",
                "error": error_msg,
                "total_turns": last_complete_event.get("total_turns", 0) if last_complete_event else 0,
                "total_input_tokens": last_complete_event.get("total_input_tokens", 0) if last_complete_event else 0,
                "total_output_tokens": last_complete_event.get("total_output_tokens", 0) if last_complete_event else 0,
                "skills_used": last_complete_event.get("skills_used", []) if last_complete_event else [],
                "steps": collected_steps,
                "duration_ms": duration_ms,
            })

            # Save full conversation messages to session (dual-store)
            if session_id:
                if not was_cancelled and last_complete_event:
                    # Normal completion — definitive save
                    final_answer = last_complete_event.get("answer", "")
                    final_msgs = last_complete_event.get("final_messages")

                    # Whole-replace display with builder output (builder is independent
                    # of agent snapshots, so compression doesn't affect it).
                    full_display = display_base + display_builder.get_messages()

                    await save_session_messages(
                        session_id,
                        final_answer,
                        request.request,
                        final_messages=final_msgs,
                        display_replace_messages=full_display,
                    )
                else:
                    # Cancelled or interrupted — save what we have (agent_context + display)
                    # Use sync DB to avoid orphaned async connections in cancelled context
                    try:
                        display_to_save = display_base + display_builder.get_snapshot()
                        if not display_to_save or display_to_save == display_base:
                            display_to_save = display_base + [{"role": "user", "content": request.request}]

                        snapshot = last_messages_snapshot
                        save_session_checkpoint_sync(
                            session_id,
                            snapshot if snapshot else (history or []) + [{"role": "user", "content": actual_request}],
                            display_messages=display_to_save,
                        )
                    except Exception:
                        pass

        if not was_cancelled:
            yield f"data: {json.dumps({'event_type': 'trace_saved', 'turn': 0, 'trace_id': trace_id})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
