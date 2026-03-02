"""
Task Scheduler service for periodic agent execution.

Polls the database for due tasks and executes them via SkillsAgent.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def _calculate_next_run(
    schedule_type: str,
    schedule_value: str,
    from_time: Optional[datetime] = None,
) -> Optional[datetime]:
    """Calculate the next run time based on schedule type.

    Args:
        schedule_type: 'cron', 'interval', or 'once'
        schedule_value: cron expression, interval in seconds, or ISO datetime
        from_time: base time for calculation (defaults to now)
    """
    now = from_time or datetime.utcnow()

    if schedule_type == "cron":
        from croniter import croniter
        cron = croniter(schedule_value, now)
        return cron.get_next(datetime)

    elif schedule_type == "interval":
        seconds = int(schedule_value)
        return now + timedelta(seconds=seconds)

    elif schedule_type == "once":
        # ISO datetime string
        run_at = datetime.fromisoformat(schedule_value.replace("Z", "+00:00"))
        if run_at > now:
            return run_at
        return None

    return None


def validate_schedule(schedule_type: str, schedule_value: str) -> str | None:
    """Validate schedule configuration. Returns error message or None."""
    if schedule_type not in ("cron", "interval", "once"):
        return f"Invalid schedule_type: {schedule_type}. Must be cron, interval, or once."

    if schedule_type == "cron":
        try:
            from croniter import croniter
            croniter(schedule_value)
        except (ValueError, KeyError) as e:
            return f"Invalid cron expression: {e}"

    elif schedule_type == "interval":
        try:
            val = int(schedule_value)
            if val < 10:
                return "Interval must be at least 10 seconds"
        except ValueError:
            return "Interval must be an integer (seconds)"

    elif schedule_type == "once":
        try:
            datetime.fromisoformat(schedule_value.replace("Z", "+00:00"))
        except ValueError:
            return "Invalid ISO datetime for once schedule"

    return None


MAX_TASK_WORKERS = 5


class TaskScheduler:
    """Singleton scheduler that polls for due tasks and executes them."""

    _instance = None
    _task: Optional[asyncio.Task] = None
    _running: bool = False
    _executor: Optional[ThreadPoolExecutor] = None

    def __new__(cls):
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._executor = ThreadPoolExecutor(max_workers=MAX_TASK_WORKERS, thread_name_prefix="sched-task")
            cls._instance = inst
        return cls._instance

    async def start(self):
        """Start the scheduler polling loop."""
        if self._running:
            return
        self._running = True
        if not self._executor:
            self._executor = ThreadPoolExecutor(max_workers=MAX_TASK_WORKERS, thread_name_prefix="sched-task")
        self._task = asyncio.create_task(self._loop())
        logger.info("TaskScheduler started")

    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
        logger.info("TaskScheduler stopped")

    async def _loop(self):
        """Main polling loop."""
        from app.config import settings
        interval = settings.scheduler_poll_interval

        while self._running:
            try:
                await self._poll_and_execute()
            except Exception as e:
                logger.error(f"Scheduler poll error: {e}", exc_info=True)
            await asyncio.sleep(interval)

    async def _poll_and_execute(self):
        """Find and execute due tasks."""
        from sqlalchemy import select, update
        from app.db.database import AsyncSessionLocal
        from app.db.models import ScheduledTaskDB, TaskRunLogDB, generate_uuid

        now = datetime.utcnow()

        async with AsyncSessionLocal() as session:
            # Find due tasks
            result = await session.execute(
                select(ScheduledTaskDB).where(
                    ScheduledTaskDB.status == "active",
                    ScheduledTaskDB.next_run <= now,
                )
            )
            due_tasks = result.scalars().all()

            for task in due_tasks:
                try:
                    # Calculate next run
                    next_run = _calculate_next_run(task.schedule_type, task.schedule_value, now)

                    # Update task next_run and last_run
                    task.next_run = next_run
                    task.last_run = now
                    task.run_count += 1

                    # Check if max_runs reached
                    if task.max_runs and task.run_count >= task.max_runs:
                        task.status = "completed"
                    elif task.schedule_type == "once":
                        task.status = "completed"

                    # Create run log
                    run_log = TaskRunLogDB(
                        id=generate_uuid(),
                        task_id=task.id,
                        started_at=now,
                        status="running",
                    )
                    # Capture attributes before commit (avoids MissingGreenlet on expired state)
                    task_id = task.id
                    task_name = task.name

                    session.add(run_log)
                    await session.commit()

                    run_log_id = run_log.id

                    # Execute in thread pool
                    self._executor.submit(self._execute_task, task_id, run_log_id)

                    logger.info(f"Scheduled task '{task_name}' (id={task_id}) dispatched, run_log={run_log_id}")

                except Exception as e:
                    logger.error(f"Error dispatching task {task.id}: {e}", exc_info=True)

    def _execute_task(self, task_id: str, run_log_id: str):
        """Execute a scheduled task in a background thread."""
        from app.db.database import SyncSessionLocal
        from app.db.models import (
            ScheduledTaskDB, TaskRunLogDB, AgentPresetDB,
            PublishedSessionDB, generate_uuid,
        )
        from app.services.agent_runner import config_from_preset, create_agent, build_completed_trace

        start_time = time.time()
        session = SyncSessionLocal()
        agent = None

        try:
            # Load task and agent preset
            task = session.get(ScheduledTaskDB, task_id)
            if not task:
                logger.error(f"Scheduled task {task_id} not found")
                return

            preset = session.get(AgentPresetDB, task.agent_id)
            if not preset:
                logger.error(f"Agent preset {task.agent_id} not found for task {task_id}")
                self._update_run_log(session, run_log_id, "failed", error="Agent preset not found")
                return

            # Build conversation history for session mode
            conversation_history = None
            if task.context_mode == "session" and task.session_id:
                pub_session = session.get(PublishedSessionDB, task.session_id)
                if pub_session and pub_session.agent_context:
                    conversation_history = pub_session.agent_context

            # Create agent via shared service
            config = config_from_preset(preset)
            agent = create_agent(config, workspace_id=task.session_id)

            # Run agent (single event loop for all async operations)
            loop = asyncio.new_event_loop()
            try:
                # Pre-compress if context exceeds threshold
                if conversation_history:
                    from app.api.v1.sessions import pre_compress_if_needed
                    conversation_history = loop.run_until_complete(
                        pre_compress_if_needed(
                            conversation_history,
                            agent.model_provider,
                            agent.model,
                        )
                    )

                result = loop.run_until_complete(
                    agent.run(task.prompt, conversation_history=conversation_history)
                )

                duration_ms = int((time.time() - start_time) * 1000)

                # Save trace via shared service
                trace = build_completed_trace(
                    request_text=task.prompt,
                    result=result,
                    agent=agent,
                    duration_ms=duration_ms,
                    executor_name=config.executor_name,
                    session_id=task.session_id,
                )
                session.add(trace)

                # Update session via save_session_messages (dual-store: agent_context + display)
                if task.context_mode == "session" and task.session_id and result.final_messages:
                    from app.api.v1.sessions import save_session_messages
                    loop.run_until_complete(
                        save_session_messages(
                            task.session_id,
                            result.answer,
                            task.prompt,
                            final_messages=result.final_messages,
                        )
                    )
            finally:
                loop.close()

            # Update run log
            self._update_run_log(
                session, run_log_id, "completed",
                result_summary=result.answer[:500] if result.answer else None,
                trace_id=trace.id,
                duration_ms=duration_ms,
            )

            # Send to channel if binding exists
            if task.channel_binding_id and result.answer:
                try:
                    from app.services.channel_manager import ChannelManager
                    manager = ChannelManager()
                    send_loop = asyncio.new_event_loop()
                    try:
                        send_loop.run_until_complete(
                            manager.send_to_channel(task.channel_binding_id, result.answer)
                        )
                    finally:
                        send_loop.close()
                except Exception as e:
                    logger.warning(f"Failed to send result to channel: {e}")

            logger.info(f"Scheduled task '{task_id}' completed in {duration_ms}ms")

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Scheduled task {task_id} failed: {e}", exc_info=True)
            self._update_run_log(
                session, run_log_id, "failed",
                error=str(e),
                duration_ms=duration_ms,
            )
        finally:
            if agent:
                agent.cleanup()
            session.close()

    def _update_run_log(
        self, session, run_log_id: str, status: str,
        result_summary: str = None, error: str = None,
        trace_id: str = None, duration_ms: int = None,
    ):
        """Update a run log record."""
        from app.db.models import TaskRunLogDB

        run_log = session.get(TaskRunLogDB, run_log_id)
        if run_log:
            run_log.status = status
            run_log.completed_at = datetime.utcnow()
            if result_summary:
                run_log.result_summary = result_summary
            if error:
                run_log.error = error
            if trace_id:
                run_log.trace_id = trace_id
            if duration_ms is not None:
                run_log.duration_ms = duration_ms
            session.commit()

    async def execute_task_async(self, task_id: str):
        """Execute a task immediately (for run-now endpoint)."""
        from app.db.database import AsyncSessionLocal
        from app.db.models import ScheduledTaskDB, TaskRunLogDB, generate_uuid

        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(ScheduledTaskDB).where(ScheduledTaskDB.id == task_id)
            )
            task = result.scalar_one_or_none()
            if not task:
                raise ValueError(f"Task {task_id} not found")

            now = datetime.utcnow()
            run_log = TaskRunLogDB(
                id=generate_uuid(),
                task_id=task.id,
                started_at=now,
                status="running",
            )
            session.add(run_log)
            task.last_run = now
            task.run_count += 1
            await session.commit()

            run_log_id = run_log.id

        # Execute in thread pool
        self._executor.submit(self._execute_task, task_id, run_log_id)
        return run_log_id
