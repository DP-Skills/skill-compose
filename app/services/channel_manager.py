"""
Channel Manager service.

Manages channel adapter lifecycle and routes inbound messages to agents.

Supports multiple Feishu apps via per-binding credentials stored in the
``config`` JSONB column. Adapters are keyed by ``feishu:{app_id}`` so that
multiple bindings sharing the same app_id reuse one WebSocket connection.
"""

import asyncio
import base64
import hashlib
import logging
import mimetypes
import re
import time
from datetime import datetime
from pathlib import Path
from collections import namedtuple
from typing import Optional
from urllib.parse import parse_qs, urlparse

from app.channels.base import ChannelAdapter, InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)

_BindingInfo = namedtuple("_BindingInfo", ["channel_type", "config"])

# Image extensions that support vision (base64 encoding for LLM)
_VISION_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def adapter_key_for_binding(binding) -> Optional[str]:
    """Derive the adapter dict key from a channel binding.

    For Feishu bindings the key is ``feishu:{app_id}`` (from config).
    For other channel types the key is just the channel_type string.
    """
    if binding.channel_type == "feishu":
        config = binding.config or {}
        app_id = config.get("app_id")
        if not app_id:
            return None
        return f"feishu:{app_id}"
    return binding.channel_type


class ChannelManager:
    """Singleton that manages all channel adapters and routes messages."""

    _instance = None
    _adapters: dict[str, ChannelAdapter]

    def __new__(cls):
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._adapters = {}
            inst._is_leader = False
            cls._instance = inst
        return cls._instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Start adapters from DB bindings + env-based Telegram."""
        self._is_leader = True
        from app.config import settings

        # --- Feishu: start one adapter per unique app_id from DB bindings ---
        await self._start_feishu_adapters_from_db()

        # --- Telegram: still env-based (single bot token) ---
        if settings.telegram_bot_token:
            try:
                from app.channels.telegram import TelegramAdapter
                adapter = TelegramAdapter(settings.telegram_bot_token)
                adapter.set_message_handler(self._handle_inbound)
                await adapter.connect()
                self._adapters["telegram"] = adapter
                logger.info("Telegram adapter started")
            except ImportError:
                logger.info("Telegram adapter not available (python-telegram-bot not installed)")
            except Exception as e:
                logger.warning(f"Failed to start Telegram adapter: {e}")

    async def _start_feishu_adapters_from_db(self):
        """Query all enabled Feishu bindings and start one adapter per unique app_id."""
        from sqlalchemy import select
        from app.db.database import AsyncSessionLocal
        from app.db.models import ChannelBindingDB

        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(ChannelBindingDB).where(
                        ChannelBindingDB.channel_type == "feishu",
                        ChannelBindingDB.enabled == True,
                    )
                )
                bindings = result.scalars().all()

                # Group by app_id, start one adapter per unique app_id
                seen_app_ids: dict[str, str] = {}  # app_id -> app_secret
                for b in bindings:
                    config = b.config or {}
                    app_id = config.get("app_id")
                    app_secret = config.get("app_secret")
                    if app_id and app_secret and app_id not in seen_app_ids:
                        seen_app_ids[app_id] = app_secret

            for app_id, app_secret in seen_app_ids.items():
                await self._start_feishu_adapter(app_id, app_secret)

        except Exception as e:
            logger.warning(f"Failed to load Feishu bindings from DB: {e}")

    async def _start_feishu_adapter(self, app_id: str, app_secret: str) -> bool:
        """Start a single Feishu adapter for the given credentials."""
        key = f"feishu:{app_id}"
        if key in self._adapters:
            logger.debug(f"Feishu adapter {key} already running")
            return True

        try:
            from app.channels.feishu import FeishuAdapter
            adapter = FeishuAdapter(app_id, app_secret)
            adapter.set_message_handler(self._handle_inbound)
            await adapter.connect()
            self._adapters[key] = adapter
            logger.info(f"Feishu adapter started: {key}")
            return True
        except ImportError:
            logger.info("Feishu adapter not available (lark-oapi not installed)")
            return False
        except Exception as e:
            logger.warning(f"Failed to start Feishu adapter {key}: {e}")
            return False

    async def _stop_adapter(self, adapter_key: str) -> bool:
        """Stop and remove an adapter by its key."""
        adapter = self._adapters.pop(adapter_key, None)
        if not adapter:
            return False
        try:
            await adapter.disconnect()
            logger.info(f"Channel adapter '{adapter_key}' stopped")
            return True
        except Exception as e:
            logger.warning(f"Error stopping adapter '{adapter_key}': {e}")
            return False

    async def stop(self):
        """Stop all adapters."""
        for name, adapter in self._adapters.items():
            try:
                await adapter.disconnect()
                logger.info(f"Channel adapter '{name}' stopped")
            except Exception as e:
                logger.warning(f"Error stopping adapter '{name}': {e}")
        self._adapters.clear()

    def get_adapter_status(self) -> dict[str, bool]:
        """Get connection status of all adapters."""
        return {name: adapter.is_connected() for name, adapter in self._adapters.items()}

    async def restart_adapter(self, adapter_type: str) -> bool:
        """Restart a specific adapter."""
        adapter = self._adapters.get(adapter_type)
        if not adapter:
            return False

        try:
            await adapter.disconnect()
            await adapter.connect()
            return True
        except Exception as e:
            logger.error(f"Failed to restart adapter '{adapter_type}': {e}")
            return False

    # ------------------------------------------------------------------
    # Hot-reload hooks (called from API after CRUD operations)
    # ------------------------------------------------------------------

    async def on_binding_created(self, binding_id: str):
        """Start adapter if this binding introduces a new app_id."""
        if not self._is_leader:
            return
        from sqlalchemy import select
        from app.db.database import AsyncSessionLocal
        from app.db.models import ChannelBindingDB

        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(ChannelBindingDB).where(ChannelBindingDB.id == binding_id)
                )
                binding = result.scalar_one_or_none()
                if not binding or binding.channel_type != "feishu":
                    return
                config = binding.config or {}

            app_id = config.get("app_id")
            app_secret = config.get("app_secret")
            if app_id and app_secret:
                await self._start_feishu_adapter(app_id, app_secret)
        except Exception as e:
            logger.warning(f"on_binding_created error: {e}", exc_info=True)

    async def on_binding_updated(self, binding_id: str, old_config: Optional[dict]):
        """If app_id changed, stop old adapter (if unused) and start new one."""
        if not self._is_leader:
            return
        from sqlalchemy import select
        from app.db.database import AsyncSessionLocal
        from app.db.models import ChannelBindingDB

        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(ChannelBindingDB).where(ChannelBindingDB.id == binding_id)
                )
                binding = result.scalar_one_or_none()
                if not binding or binding.channel_type != "feishu":
                    return
                new_config = binding.config or {}

            old_app_id = (old_config or {}).get("app_id")
            new_app_id = new_config.get("app_id")
            new_app_secret = new_config.get("app_secret")

            # Start new adapter if needed
            if new_app_id and new_app_secret:
                await self._start_feishu_adapter(new_app_id, new_app_secret)

            # Stop old adapter if app_id changed and no other bindings use it
            if old_app_id and old_app_id != new_app_id:
                await self._maybe_stop_feishu_adapter(old_app_id)

        except Exception as e:
            logger.warning(f"on_binding_updated error: {e}", exc_info=True)

    async def on_binding_deleted(self, binding_id: str, config: Optional[dict]):
        """Stop adapter if no remaining bindings use this app_id."""
        if not self._is_leader:
            return
        if not config:
            return
        app_id = config.get("app_id")
        if not app_id:
            return

        try:
            await self._maybe_stop_feishu_adapter(app_id)
        except Exception as e:
            logger.warning(f"on_binding_deleted error for {binding_id}: {e}", exc_info=True)

    async def _maybe_stop_feishu_adapter(self, app_id: str):
        """Stop the Feishu adapter for app_id if no enabled bindings still reference it."""
        from sqlalchemy import select, func
        from app.db.database import AsyncSessionLocal
        from app.db.models import ChannelBindingDB

        key = f"feishu:{app_id}"
        if key not in self._adapters:
            return

        async with AsyncSessionLocal() as session:
            # Count remaining enabled bindings that use this app_id
            count_result = await session.execute(
                select(func.count()).select_from(ChannelBindingDB).where(
                    ChannelBindingDB.channel_type == "feishu",
                    ChannelBindingDB.enabled == True,
                    ChannelBindingDB.config["app_id"].astext == app_id,
                )
            )
            remaining = count_result.scalar() or 0

        if remaining == 0:
            await self._stop_adapter(key)

    # ------------------------------------------------------------------
    # Adapter lookup helper
    # ------------------------------------------------------------------

    def _get_adapter_for_binding(self, binding) -> Optional[ChannelAdapter]:
        """Look up the correct adapter for a binding."""
        key = adapter_key_for_binding(binding)
        if not key:
            return None
        return self._adapters.get(key)

    # ------------------------------------------------------------------
    # Inbound message handling
    # ------------------------------------------------------------------

    async def _handle_inbound(self, msg: InboundMessage):
        """Handle an inbound message from a channel adapter.

        Uses two-level lookup:
        1. Exact match: channel_type + external_id (specific group binding)
        2. Fallback: channel_type + external_id='*' + config app_id match (global binding)
        """
        from sqlalchemy import select
        from app.db.database import AsyncSessionLocal
        from app.db.models import (
            ChannelBindingDB, ChannelMessageDB, AgentPresetDB,
            PublishedSessionDB, generate_uuid,
        )

        try:
            async with AsyncSessionLocal() as session:
                # Level 1: Exact match on external_id
                result = await session.execute(
                    select(ChannelBindingDB).where(
                        ChannelBindingDB.channel_type == msg.channel_type,
                        ChannelBindingDB.external_id == msg.external_id,
                        ChannelBindingDB.enabled == True,
                    )
                )
                binding = result.scalar_one_or_none()

                # Level 2: Fallback to global binding (external_id='*') matched by app_id
                if not binding:
                    app_id = (msg.metadata or {}).get("app_id")
                    if app_id:
                        result = await session.execute(
                            select(ChannelBindingDB).where(
                                ChannelBindingDB.channel_type == msg.channel_type,
                                ChannelBindingDB.external_id == "*",
                                ChannelBindingDB.enabled == True,
                                ChannelBindingDB.config["app_id"].astext == app_id,
                            )
                        )
                        binding = result.scalar_one_or_none()

                if not binding:
                    logger.debug(f"No binding for {msg.channel_type}:{msg.external_id}")
                    return

                # Check trigger pattern (skip for media messages — images/files
                # should always be processed regardless of text trigger)
                if binding.trigger_pattern and not msg.media:
                    try:
                        if not re.search(binding.trigger_pattern, msg.content):
                            return
                    except re.error:
                        logger.warning(f"Invalid trigger pattern '{binding.trigger_pattern}' for binding {binding.id}, processing anyway")

                # Record inbound message
                inbound_record = ChannelMessageDB(
                    id=generate_uuid(),
                    channel_binding_id=binding.id,
                    direction="inbound",
                    external_message_id=msg.external_message_id,
                    sender_id=msg.sender_id,
                    sender_name=msg.sender_name,
                    content=msg.content,
                    message_type=msg.message_type,
                    msg_metadata=msg.metadata,
                )
                session.add(inbound_record)

                # Generate deterministic session_id
                session_key = f"{msg.channel_type}:{msg.external_id}"
                session_id = hashlib.sha256(session_key.encode()).hexdigest()[:36]

                # Load agent preset
                preset = await session.execute(
                    select(AgentPresetDB).where(AgentPresetDB.id == binding.agent_id)
                )
                preset = preset.scalar_one_or_none()
                if not preset:
                    logger.error(f"Agent preset {binding.agent_id} not found for binding {binding.id}")
                    return

                # Load or create session
                pub_session = await session.execute(
                    select(PublishedSessionDB).where(PublishedSessionDB.id == session_id)
                )
                pub_session = pub_session.scalar_one_or_none()

                conversation_history = None
                if pub_session and pub_session.agent_context:
                    conversation_history = pub_session.agent_context
                elif not pub_session:
                    pub_session = PublishedSessionDB(
                        id=session_id,
                        agent_id=binding.agent_id,
                        messages=[],
                    )
                    session.add(pub_session)

                # Save binding config for adapter lookup after session closes
                binding_config = binding.config
                binding_channel_type = binding.channel_type

                await session.commit()

            # Build image_contents and augment prompt for media files
            image_contents = None
            actual_prompt = msg.content
            if msg.media:
                image_contents, actual_prompt = self._build_media_context(
                    msg.media, msg.content
                )

            # Run agent (outside DB session)
            answer, output_files = await self._run_agent(
                preset, actual_prompt, conversation_history, session_id,
                image_contents=image_contents,
            )

            # Extract file paths from agent output_files for sending back
            media_paths = self._extract_output_file_paths(output_files)

            # Record outbound and update session
            async with AsyncSessionLocal() as session:
                outbound_record = ChannelMessageDB(
                    id=generate_uuid(),
                    channel_binding_id=binding.id,
                    direction="outbound",
                    content=answer or "",
                    message_type="text",
                )
                session.add(outbound_record)
                await session.commit()

            # Send response via adapter — use binding config to find correct adapter
            if answer or media_paths:
                adapter = self._get_adapter_for_binding(
                    _BindingInfo(binding_channel_type, binding_config)
                )
                if adapter and adapter.is_connected():
                    await adapter.send_message(OutboundMessage(
                        external_id=msg.external_id,
                        content=answer or "",
                        media=media_paths,
                    ))

        except Exception as e:
            logger.error(f"Error handling inbound message: {e}", exc_info=True)

    async def _run_agent(
        self,
        preset,
        prompt: str,
        conversation_history: Optional[list] = None,
        session_id: Optional[str] = None,
        image_contents: Optional[list[dict]] = None,
    ) -> tuple[Optional[str], list[dict]]:
        """Run a SkillsAgent with the given preset config.

        Returns:
            (answer, output_files) tuple.
        """
        from app.services.agent_runner import config_from_preset, create_agent, build_completed_trace

        agent = None
        try:
            # Create agent via shared service
            config = config_from_preset(preset)
            config.verbose = False
            agent = create_agent(config, workspace_id=session_id)

            # Pre-compress if context exceeds threshold
            if conversation_history:
                from app.api.v1.sessions import pre_compress_if_needed
                conversation_history = await pre_compress_if_needed(
                    conversation_history,
                    agent.model_provider,
                    agent.model,
                )

            start_time = time.time()
            result = await agent.run(
                prompt,
                conversation_history=conversation_history,
                image_contents=image_contents,
            )
            duration_ms = int((time.time() - start_time) * 1000)

            # Save trace (the core bug fix — channel messages now get traced)
            from app.db.database import AsyncSessionLocal
            async with AsyncSessionLocal() as trace_db:
                trace = build_completed_trace(
                    request_text=prompt,
                    result=result,
                    agent=agent,
                    duration_ms=duration_ms,
                    executor_name=config.executor_name,
                    session_id=session_id,
                )
                trace_db.add(trace)
                await trace_db.commit()

            # Update session via save_session_messages (dual-store: agent_context + display)
            if session_id and result.final_messages:
                from app.api.v1.sessions import save_session_messages
                await save_session_messages(
                    session_id,
                    result.answer,
                    prompt,
                    final_messages=result.final_messages,
                )

            return result.answer, result.output_files

        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            return f"Error: {str(e)}", []
        finally:
            if agent:
                agent.cleanup()

    @staticmethod
    def _build_media_context(
        media_paths: list[str], prompt: str
    ) -> tuple[Optional[list[dict]], str]:
        """Build image_contents blocks and augment prompt for non-image files.

        Returns:
            (image_contents, augmented_prompt)
        """
        image_contents: list[dict] = []
        non_image_info: list[str] = []

        for media_path in media_paths:
            p = Path(media_path)
            if not p.exists():
                continue

            ext = p.suffix.lower()
            if ext in _VISION_EXTENSIONS:
                try:
                    media_type = mimetypes.guess_type(media_path)[0] or "image/png"
                    with open(media_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    image_contents.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    })
                except Exception as e:
                    logger.warning(f"Failed to encode image {media_path}: {e}")
                    non_image_info.append(
                        f"- {p.name}: {p.resolve()} (type: image)"
                    )
            else:
                content_type = mimetypes.guess_type(media_path)[0] or "application/octet-stream"
                non_image_info.append(
                    f"- {p.name}: {p.resolve()} (type: {content_type})"
                )

        actual_prompt = prompt
        if non_image_info:
            files_info = "\n".join(non_image_info)
            actual_prompt = f"""{prompt}

[Uploaded Files]
The user has uploaded the following files that you can access:
{files_info}

IMPORTANT: Use the absolute file paths shown above when reading or processing files."""

        return (image_contents or None), actual_prompt

    @staticmethod
    def _extract_output_file_paths(output_files: list[dict]) -> list[str]:
        """Extract absolute file paths from agent output_files.

        Each output_file dict has a ``download_url`` like:
        ``/api/v1/files/output/download?path=<base64url_encoded_path>``
        """
        paths: list[str] = []
        for f in output_files:
            download_url = f.get("download_url", "")
            if not download_url:
                continue
            try:
                parsed = urlparse(download_url)
                qs = parse_qs(parsed.query)
                encoded_path = qs.get("path", [""])[0]
                if encoded_path:
                    decoded = base64.urlsafe_b64decode(
                        encoded_path.encode("ascii")
                    ).decode("utf-8")
                    if Path(decoded).exists():
                        paths.append(decoded)
            except Exception as e:
                logger.warning(f"Failed to extract file path from {download_url}: {e}")
        return paths

    async def send_to_channel(self, binding_id: str, content: str):
        """Send a message to a channel binding (used by scheduler).

        Global bindings (external_id='*') are skipped because there is no
        specific chat_id to send to.
        """
        from sqlalchemy import select
        from app.db.database import AsyncSessionLocal
        from app.db.models import ChannelBindingDB, ChannelMessageDB, generate_uuid

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ChannelBindingDB).where(ChannelBindingDB.id == binding_id)
            )
            binding = result.scalar_one_or_none()
            if not binding:
                logger.warning(f"Channel binding {binding_id} not found")
                return

            # Global bindings have no specific target chat_id
            if binding.external_id == "*":
                logger.info(f"Skipping scheduled message for global binding {binding_id} (no target chat_id)")
                return

            adapter = self._get_adapter_for_binding(binding)
            if not adapter or not adapter.is_connected():
                logger.warning(f"No connected adapter for binding {binding_id}")
                return

            # Send message
            await adapter.send_message(OutboundMessage(
                external_id=binding.external_id,
                content=content,
            ))

            # Record outbound
            msg_record = ChannelMessageDB(
                id=generate_uuid(),
                channel_binding_id=binding.id,
                direction="outbound",
                content=content,
                message_type="text",
            )
            session.add(msg_record)
            await session.commit()
