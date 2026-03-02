"""
Feishu (Lark) channel adapter.

Uses the official lark-oapi SDK with WebSocket long-connection mode,
so no public URL / webhook endpoint is needed. The lark WS client runs
in a daemon thread; inbound messages are bridged to the async event loop
via ``asyncio.run_coroutine_threadsafe``.

Credentials (app_id / app_secret) are passed via constructor parameters,
typically sourced from the channel binding's ``config`` JSONB column.

Feishu app permissions needed:
    im:message            - Send messages
    im:message.receive_v1 - Receive messages (event subscription)
    im:resource           - Download message resources (images/files)
    im:image              - Upload images
    im:file               - Upload files
"""

import asyncio
import json
import logging
import mimetypes
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import lark_oapi as lark
import lark_oapi.ws.client as _lark_ws_mod


class _LoopProxy:
    """Proxy that delegates all attribute access to the thread-local event loop.

    The lark SDK captures a module-level ``loop = asyncio.get_event_loop()``
    at import time, then calls ``loop.run_until_complete()`` on it.  Under
    uvicorn this is the running uvloop, causing RuntimeError.  This proxy
    always resolves to the *current thread's* event loop, enabling multiple
    adapters to each run their own loop in separate threads.

    Uses ``get_event_loop_policy().get_event_loop()`` to avoid the
    DeprecationWarning emitted by ``asyncio.get_event_loop()`` in
    Python 3.10+ when no running loop exists on the current thread.
    """
    def __getattr__(self, name):
        loop = asyncio.get_event_loop_policy().get_event_loop()
        return getattr(loop, name)
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    GetMessageResourceRequest,
    CreateImageRequest,
    CreateImageRequestBody,
    CreateFileRequest,
    CreateFileRequestBody,
)
from lark_oapi.event.dispatcher_handler import P2ImMessageReceiveV1

from app.channels.base import ChannelAdapter, InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)

MEDIA_DIR = Path("/tmp/feishu_media")
MAX_DEDUP_SIZE = 1000

# Extensions that should be sent as images (vs generic files) in Feishu
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


class FeishuAdapter(ChannelAdapter):
    """Feishu / Lark messaging adapter.

    Connects via WebSocket (``lark.ws.Client``), receives ``im.message.receive_v1``
    events and sends replies through ``im.v1.message.create``.
    """

    channel_type = "feishu"

    def __init__(self, app_id: str, app_secret: str):
        self._app_id = app_id
        self._app_secret = app_secret
        self._client: Optional[lark.Client] = None
        self._ws_client = None
        self._connected = False
        self._bot_open_id: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._seen_messages: OrderedDict = OrderedDict()

        MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def app_id(self) -> str:
        """Return the Feishu app_id for adapter identification."""
        return self._app_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Start the Feishu WebSocket connection."""
        # Capture the current event loop so the WS thread can schedule coroutines
        self._loop = asyncio.get_running_loop()

        # Build the API client (used for sending messages)
        self._client = (
            lark.Client.builder()
            .app_id(self._app_id)
            .app_secret(self._app_secret)
            .build()
        )

        # Fetch bot's own open_id so we can filter self-sent messages
        self._bot_open_id = await self._get_bot_open_id()
        if self._bot_open_id:
            logger.info(f"Feishu bot open_id: {self._bot_open_id}")
        else:
            logger.warning("Could not retrieve Feishu bot open_id; self-message filtering disabled")

        # Build event dispatcher (encryption/verification tokens left empty for WS mode)
        handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._on_message)
            .build()
        )

        # Replace the lark SDK's module-level event loop with our _LoopProxy
        # (defined at module level) so each adapter thread uses its own loop.
        import threading

        if not isinstance(getattr(_lark_ws_mod, 'loop', None), _LoopProxy):
            _lark_ws_mod.loop = _LoopProxy()

        def _run_ws():
            # Create a clean stdlib event loop for this thread.
            fresh_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(fresh_loop)

            ws_client = lark.ws.Client(
                self._app_id,
                self._app_secret,
                event_handler=handler,
                log_level=lark.LogLevel.WARNING,
            )
            self._ws_client = ws_client

            while self._connected:
                try:
                    ws_client.start()
                except Exception as e:
                    logger.error(f"Feishu WS client error: {e}", exc_info=True)
                if self._connected:
                    import time
                    time.sleep(5)  # reconnect backoff

        self._connected = True
        self._ws_thread = threading.Thread(
            target=_run_ws,
            daemon=True,
            name=f"feishu-ws-{self._app_id[:8]}",
        )
        self._ws_thread.start()
        logger.info(f"Feishu adapter connected via WebSocket (app_id={self._app_id[:8]}...)")

    async def disconnect(self) -> None:
        """Stop the Feishu connection."""
        self._connected = False
        # The lark WS client does not expose a clean shutdown API;
        # dropping the reference lets the daemon thread exit with the process.
        self._ws_client = None
        self._client = None
        logger.info("Feishu adapter disconnected")

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send_message(self, message: OutboundMessage) -> None:
        """Send a message to a Feishu chat.

        If ``message.media`` contains file paths, each file is uploaded and
        sent as a separate image or file message before the text reply.
        """
        if not self._client:
            raise RuntimeError("Feishu adapter not connected")

        loop = asyncio.get_running_loop()

        # Upload and send media files first
        for file_path in message.media:
            try:
                p = Path(file_path)
                if not p.exists():
                    logger.warning(f"Media file not found, skipping: {file_path}")
                    continue

                ext = p.suffix.lower()
                if ext in _IMAGE_EXTENSIONS:
                    image_key = await loop.run_in_executor(
                        None, self._upload_image, file_path
                    )
                    if image_key:
                        self._send_image_message(message.external_id, image_key)
                else:
                    file_key = await loop.run_in_executor(
                        None, self._upload_file, file_path
                    )
                    if file_key:
                        self._send_file_message(
                            message.external_id, file_key, p.name
                        )
            except Exception as e:
                logger.error(f"Failed to send media {file_path}: {e}", exc_info=True)

        # Send text content
        if message.content:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(message.external_id)
                    .msg_type("text")
                    .content(json.dumps({"text": message.content}))
                    .build()
                )
                .build()
            )

            response = self._client.im.v1.message.create(request)
            if not response.success():
                logger.error(
                    f"Failed to send Feishu message: code={response.code} msg={response.msg}"
                )

    # ------------------------------------------------------------------
    # Inbound (called from lark WS daemon thread)
    # ------------------------------------------------------------------

    def _on_message(self, data: P2ImMessageReceiveV1) -> None:
        """Handle an inbound message event from the Feishu WebSocket.

        This callback is invoked on the lark WS daemon thread, so we bridge
        to the main asyncio loop via ``run_coroutine_threadsafe``.
        """
        try:
            event = data.event
            msg = event.message
            sender = event.sender

            chat_type = getattr(msg, "chat_type", "unknown")
            logger.info(
                f"Feishu inbound event: chat_type={chat_type} chat_id={msg.chat_id} "
                f"msg_id={msg.message_id} msg_type={msg.message_type}"
            )

            # Message dedup
            msg_id = msg.message_id
            if msg_id in self._seen_messages:
                return
            self._seen_messages[msg_id] = True
            if len(self._seen_messages) > MAX_DEDUP_SIZE:
                self._seen_messages.popitem(last=False)

            # Derive sender open_id
            sender_id = ""
            if sender and sender.sender_id:
                sender_id = sender.sender_id.open_id or ""

            # Skip messages sent by the bot itself
            if self._bot_open_id and sender_id == self._bot_open_id:
                return

            # In group chats, only respond when the bot is @mentioned
            mentions = getattr(msg, "mentions", None) or []
            bot_mention_key = None
            if chat_type == "group" and self._bot_open_id:
                for m in mentions:
                    if m.id and m.id.open_id == self._bot_open_id:
                        bot_mention_key = m.key  # e.g. "@_user_1"
                        break
                if not bot_mention_key:
                    return

            msg_type = msg.message_type
            content = ""
            media_paths: list[str] = []

            if msg_type == "text":
                try:
                    content = json.loads(msg.content).get("text", "")
                except (json.JSONDecodeError, TypeError):
                    content = ""

            elif msg_type == "post":
                # Rich text — extract plain text from JSON structure
                content = self._extract_post_text(msg.content)

            elif msg_type in ("image", "file", "audio", "media"):
                # Download media and set content to a descriptive placeholder
                logger.info(f"Feishu inbound {msg_type} message: {msg.message_id}")
                paths, desc = self._download_and_save_media(
                    msg_type, msg.content, msg.message_id
                )
                media_paths = [str(p) for p in paths]
                content = desc or f"[{msg_type}]"
                if not paths:
                    logger.warning(f"Media download returned no files for {msg_type} message {msg.message_id}")

            else:
                # Unsupported message type — ignore
                return

            # Strip the bot @mention placeholder from content (e.g. "@_user_1")
            if bot_mention_key and content:
                content = content.replace(bot_mention_key, "").strip()

            if not content and not media_paths:
                return

            inbound = InboundMessage(
                channel_type="feishu",
                external_id=msg.chat_id,
                sender_id=sender_id,
                sender_name=sender_id,
                content=content,
                message_type=msg_type,
                external_message_id=msg.message_id,
                media=media_paths,
                metadata={"app_id": self._app_id, "chat_type": chat_type},
            )

            # Bridge from the WS thread into the async event loop
            if self._message_handler and self._loop:
                asyncio.run_coroutine_threadsafe(
                    self._message_handler(inbound),
                    self._loop,
                )
        except Exception as e:
            logger.error(f"Error handling Feishu message: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Media download (sync — called from WS thread or via run_in_executor)
    # ------------------------------------------------------------------

    def _download_image(self, message_id: str, image_key: str) -> Optional[Path]:
        """Download an image from a Feishu message."""
        try:
            request = (
                GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(image_key)
                .type("image")
                .build()
            )
            response = self._client.im.v1.message_resource.get(request)
            if not response.success():
                logger.error(f"Download image failed: code={response.code} msg={response.msg}")
                return None
            filepath = MEDIA_DIR / f"{image_key}.png"
            filepath.write_bytes(response.file.read())
            return filepath
        except Exception as e:
            logger.error(f"Error downloading image {image_key}: {e}", exc_info=True)
            return None

    def _download_file(self, message_id: str, file_key: str, filename: str) -> Optional[Path]:
        """Download a file from a Feishu message."""
        try:
            request = (
                GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(file_key)
                .type("file")
                .build()
            )
            response = self._client.im.v1.message_resource.get(request)
            if not response.success():
                logger.error(f"Download file failed: code={response.code} msg={response.msg}")
                return None
            filepath = MEDIA_DIR / f"{file_key}_{filename}"
            filepath.write_bytes(response.file.read())
            return filepath
        except Exception as e:
            logger.error(f"Error downloading file {file_key}: {e}", exc_info=True)
            return None

    def _download_and_save_media(
        self, msg_type: str, content_json: str, message_id: str
    ) -> tuple[list[Path], str]:
        """Dispatch media download by message type.

        Returns (file_paths, extracted_text_description).
        """
        paths: list[Path] = []
        desc = ""

        try:
            content = json.loads(content_json)
        except (json.JSONDecodeError, TypeError):
            return paths, desc

        if msg_type == "image":
            image_key = content.get("image_key", "")
            if image_key:
                p = self._download_image(message_id, image_key)
                if p:
                    paths.append(p)
                    desc = "[User sent an image]"

        elif msg_type == "file":
            file_key = content.get("file_key", "")
            file_name = content.get("file_name", "file")
            if file_key:
                p = self._download_file(message_id, file_key, file_name)
                if p:
                    paths.append(p)
                    desc = f"[User sent a file: {file_name}]"

        elif msg_type == "audio":
            file_key = content.get("file_key", "")
            if file_key:
                p = self._download_file(message_id, file_key, "audio.opus")
                if p:
                    paths.append(p)
                    desc = "[User sent an audio message]"

        elif msg_type == "media":
            # Video message
            file_key = content.get("file_key", "")
            file_name = content.get("file_name", "video.mp4")
            if file_key:
                p = self._download_file(message_id, file_key, file_name)
                if p:
                    paths.append(p)
                    desc = f"[User sent a video: {file_name}]"

        return paths, desc

    # ------------------------------------------------------------------
    # Media upload (sync — called via run_in_executor)
    # ------------------------------------------------------------------

    def _upload_image(self, file_path: str) -> Optional[str]:
        """Upload an image to Feishu, returning the image_key."""
        try:
            with open(file_path, "rb") as f:
                request = (
                    CreateImageRequest.builder()
                    .request_body(
                        CreateImageRequestBody.builder()
                        .image_type("message")
                        .image(f)
                        .build()
                    )
                    .build()
                )
                response = self._client.im.v1.image.create(request)
            if not response.success():
                logger.error(f"Upload image failed: code={response.code} msg={response.msg}")
                return None
            return response.data.image_key
        except Exception as e:
            logger.error(f"Error uploading image {file_path}: {e}", exc_info=True)
            return None

    def _upload_file(self, file_path: str) -> Optional[str]:
        """Upload a file to Feishu, returning the file_key."""
        try:
            p = Path(file_path)
            content_type = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            # Map content type to Feishu file_type
            file_type = "stream"
            if content_type.startswith("audio/"):
                file_type = "opus"
            elif content_type == "application/pdf":
                file_type = "pdf"
            elif content_type in (
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ):
                file_type = "doc"
            elif content_type in (
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ):
                file_type = "xls"
            elif content_type in (
                "application/vnd.ms-powerpoint",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ):
                file_type = "ppt"

            with open(file_path, "rb") as f:
                request = (
                    CreateFileRequest.builder()
                    .request_body(
                        CreateFileRequestBody.builder()
                        .file_type(file_type)
                        .file_name(p.name)
                        .file(f)
                        .build()
                    )
                    .build()
                )
                response = self._client.im.v1.file.create(request)
            if not response.success():
                logger.error(f"Upload file failed: code={response.code} msg={response.msg}")
                return None
            return response.data.file_key
        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Outbound helpers
    # ------------------------------------------------------------------

    def _send_image_message(self, chat_id: str, image_key: str) -> None:
        """Send an image message to a Feishu chat."""
        request = (
            CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type("image")
                .content(json.dumps({"image_key": image_key}))
                .build()
            )
            .build()
        )
        response = self._client.im.v1.message.create(request)
        if not response.success():
            logger.error(
                f"Failed to send image message: code={response.code} msg={response.msg}"
            )

    def _send_file_message(self, chat_id: str, file_key: str, file_name: str) -> None:
        """Send a file message to a Feishu chat."""
        request = (
            CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type("file")
                .content(json.dumps({"file_key": file_key}))
                .build()
            )
            .build()
        )
        response = self._client.im.v1.message.create(request)
        if not response.success():
            logger.error(
                f"Failed to send file message: code={response.code} msg={response.msg}"
            )

    # ------------------------------------------------------------------
    # Text extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_post_text(content_json: str) -> str:
        """Extract plain text from a Feishu rich-text (post) message."""
        try:
            content = json.loads(content_json)
            # Post structure: {"title": "...", "content": [[{"tag":"text","text":"..."},...],...]}
            parts: list[str] = []
            title = content.get("title", "")
            if title:
                parts.append(title)
            for line in content.get("content", []):
                for elem in line:
                    if elem.get("tag") == "text":
                        parts.append(elem.get("text", ""))
                    elif elem.get("tag") == "a":
                        parts.append(elem.get("text", "") or elem.get("href", ""))
            return " ".join(parts).strip()
        except (json.JSONDecodeError, TypeError, AttributeError):
            return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_bot_open_id(self) -> Optional[str]:
        """Fetch the bot's own ``open_id`` via the Feishu REST API.

        This is used to filter out messages the bot sends to itself.
        Uses ``httpx`` (already a project dependency) to obtain a
        tenant_access_token and then call ``GET /bot/v3/info``.
        """
        try:
            import httpx

            base_url = "https://open.feishu.cn/open-apis"

            async with httpx.AsyncClient(timeout=10) as http:
                # Step 1: obtain tenant access token
                token_resp = await http.post(
                    f"{base_url}/auth/v3/tenant_access_token/internal",
                    json={
                        "app_id": self._app_id,
                        "app_secret": self._app_secret,
                    },
                )
                token_data = token_resp.json()
                tenant_token = token_data.get("tenant_access_token")
                if not tenant_token:
                    logger.warning(
                        f"Failed to get tenant_access_token: {token_data.get('msg')}"
                    )
                    return None

                # Step 2: fetch bot info
                bot_resp = await http.get(
                    f"{base_url}/bot/v3/info",
                    headers={"Authorization": f"Bearer {tenant_token}"},
                )
                bot_data = bot_resp.json()
                return bot_data.get("bot", {}).get("open_id")

        except Exception as e:
            logger.warning(f"Failed to get Feishu bot open_id: {e}")
            return None
