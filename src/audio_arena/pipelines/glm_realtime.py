"""GLM Realtime pipeline for Zhipu AI GLM-Realtime speech-to-speech API.

This pipeline extends the realtime pipeline to support Zhipu's GLM-Realtime API,
which is compatible with OpenAI's Realtime API but has several protocol differences:

- Event names omit the ``output_`` segment (``response.audio.delta`` vs
  ``response.output_audio.delta``)
- Sends ``heartbeat`` keepalive events and ``rate_limites.updated`` (sic)
- Sends ``conversation.created`` before the first response
- Tool format uses flat keys (same as OpenAI Realtime, not nested ``function``)
- Session config requires ``beta_fields`` (``chat_mode``, ``tts_source``)
- Function calls arrive in ``response.function_call_arguments.done`` and are also
  included in the ``response.done`` output array

Usage:
    uv run audio-arena run conversation_bench --model glm-realtime-flash
    uv run audio-arena run grocery_bench --model glm-realtime-air
"""

import asyncio
import base64
import io
import json
import os
import struct
import wave
from typing import Callable, Optional

from loguru import logger

from pipecat.frames.frames import LLMFullResponseEndFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.openai.realtime import events as rt_events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from audio_arena.pipelines.openai_realtime import ReconnectOnDisconnectMixin
from audio_arena.pipelines.realtime import RealtimePipeline

_GLM_AUDIO_FLUSH_INTERVAL = 0.250  # 250ms → 4 sends/sec; GLM VAD needs larger chunks
_PIPECAT_SAMPLE_RATE = 24000
_GLM_SAMPLE_RATE = 16000  # GLM default
_GLM_CHANNELS = 1
_GLM_SAMPLE_WIDTH = 2  # 16-bit PCM


def _resample_pcm16(pcm_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Naive linear-interpolation resample of 16-bit mono PCM."""
    if src_rate == dst_rate:
        return pcm_bytes
    samples = struct.unpack(f"<{len(pcm_bytes) // 2}h", pcm_bytes)
    n_src = len(samples)
    n_dst = int(n_src * dst_rate / src_rate)
    ratio = src_rate / dst_rate
    out = []
    for i in range(n_dst):
        src_idx = i * ratio
        idx = int(src_idx)
        frac = src_idx - idx
        s0 = samples[min(idx, n_src - 1)]
        s1 = samples[min(idx + 1, n_src - 1)]
        out.append(int(s0 + frac * (s1 - s0)))
    return struct.pack(f"<{len(out)}h", *out)


def _pcm_to_wav(pcm_bytes: bytes) -> bytes:
    """Resample 24kHz PCM to 16kHz and wrap in WAV for GLM."""
    resampled = _resample_pcm16(pcm_bytes, _PIPECAT_SAMPLE_RATE, _GLM_SAMPLE_RATE)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_GLM_CHANNELS)
        wf.setsampwidth(_GLM_SAMPLE_WIDTH)
        wf.setframerate(_GLM_SAMPLE_RATE)
        wf.writeframes(resampled)
    return buf.getvalue()


class GLMRealtimeLLMService(ReconnectOnDisconnectMixin, OpenAIRealtimeLLMService):
    """Zhipu GLM-Realtime speech-to-speech API service.

    Extends OpenAI Realtime service to handle GLM-specific protocol differences:
    - ``heartbeat`` keepalive events
    - ``conversation.created`` event after session creation
    - ``rate_limites.updated`` (their typo) rate-limit notices
    - Event names without ``output_`` prefix for audio/text/transcript deltas
    - Function calls in ``response.done`` output array (like Grok)

    Auto-reconnects on unexpected WS close via ``ReconnectOnDisconnectMixin``.
    """

    def __init__(
        self,
        get_last_tool_result: Optional[Callable[[], dict]] = None,
        on_reconnecting: Optional[Callable[[], None]] = None,
        on_reconnected: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        raw_base_url = kwargs.get("base_url", "wss://open.bigmodel.cn/api/paas/v4/realtime")
        self._model_name = kwargs.get("model", "glm-realtime-flash")
        super().__init__(**kwargs)
        # GLM doesn't use ?model= query param — restore the raw URL.
        self.base_url = raw_base_url
        self._get_last_tool_result = get_last_tool_result
        self._init_reconnection_callbacks(on_reconnecting, on_reconnected)

        # GLM's server_vad doesn't reliably fire speech_stopped, so we
        # always use client_vad and commit + create responses manually.
        self._manual_turn_handling = True
        self._glm_initial_session_configured = False
        self._glm_response_pending = False  # guard against double commit+create
        self._glm_waiting_for_committed = False  # delay response.create until committed ack

        # Audio buffering to stay under GLM's 50 QPS limit on
        # input_audio_buffer.append events.
        self._audio_buf = bytearray()
        self._audio_buf_lock = asyncio.Lock()
        self._audio_flush_task: Optional[asyncio.Task] = None

    async def _start_audio_flush_loop(self):
        """Periodically flush buffered audio to GLM at a throttled rate."""
        try:
            while True:
                await asyncio.sleep(_GLM_AUDIO_FLUSH_INTERVAL)
                await self._flush_audio_buf()
        except asyncio.CancelledError:
            await self._flush_audio_buf()

    async def _flush_audio_buf(self):
        """Send any accumulated PCM bytes as a WAV-wrapped append event."""
        async with self._audio_buf_lock:
            if not self._audio_buf:
                return
            pcm_chunk = bytes(self._audio_buf)
            self._audio_buf.clear()
        wav_bytes = _pcm_to_wav(pcm_chunk)
        encoded = base64.b64encode(wav_bytes).decode("ascii")
        await self._ws_send({
            "type": "input_audio_buffer.append",
            "audio": encoded,
        })

    def _ensure_audio_flush_loop(self):
        """Start the flush loop if it isn't already running."""
        if self._audio_flush_task is None or self._audio_flush_task.done():
            self._audio_flush_task = asyncio.ensure_future(self._start_audio_flush_loop())

    async def _handle_user_stopped_speaking(self, frame):
        """Manually commit and create response when VAD is disabled.

        Guarded so Pipecat's duplicate UserStoppedSpeakingFrame events
        don't trigger multiple commits within one turn.
        """
        if self._manual_turn_handling:
            if self._glm_response_pending:
                logger.debug("[GLM] Ignoring duplicate user-stopped-speaking (response already pending)")
                return
            self._glm_response_pending = True
            logger.info("[GLM] User stopped speaking, committing audio")
            await self._flush_audio_buf()
            if self._audio_flush_task and not self._audio_flush_task.done():
                self._audio_flush_task.cancel()
            self._glm_waiting_for_committed = True
            await self.send_client_event(rt_events.InputAudioBufferCommitEvent())
        else:
            await super()._handle_user_stopped_speaking(frame)

    async def send_client_event(self, event):
        """Reformat session.update events to GLM's expected schema.

        GLM expects ``turn_detection``, ``input_audio_format``, ``output_audio_format``,
        ``voice``, ``modalities``, and ``beta_fields`` as top-level session fields
        instead of OpenAI's nested ``audio.input.turn_detection`` structure.
        """
        # Buffer audio append events to stay under 50 QPS.
        if hasattr(event, "type") and event.type == "input_audio_buffer.append":
            audio_b64 = getattr(event, "audio", None)
            if audio_b64:
                raw = base64.b64decode(audio_b64)
                async with self._audio_buf_lock:
                    self._audio_buf.extend(raw)
                self._ensure_audio_flush_loop()
            return

        if hasattr(event, "type") and event.type == "session.update":
            dump = event.model_dump(exclude_none=True)
            session = dump.get("session", {})

            # Extract turn_detection from OpenAI's nested audio.input path
            audio = session.pop("audio", None)
            if audio and "input" in audio:
                td = audio["input"].get("turn_detection")
                if td is not None:
                    session["turn_detection"] = td

            # Force client_vad — GLM's server_vad doesn't reliably detect
            # speech_stopped, so we commit + create responses manually.
            session["turn_detection"] = {"type": "client_vad"}

            # GLM-required fields
            session.setdefault("model", self._model_name)
            session.setdefault("modalities", ["audio", "text"])
            session.setdefault("input_audio_format", "wav")
            session.setdefault("output_audio_format", "pcm")
            session.setdefault("voice", "tongtong")
            session.setdefault("beta_fields", {
                "chat_mode": "audio",
                "tts_source": "e2e",
            })

            dump["session"] = session
            logger.info(f"[GLM] Sending reformatted session.update (keys: {list(session.keys())})")
            await self._ws_send(dump)
            return

        await super().send_client_event(event)

    async def _handle_glm_response_done(self, raw_event):
        """Handle GLM's response.done format which includes function calls in the output array.

        Runs each tool via run_function_calls(), then sends the result back with
        conversation.item.create (function_call_output) and triggers response.create
        so the model continues with the tool output in context.
        """
        response = raw_event.get("response", {})
        output_items = response.get("output", [])
        function_call_ids_handled = []

        for item in output_items:
            item_type = item.get("type")
            if item_type == "function_call":
                call_id = item.get("call_id")
                func_name = item.get("name")
                arguments_str = item.get("arguments", "{}")

                logger.info(f"[GLM] Function call detected in response.done: {func_name}")
                logger.debug(f"[GLM]   call_id: {call_id}")
                logger.debug(f"[GLM]   arguments: {arguments_str}")

                try:
                    args = json.loads(arguments_str)
                    function_calls = [
                        FunctionCallFromLLM(
                            context=self._context,
                            tool_call_id=call_id,
                            function_name=func_name,
                            arguments=args,
                        )
                    ]
                    await self.run_function_calls(function_calls)
                    function_call_ids_handled.append(call_id)
                    logger.info(f"[GLM] Executed function call: {func_name}")

                    tool_result = (
                        self._get_last_tool_result()
                        if self._get_last_tool_result
                        else {"status": "success"}
                    )
                    output_json = json.dumps(tool_result)
                    create_ev = rt_events.ConversationItemCreateEvent(
                        item=rt_events.ConversationItem(
                            type="function_call_output",
                            call_id=call_id,
                            output=output_json,
                        )
                    )
                    await self.send_client_event(create_ev)
                    logger.info(f"[GLM] Sent function_call_output for call_id={call_id}")
                except Exception as e:
                    logger.error(f"[GLM] Failed to execute function call {func_name}: {e}")

        if function_call_ids_handled:
            await self.send_client_event(rt_events.ResponseCreateEvent())
            logger.info("[GLM] Sent response.create to continue after tool result(s)")

    async def _update_settings(self):
        """Send session.update to GLM only once (initial setup).

        Pipecat calls _update_settings() on context changes, but GLM treats
        each session.update as a signal to start a new response — causing
        multi-response cascades.  Guard to send only the initial config.
        """
        if self._glm_initial_session_configured:
            logger.debug("[GLM] Session already configured, skipping redundant update")
            return
        self._glm_initial_session_configured = True

        if self._session_properties.tools:
            if hasattr(self._session_properties.tools, "standard_tools"):
                for t in self._session_properties.tools.standard_tools:
                    logger.debug(f"[GLM]   - Tool: {t.name}")
        else:
            logger.warning("[GLM] No tools in session_properties!")

        await super()._update_settings()
        logger.debug("[GLM] Initial session update sent")

    async def _receive_task_handler(self):
        """Custom receive loop that handles GLM-specific events and name mappings.

        GLM-Realtime uses slightly different event names than OpenAI:
        - ``response.audio.delta`` instead of ``response.output_audio.delta``
        - ``response.text.delta`` instead of ``response.output_text.delta``
        - ``response.audio_transcript.delta`` instead of ``response.output_audio_transcript.delta``
        - Extra events: ``heartbeat``, ``rate_limites.updated``, ``conversation.created``

        For audio/text/transcript events we parse the raw JSON and rewrite the type
        field so we can delegate to the standard OpenAI handlers (which expect the
        ``output_`` naming). For everything else, we handle or skip as appropriate.
        """
        try:
            async for message in self._websocket:
                try:
                    raw_event = json.loads(message)
                    event_type = raw_event.get("type")

                    logger.info(f"[GLM] Received event: {event_type}")

                    # --- GLM-specific events to ignore or handle specially ---

                    if event_type == "heartbeat":
                        logger.debug("[GLM] Heartbeat received, ignoring")
                        continue

                    if event_type == "rate_limites.updated":
                        logger.debug("[GLM] Rate limits updated, ignoring")
                        continue

                    if event_type == "conversation.created":
                        logger.info("[GLM] Conversation created")
                        continue

                    if event_type == "session.updated":
                        session_data = raw_event.get("session", {})
                        tools_in_response = session_data.get("tools", [])
                        logger.debug(f"[GLM] session.updated - tools count: {len(tools_in_response)}")
                        logger.info("[GLM] Session updated")
                        self._api_session_ready = True
                        # Only auto-create response on the very first session.updated
                        # (initial setup).  Later session.updated events must NOT
                        # trigger new responses — GLM would cascade multiple replies.
                        continue

                    if event_type == "response.created":
                        logger.info("[GLM] Response created")
                        continue

                    if event_type == "input_audio_buffer.committed":
                        logger.debug("[GLM] Audio buffer committed")
                        self._glm_waiting_for_committed = False
                        # GLM auto-starts inference after commit in client_vad
                        # mode, so we do NOT send an explicit response.create
                        # (doing so would create a duplicate response).
                        continue

                    # GLM error events may lack fields OpenAI requires (e.g. error.type)
                    if event_type == "error":
                        err = raw_event.get("error", {})
                        logger.error(f"[GLM] Server error: code={err.get('code')}, message={err.get('message')}")
                        continue

                    if event_type == "response.content_part.added":
                        logger.debug("[GLM] Content part added")
                        continue

                    if event_type == "response.content_part.done":
                        logger.debug("[GLM] Content part done")
                        continue

                    if event_type == "response.output_item.added":
                        logger.debug("[GLM] Output item added")
                        continue

                    if event_type == "response.output_item.done":
                        logger.debug("[GLM] Output item done")
                        continue

                    if event_type == "response.function_call_arguments.delta":
                        logger.debug("[GLM] Function call arguments delta")
                        continue

                    if event_type == "response.function_call_arguments.done":
                        # Handled in response.done instead (same as Grok)
                        logger.debug("[GLM] Function call arguments done")
                        continue

                    # GLM's speech_started/stopped events lack item_id, which
                    # fails OpenAI's pydantic model. Handle them directly.
                    if event_type == "input_audio_buffer.speech_started":
                        logger.info(f"[GLM] Speech started (audio_start_ms={raw_event.get('audio_start_ms')})")
                        continue

                    if event_type == "input_audio_buffer.speech_stopped":
                        logger.info(f"[GLM] Speech stopped (audio_end_ms={raw_event.get('audio_end_ms')})")
                        continue

                    if event_type == "conversation.item.added":
                        item = raw_event.get("item", {})
                        item_role = item.get("role")
                        item_type_inner = item.get("type")
                        if item_role == "tool" or item_type_inner in ("function_call", "function_call_output"):
                            logger.debug(f"[GLM] Conversation item added (type={item_type_inner}, role={item_role})")
                            continue

                    if event_type == "conversation.item.created":
                        logger.debug("[GLM] Conversation item created")
                        continue

                    if event_type == "response.done":
                        await self._handle_glm_response_done(raw_event)
                        await self.push_frame(LLMFullResponseEndFrame())
                        self._current_assistant_response = None
                        self._glm_response_pending = False
                        # Clear any residual audio so GLM doesn't auto-start
                        # a follow-up response from buffered silence.
                        await self._ws_send({"type": "input_audio_buffer.clear"})
                        continue

                    # --- Map GLM event names to OpenAI equivalents and delegate ---

                    # GLM uses ``response.audio.delta`` where OpenAI uses
                    # ``response.output_audio.delta``. Rewrite in the raw JSON
                    # before handing off to the Pipecat parser.
                    # Debug: log audio delta details to diagnose sparse audio
                    if event_type == "response.audio.delta":
                        delta = raw_event.get("delta", "")
                        raw_bytes = base64.b64decode(delta) if delta else b""
                        logger.info(f"[GLM] Audio delta: {len(raw_bytes)} bytes, first4={raw_bytes[:4].hex() if raw_bytes else 'empty'}")

                    remapped = False
                    glm_to_openai = {
                        "response.audio.delta": "response.output_audio.delta",
                        "response.audio.done": "response.output_audio.done",
                        "response.text.delta": "response.output_text.delta",
                        "response.text.done": "response.output_text.done",
                        "response.audio_transcript.delta": "response.output_audio_transcript.delta",
                        "response.audio_transcript.done": "response.output_audio_transcript.done",
                    }
                    if event_type in glm_to_openai:
                        raw_event["type"] = glm_to_openai[event_type]
                        message = json.dumps(raw_event)
                        remapped = True

                    evt = rt_events.parse_server_event(message)
                    if evt.type == "session.created":
                        await self._handle_evt_session_created(evt)
                    elif evt.type == "session.updated":
                        await self._handle_evt_session_updated(evt)
                    elif evt.type == "response.output_audio.delta":
                        await self._handle_evt_audio_delta(evt)
                    elif evt.type == "response.output_audio.done":
                        await self._handle_evt_audio_done(evt)
                    elif evt.type == "conversation.item.added":
                        await self._handle_evt_conversation_item_added(evt)
                    elif evt.type == "conversation.item.done":
                        await self._handle_evt_conversation_item_done(evt)
                    elif evt.type == "conversation.item.input_audio_transcription.completed":
                        await self.handle_evt_input_audio_transcription_completed(evt)
                    elif evt.type == "conversation.item.input_audio_transcription.delta":
                        await self._handle_evt_input_audio_transcription_delta(evt)
                    elif evt.type == "conversation.item.retrieved":
                        await self._handle_conversation_item_retrieved(evt)
                    elif evt.type == "response.done":
                        await self._handle_evt_response_done(evt)
                    elif evt.type == "input_audio_buffer.speech_started":
                        await self._handle_evt_speech_started(evt)
                    elif evt.type == "input_audio_buffer.speech_stopped":
                        await self._handle_evt_speech_stopped(evt)
                    elif evt.type == "response.output_text.delta":
                        await self._handle_evt_text_delta(evt)
                    elif evt.type == "response.output_audio_transcript.delta":
                        await self._handle_evt_audio_transcript_delta(evt)
                    elif evt.type == "response.function_call_arguments.done":
                        await self._handle_evt_function_call_arguments_done(evt)
                    elif evt.type == "error":
                        if not await self._maybe_handle_evt_retrieve_conversation_item_error(evt):
                            await self._handle_evt_error(evt)
                            return
                    else:
                        if not remapped:
                            logger.debug(f"[GLM] Ignoring unhandled event type: {evt.type}")
                except Exception as e:
                    logger.warning(f"[GLM] Error processing event: {e}")
        except Exception as e:
            logger.warning(f"[GLM] WebSocket receive loop ended with exception: {e}")

        await self._handle_ws_close()


class GLMRealtimePipeline(RealtimePipeline):
    """Pipeline for Zhipu AI GLM-Realtime speech-to-speech API.

    Extends RealtimePipeline to use GLM-specific configuration:
    - ZHIPU_API_KEY environment variable
    - wss://open.bigmodel.cn/api/paas/v4/realtime endpoint
    - GLMRealtimeLLMService for protocol handling
    - GLM-specific session config (beta_fields, audio format)
    - Automatic reconnection with conversation history replay on disconnect
    """

    requires_service = False

    def _create_llm(
        self, service_class: Optional[type], model: str
    ) -> FrameProcessor:
        """Create Zhipu GLM-Realtime LLM service.

        Args:
            service_class: Ignored - always uses GLMRealtimeLLMService.
            model: Model name (e.g., "glm-realtime-flash", "glm-realtime-air").

        Returns:
            Configured GLMRealtimeLLMService instance.
        """
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise EnvironmentError("ZHIPU_API_KEY environment variable is required for GLM-Realtime models")

        base_url = "wss://open.bigmodel.cn/api/paas/v4/realtime"
        logger.info(f"Using Zhipu GLM-Realtime API at {base_url}")

        system_instruction = getattr(self.benchmark, "system_instruction", "")
        tools = getattr(self.benchmark, "tools_schema", None)

        audio_config = rt_events.AudioConfiguration(
            input=rt_events.AudioInput(
                turn_detection=False,
            )
        )

        session_props = rt_events.SessionProperties(
            instructions=system_instruction,
            tools=tools,
            audio=audio_config,
        )

        return GLMRealtimeLLMService(
            api_key=api_key,
            model=model,
            base_url=base_url,
            system_instruction=system_instruction,
            session_properties=session_props,
            get_last_tool_result=lambda: getattr(
                self, "_last_tool_result", {"status": "success"}
            ),
            on_reconnecting=self._on_ws_reconnecting,
            on_reconnected=self._on_ws_reconnected,
        )
