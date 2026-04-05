"""GLM Realtime pipeline for Zhipu AI GLM-Realtime speech-to-speech API.

This pipeline extends the realtime pipeline to support Zhipu's GLM-Realtime API,
which is compatible with OpenAI's Realtime API but has several protocol differences:

- Event names omit the ``output_`` segment (``response.audio.delta`` vs
  ``response.output_audio.delta``)
- Sends ``heartbeat`` keepalive events and ``rate_limites.updated`` (sic)
- Sends ``conversation.created`` before the first response
- Tool format uses flat keys (same as OpenAI Realtime, not nested ``function``)
- Session config requires ``beta_fields`` (``chat_mode``, ``tts_source``)
- Function calls arrive via ``response.function_call_arguments.done`` (NOT in
  the ``response.done`` output array, unlike Grok/xAI)
- Built-in ``inner_tool`` events for GLM's internal tools (ignored)

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
import time
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


def _sanitize_tools_for_glm(tools: list) -> list:
    """Ensure every tool has non-empty ``properties`` and ``required`` for GLM.

    GLM's realtime session stores tools in flat format, but its inference
    backend converts empty ``{}``/``[]`` to ``null`` during an internal
    flat→nested transformation — which Pydantic then rejects with HTTP 422.

    Fix: guarantee every tool has at least one property and a non-empty
    ``required`` list so nothing maps to ``null``.  For no-arg tools like
    ``end_session``, a harmless ``confirm`` flag is injected.
    """
    for tool in tools:
        func = tool.get("function", tool)
        params = func.get("parameters")
        if params is None:
            params = {"type": "object"}
            func["parameters"] = params

        params.setdefault("type", "object")

        props = params.get("properties")
        if not props:
            params["properties"] = {
                "confirm": {
                    "type": "string",
                    "description": "Pass 'yes' to confirm this action.",
                }
            }
            params["required"] = ["confirm"]
        else:
            req = params.get("required")
            if not req:
                first_key = next(iter(props))
                params["required"] = [first_key]

    return tools


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
    - Function calls arrive via ``response.function_call_arguments.done``
    - Built-in ``inner_tool`` events for GLM's internal tools (web search etc.)

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
        self._pending_glm_function_calls: list[dict] = []  # queued from function_call_arguments.done

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

    async def _handle_user_started_speaking(self, frame):
        """Clear GLM's server-side audio buffer when a new turn starts.

        GLM has a 10 MB server-side audio buffer limit. Without clearing
        between turns, residual audio from prior turns accumulates and
        eventually triggers ``audio_buffer_size_exceeded``.

        Guarded: if a response is already pending (we committed audio and
        are waiting for GLM to reply), ignore spurious start events from
        Pipecat's transcription-based turn strategy.  Without this guard,
        the TranscriptionUserTurnStartStrategy fires between the VAD-based
        stop and the transcription-based stop, clearing the just-committed
        buffer and resetting the pending flag — which allows a second
        commit of an empty/stale buffer and doubles the response latency.
        """
        if self._glm_response_pending:
            logger.debug("[GLM] Ignoring user-started-speaking while response is pending")
            return
        await self._ws_send({"type": "input_audio_buffer.clear"})
        logger.debug("[GLM] Cleared server audio buffer for new turn")
        await super()._handle_user_started_speaking(frame)

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
            beta = session.get("beta_fields", {})
            beta.setdefault("chat_mode", "audio")
            beta.setdefault("tts_source", "e2e")
            beta["auto_search"] = False
            session["beta_fields"] = beta

            if "tools" in session:
                _sanitize_tools_for_glm(session["tools"])

            dump["session"] = session
            # Log the actual tool JSON for debugging schema issues
            for i, tool in enumerate(session.get("tools", [])):
                logger.debug(f"[GLM] Tool[{i}] after sanitization: {json.dumps(tool)}")
            logger.info(f"[GLM] Sending reformatted session.update (keys: {list(session.keys())})")
            await self._ws_send(dump)
            return

        await super().send_client_event(event)

    async def _handle_glm_response_done(self, raw_event):
        """Execute pending function calls and send results back to GLM.

        GLM sends function call details in ``response.function_call_arguments.done``
        events (queued in ``_pending_glm_function_calls``). When ``response.done``
        arrives we execute each queued call, send the result via
        ``conversation.item.create`` (``function_call_output``), then trigger
        ``response.create`` so the model continues with the tool output in context.

        Also checks the ``response.done`` output array as a fallback (Grok/xAI
        style), deduplicating by ``call_id``.
        """
        # Collect calls: primary source is the pending queue from
        # response.function_call_arguments.done; fallback is response.done output.
        all_calls: list[dict] = list(self._pending_glm_function_calls)
        self._pending_glm_function_calls.clear()

        seen_call_ids = {c.get("call_id") for c in all_calls}
        response = raw_event.get("response", {})
        for item in response.get("output", []):
            if item.get("type") == "function_call" and item.get("call_id") not in seen_call_ids:
                all_calls.append(item)

        function_call_ids_handled = []
        for item in all_calls:
            call_id = item.get("call_id")
            func_name = item.get("name")
            arguments_str = item.get("arguments", "{}")

            logger.info(f"[GLM] Executing function call: {func_name}")
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

    async def _reconnect_on_disconnect(self):
        """Reset all per-session state before reconnecting.

        Without this, stale flags from the old session break the new one:
        - ``_glm_initial_session_configured``: must be False so
          ``_update_settings`` sends ``session.update`` to the fresh session.
        - ``_glm_response_pending``: must be False so the next turn's
          ``_handle_user_stopped_speaking`` actually commits audio and
          sends ``response.create``.  Otherwise GLM never responds.
        - ``_glm_waiting_for_committed``: stale True blocks response.create.
        - ``_pending_glm_function_calls``: stale calls from the old session.
        """
        self._glm_initial_session_configured = False
        self._glm_response_pending = False
        self._glm_waiting_for_committed = False
        self._pending_glm_function_calls.clear()
        await super()._reconnect_on_disconnect()

    async def _update_settings(self):
        """Send session.update to GLM only once per session.

        Pipecat calls _update_settings() on context changes, but GLM treats
        each session.update as a signal to start a new response — causing
        multi-response cascades.  Guard to send only the initial config.
        The flag is reset by ``_reconnect_on_disconnect`` before each new
        session.
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
                        if self._glm_waiting_for_committed:
                            self._glm_waiting_for_committed = False
                            # GLM client_vad requires the client to send
                            # response.create after committing audio.
                            logger.info("[GLM] Sending response.create after commit")
                            await self._ws_send({"type": "response.create"})
                        continue

                    # GLM error events may lack fields OpenAI requires (e.g. error.type)
                    if event_type == "error":
                        err = raw_event.get("error", {})
                        err_msg = str(err.get("message", ""))
                        logger.error(f"[GLM] Server error: code={err.get('code')}, message={err_msg}")
                        if "input validation error" in err_msg.lower() or "max_new_tokens" in err_msg.lower():
                            logger.error(
                                f"[GLM] Context overflow detected — breaking receive loop "
                                f"to trigger reconnection with truncated history"
                            )
                            break
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
                        # GLM sends function call details HERE, not in the
                        # response.done output array (unlike Grok/xAI).
                        # Queue the call; it will be executed when response.done arrives.
                        call_id = raw_event.get("call_id")
                        func_name = raw_event.get("name")
                        arguments_str = raw_event.get("arguments", "{}")
                        if call_id and func_name:
                            self._pending_glm_function_calls.append({
                                "call_id": call_id,
                                "name": func_name,
                                "arguments": arguments_str,
                            })
                            logger.info(
                                f"[GLM] Queued function call: {func_name} "
                                f"(call_id={call_id}, args={arguments_str})"
                            )
                        else:
                            logger.warning(
                                f"[GLM] Incomplete function_call_arguments.done event "
                                f"(call_id={call_id}, name={func_name})"
                            )
                        continue

                    # GLM's built-in internal tools (web search, etc.) fire
                    # these events. They are NOT our custom function calls —
                    # those come through response.function_call_arguments.done.
                    if event_type in (
                        "response.function_call.inner_tool",
                        "response.function_call.inner_tool.result",
                    ):
                        logger.debug(f"[GLM] Built-in inner_tool event, ignoring: {event_type}")
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

    Context management:
    - Reactive reconnection when "input validation error" (context overflow)
      is received — the receive loop breaks and ``_handle_ws_close`` triggers
      ``_reconnect_on_disconnect``
    - Budget-aware history injection on reconnect: tool-call entries are
      prioritized, then recent Q&A fills the remaining char budget
    - Falls back to zero history when the system instruction already fills
      the char budget
    """

    requires_service = False

    # GLM Realtime has 8192 total tokens (7168 input + 1024 max_new_tokens).
    # Prompts exceeding this are positionally truncated.
    GLM_MAX_INSTRUCTION_CHARS = 12000  # ~3K tokens

    GLM_MAX_SINGLE_ENTRY_CHARS = 500
    GLM_HISTORY_HEADER_CHARS = 200

    # ------------------------------------------------------------------
    # LLM creation
    # ------------------------------------------------------------------

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
        if len(system_instruction) > self.GLM_MAX_INSTRUCTION_CHARS:
            original_len = len(system_instruction)
            system_instruction = system_instruction[:self.GLM_MAX_INSTRUCTION_CHARS]
            logger.warning(
                f"[GLM] System instruction too large ({original_len} chars, "
                f"~{original_len // 4} tokens). Truncated to "
                f"{self.GLM_MAX_INSTRUCTION_CHARS} chars (~{self.GLM_MAX_INSTRUCTION_CHARS // 4} tokens)."
            )

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

        if "air" in model.lower():
            self._turn_watchdog_timeout = 90.0
            logger.info(f"[GLM] Air model detected — watchdog timeout set to {self._turn_watchdog_timeout}s")

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

    # ------------------------------------------------------------------
    # Reconnection callbacks (override base to use budget-aware history)
    # ------------------------------------------------------------------

    def _on_ws_reconnecting(self):
        """Pause audio and enrich system instructions with budgeted history.

        Overrides the base ``_on_ws_reconnecting`` to use
        ``_update_instructions_with_history_for_glm`` which respects
        GLM's tight char/token budget.
        """
        logger.info(f"[GLM] Reconnecting: pausing audio, turn {self.turn_idx} will be retried")
        self.needs_turn_retry = True
        self.paced_input.pause()
        self.assistant_shim.clear_buffer()
        if self.turn_gate:
            self.turn_gate.clear_pending()
        self.reconnection_grace_until = time.monotonic() + 10.0

        self._update_instructions_with_history_for_glm()

    # ------------------------------------------------------------------
    # Budget-aware history injection
    # ------------------------------------------------------------------

    def _update_instructions_with_history_for_glm(self):
        """Inject conversation history into GLM system instructions within budget.

        Unlike the base ``_update_instructions_with_history`` which simply
        appends the last N turns, this method:

        1. Computes available char budget from ``GLM_MAX_INSTRUCTION_CHARS``
           minus the original (truncated) system instruction.
        2. Prioritises tool-call entries (they carry state: bookings,
           registrations, schedule changes).
        3. Fills remaining budget with the most recent Q&A entries.
        4. Truncates individual entries that exceed ``GLM_MAX_SINGLE_ENTRY_CHARS``.
        5. Falls back to zero history if no budget remains.
        """
        if not self._conversation_history:
            logger.info("[GLM] No conversation history to inject")
            return

        original = getattr(self.benchmark, "system_instruction", "")
        if len(original) > self.GLM_MAX_INSTRUCTION_CHARS:
            original = original[:self.GLM_MAX_INSTRUCTION_CHARS]

        budget = self.GLM_MAX_INSTRUCTION_CHARS - len(original) - self.GLM_HISTORY_HEADER_CHARS
        if budget <= 0:
            logger.warning(
                f"[GLM] No char budget for history injection "
                f"(instruction={len(original)} chars, limit={self.GLM_MAX_INSTRUCTION_CHARS}). "
                f"Zero-context reconnection."
            )
            self.llm._session_properties.instructions = original
            return

        history = list(self._conversation_history)

        tool_entries = [e for e in history if e.get("tool_calls")]
        qa_entries = [e for e in history if not e.get("tool_calls")]

        kept_lines: list[str] = []
        used_chars = 0

        for entry in tool_entries:
            block = self._format_history_entry(entry)
            if used_chars + len(block) > budget:
                break
            kept_lines.append(block)
            used_chars += len(block)

        for entry in reversed(qa_entries):
            block = self._format_history_entry(entry)
            if used_chars + len(block) > budget:
                break
            kept_lines.append(block)
            used_chars += len(block)

        if not kept_lines:
            logger.warning("[GLM] History entries too large for budget, zero-context reconnection")
            self.llm._session_properties.instructions = original
            return

        header = (
            "\n\n--- CONVERSATION HISTORY ---\n"
            "The following is the conversation so far. The user's name, "
            "preferences, and any actions you have taken (tool calls, "
            "registrations, schedule changes, etc.) are still in effect. "
            "Continue naturally.\n\n"
        )
        enriched = original + header + "\n".join(kept_lines)

        self.llm._session_properties.instructions = enriched
        logger.info(
            f"[GLM] Enriched instructions with {len(kept_lines)} of "
            f"{len(history)} history entries "
            f"({used_chars} history chars, {len(enriched)} total chars)"
        )

    def _format_history_entry(self, entry: dict) -> str:
        """Format a single conversation history entry, truncating if needed."""
        lines = [f"User: {entry.get('user', '')}"]
        if entry.get("tool_calls"):
            for tc in entry["tool_calls"]:
                lines.append(f"  [Tool call: {tc['name']}({tc['args']})]")
        if entry.get("tool_results"):
            for tr in entry["tool_results"]:
                lines.append(f"  [Tool result: {json.dumps(tr.get('response', {}))}]")
        lines.append(f"Assistant: {entry.get('assistant', '')}")
        block = "\n".join(lines)
        if len(block) > self.GLM_MAX_SINGLE_ENTRY_CHARS:
            block = block[:self.GLM_MAX_SINGLE_ENTRY_CHARS - 15] + "... [truncated]"
        return block

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    async def _retry_current_turn(self):
        """Reset GLM per-turn flags before retry so the next commit goes through."""
        self.llm._glm_response_pending = False
        self.llm._glm_waiting_for_committed = False
        await super()._retry_current_turn()

    async def _on_turn_end(self, assistant_text: str) -> None:
        """Reset per-turn LLM flags so the next turn can commit cleanly."""
        self.llm._glm_response_pending = False
        self.llm._glm_waiting_for_committed = False
        await super()._on_turn_end(assistant_text)

