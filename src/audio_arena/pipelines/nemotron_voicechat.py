"""NVIDIA NemotronLabs VoiceChat pipeline (self-hosted, OpenAI Realtime-compatible).

VoiceChat is a full-duplex speech-to-speech model served by NVIDIA's inference
container (``nvcr.io/nim/nvidia/nemotron-labs-voicechat``). Its WebSocket server
speaks a subset of the OpenAI Realtime protocol with these differences:

- One server "response" spans the whole session (``response.done`` only arrives at
  session end). Each agent utterance ends with
  ``response.output_audio_transcript.done``, so transcript boundaries drive
  TTS start/stop.
- The server streams digital silence (all-zero PCM) whenever the agent is not
  speaking. Those chunks are dropped so the output transport can detect when the
  agent stops speaking; the recorder re-inserts wall-clock silence for gaps.
- ``transcript.done`` can arrive seconds after the utterance's audio ends, later
  than the output transport's bot-stopped detection. Utterances therefore also
  close after a short run of digital silence, keeping TTSStopped ahead of
  BotStoppedSpeaking as RealtimePipeline's TurnGate expects.
- Tool calls arrive only as ``response.function_call_arguments.done`` (no
  ``conversation.item.added``). The server never acknowledges
  ``function_call_output`` items and ignores ``response.create``, so tool results
  are sent immediately and the model continues on its own.
- System prompt, tool schemas and tool outputs must be ASCII.

Server setup and required serving limits: docs/nemotron-voicechat.md

Usage:
    export NEMOTRON_VOICECHAT_WS_URL=ws://localhost:9000/v1/realtime
    uv run audio-arena run appointment_bench --model nemotron-voicechat
"""

import asyncio
import base64
import json
import os
import time
import unicodedata
from typing import Any, Awaitable, Callable, Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallFromLLM, FunctionCallParams
from pipecat.services.openai.realtime import events as rt_events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.utils.time import time_now_iso8601

from audio_arena.pipelines.realtime import RealtimePipeline

WS_URL_ENV = "NEMOTRON_VOICECHAT_WS_URL"
SERVER_SAMPLE_RATE_HZ = 24000
# Must stay below pipecat's BOT_VAD_STOP_SECS (2.0s in RealtimePipeline).
UTTERANCE_END_SILENCE_S = 1.0
TOOL_RESULT_TIMEOUT_S = 10.0
ASCII_REPLACEMENTS = {
    "\u2013": "-",
    "\u2014": " - ",
    "\u2192": "->",
    "\u2022": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
}


def to_ascii(text: str) -> str:
    """Map typographic punctuation to ASCII and strip remaining accents."""
    for src, dst in ASCII_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


class NemotronVoiceChatLLMService(OpenAIRealtimeLLMService):
    """OpenAI Realtime service adapted to the VoiceChat server protocol."""

    def __init__(self, await_tool_result: Callable[[str], Awaitable[dict[str, Any]]], **kwargs):
        # The parent appends ?model= to base_url; the VoiceChat server serves one model.
        base_url = kwargs["base_url"]
        super().__init__(**kwargs)
        self.base_url = base_url
        self._await_tool_result = await_tool_result
        self._utterance_active = False
        self._last_voiced_audio_at: Optional[float] = None

    async def send_client_event(self, event):
        if event.type == "session.update":
            event.session.instructions = to_ascii(event.session.instructions or "")
            event.session.tools = json.loads(to_ascii(json.dumps(event.session.tools or [])))
        await super().send_client_event(event)

    async def _receive_task_handler(self):
        async for message in self._websocket:
            evt = json.loads(message)
            kind = evt["type"]
            if kind == "session.created":
                await self._update_settings()
            elif kind == "session.updated":
                await self._handle_evt_session_updated(evt)
            elif kind == "response.output_audio.delta":
                await self._handle_audio_chunk(base64.b64decode(evt["delta"]))
            elif kind == "response.output_audio_transcript.delta":
                await self._handle_transcript_delta(evt["delta"])
            elif kind == "response.output_audio_transcript.done":
                await self._end_utterance()
            elif kind == "conversation.item.input_audio_transcription.completed":
                await self.push_frame(
                    TranscriptionFrame(evt["transcript"], "", time_now_iso8601()),
                    FrameDirection.UPSTREAM,
                )
            elif kind == "input_audio_buffer.speech_started":
                await self.broadcast_frame(UserStartedSpeakingFrame)
            elif kind == "input_audio_buffer.speech_stopped":
                await self.broadcast_frame(UserStoppedSpeakingFrame)
            elif kind == "response.function_call_arguments.done":
                await self._handle_function_call(evt)
            elif kind == "error":
                error = evt["error"]
                await self.push_error(
                    error_msg=f"VoiceChat server error {error['code']}: {error['message']}", fatal=True
                )
                return
        if not self._disconnecting:
            # Without this the harness records every remaining turn as an empty response.
            await self.push_error(error_msg="VoiceChat server connection closed mid-session", fatal=True)

    async def _handle_audio_chunk(self, pcm: bytes):
        now = time.monotonic()
        if not np.frombuffer(pcm, dtype=np.int16).any():
            if self._last_voiced_audio_at is not None and now - self._last_voiced_audio_at >= UTTERANCE_END_SILENCE_S:
                await self._end_utterance()
            return
        self._last_voiced_audio_at = now
        await self.stop_ttfb_metrics()
        await self.push_frame(TTSAudioRawFrame(audio=pcm, sample_rate=SERVER_SAMPLE_RATE_HZ, num_channels=1))

    async def _handle_transcript_delta(self, text: str):
        if not self._utterance_active:
            self._utterance_active = True
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(TTSStartedFrame())
        await self._push_output_transcript_text_frames(text)

    async def _end_utterance(self):
        self._last_voiced_audio_at = None
        # Late transcript.done after a silence close, or an empty utterance (no deltas).
        if not self._utterance_active:
            return
        self._utterance_active = False
        await self.push_frame(LLMFullResponseEndFrame())
        await self.push_frame(TTSStoppedFrame())

    async def _handle_function_call(self, evt: dict):
        call_id = evt["call_id"]
        # Mark complete up front so the parent never auto-sends the result plus
        # response.create; the result is sent below in the server's expected shape.
        self._completed_tool_calls.add(call_id)
        await self.run_function_calls(
            [
                FunctionCallFromLLM(
                    context=self._context,
                    tool_call_id=call_id,
                    function_name=evt["name"],
                    arguments=json.loads(evt["arguments"]),
                )
            ]
        )
        output = to_ascii(json.dumps(await self._await_tool_result(call_id)))
        await self.send_client_event(
            rt_events.ConversationItemCreateEvent(
                item=rt_events.ConversationItem(type="function_call_output", call_id=call_id, output=output)
            )
        )
        logger.info(f"[VoiceChat] Sent function_call_output for {evt['name']} (call_id={call_id})")


class NemotronVoiceChatPipeline(RealtimePipeline):
    """RealtimePipeline wired to a self-hosted VoiceChat server."""

    requires_service = False

    def __init__(self, benchmark):
        super().__init__(benchmark)
        self._tool_results: dict[str, asyncio.Future] = {}

    def _create_llm(self, service_class: Optional[type], model: str) -> FrameProcessor:
        base_url = os.getenv(WS_URL_ENV)
        if not base_url:
            raise EnvironmentError(f"{WS_URL_ENV} is required (e.g. ws://localhost:9000/v1/realtime via SSH tunnel)")
        logger.info(f"Using NemotronLabs VoiceChat server at {base_url}")

        system_instruction = getattr(self.benchmark, "system_instruction", "")
        session_props = rt_events.SessionProperties(
            instructions=system_instruction,
            tools=getattr(self.benchmark, "tools_schema", None),
            audio=rt_events.AudioConfiguration(
                input=rt_events.AudioInput(
                    format=rt_events.PCMAudioFormat(rate=SERVER_SAMPLE_RATE_HZ),
                ),
                output=rt_events.AudioOutput(
                    format=rt_events.PCMAudioFormat(rate=SERVER_SAMPLE_RATE_HZ),
                ),
            ),
        )
        return NemotronVoiceChatLLMService(
            api_key="unused",
            model=model,
            base_url=base_url,
            system_instruction=system_instruction,
            session_properties=session_props,
            await_tool_result=self._await_tool_result,
        )

    async def _function_catchall(self, params: FunctionCallParams) -> None:
        await super()._function_catchall(params)
        self._tool_result_future(params.tool_call_id).set_result(self._last_tool_result)

    async def _await_tool_result(self, call_id: str) -> dict[str, Any]:
        """Scripted result for one call; pipecat runs tool handlers in a background task."""
        result = await asyncio.wait_for(self._tool_result_future(call_id), TOOL_RESULT_TIMEOUT_S)
        del self._tool_results[call_id]
        return result

    def _tool_result_future(self, call_id: str) -> asyncio.Future:
        return self._tool_results.setdefault(call_id, asyncio.get_running_loop().create_future())
