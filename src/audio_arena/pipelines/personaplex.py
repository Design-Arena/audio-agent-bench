"""NVIDIA PersonaPlex via fal (offline full-duplex speech-to-speech).

WORK IN PROGRESS, NOT USABLE YET. fal's endpoint takes a complete caller-audio file and
returns the model's time-aligned audio track plus its text. This pipeline replays a live
call: after each caller turn, find where the model stops talking (same rules as
RealtimePipeline), cut the call there, append the next caller turn, and resend. That
assumes a seeded request reproduces earlier output for the shared prefix, which fal does
not guarantee across workers, so every run fails the prefix-stability assertion.

Differences from live pipelines:
- No tool calling, so the prompt is the benchmark system instruction minus its tools section.
- Prompts longer than the model's context (conversation_bench) are rejected.
- No turn-taking tags are produced, so the judge skips the turn-taking dimension.

Usage:
    export FAL_KEY=...
    uv run audio-arena run appointment_bench --model personaplex
"""

import base64
import io
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import soundfile as sf
from loguru import logger

from audio_arena.recording.transcript_recorder import TranscriptRecorder

FAL_URL = "https://fal.run/fal-ai/personaplex"
FAL_KEY_ENV = "FAL_KEY"
VOICE = "NATF2"
# Fixed seed is what makes prefix replay reproduce earlier output.
SEED = 0
REQUEST_TIMEOUT_S = 1800.0
MAX_REQUEST_ATTEMPTS = 3

TOOLS_SECTION_MARKER = "---\n### **AVAILABLE TOOLS**"
# Measured on fal: 10.2k-char prompts stay coherent, 13.1k-char prompts produce garbage.
PROMPT_MAX_CHARS = 11000

# Turn timing mirrors RealtimePipeline.
GREETING_START_WINDOW_S = 8.0
BOT_STOP_SILENCE_S = 2.0
TURN_DRAIN_S = 0.5
NO_RESPONSE_TIMEOUT_S = 30.0
LISTEN_EXTENSION_S = 30.0
MAX_LISTEN_S = 120.0

VOICED_FRAME_S = 0.1
VOICED_RMS_THRESHOLD = 0.01


@dataclass
class Generation:
    audio: np.ndarray
    sample_rate: int
    text: str


def build_persona_prompt(system_instruction: str) -> str:
    """Benchmark system instruction without the tools section PersonaPlex cannot use."""
    assert system_instruction.count(TOOLS_SECTION_MARKER) == 1, "tools section marker not found exactly once"
    prompt = system_instruction.split(TOOLS_SECTION_MARKER)[0].strip()
    if len(prompt) > PROMPT_MAX_CHARS:
        raise ValueError(
            f"Prompt is {len(prompt)} chars; PersonaPlex's context fits ~{PROMPT_MAX_CHARS}. "
            "This benchmark cannot be run faithfully on PersonaPlex."
        )
    return prompt


def voiced_frame_starts(audio: np.ndarray, sample_rate: int) -> List[float]:
    frame = int(VOICED_FRAME_S * sample_rate)
    frames = audio[: len(audio) // frame * frame].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return [i * VOICED_FRAME_S for i in np.flatnonzero(rms > VOICED_RMS_THRESHOLD)]


def find_turn_end(
    voiced_s: List[float], turn_start_s: float, caller_end_s: float, start_deadline_s: float, probe_end_s: float
) -> Optional[float]:
    """When the harness would queue the next turn, or None if the probe is too short to tell.

    Like RealtimePipeline: the model's reply ends after BOT_STOP_SILENCE_S of silence, the
    turn advances after TURN_DRAIN_S but never before the caller's audio has finished, and
    with no reply the harness gives up at start_deadline_s.
    """
    reply = [t for t in voiced_s if t >= turn_start_s]
    if not reply or reply[0] >= start_deadline_s:
        return max(caller_end_s, start_deadline_s)
    for current, following in zip(reply, reply[1:] + [math.inf]):
        if following - current - VOICED_FRAME_S < BOT_STOP_SILENCE_S:
            continue
        advance_s = current + VOICED_FRAME_S + BOT_STOP_SILENCE_S + TURN_DRAIN_S
        return None if advance_s > probe_end_s else max(caller_end_s, advance_s)
    raise AssertionError("unreachable: the last voiced frame is always followed by infinite silence")


def encode_wav_data_uri(audio: np.ndarray, sample_rate: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


class PersonaPlexPipeline:
    """Offline replay runner with the same run() contract the CLI uses for pipecat pipelines."""

    requires_service = False

    def __init__(self, benchmark):
        self.benchmark = benchmark

    async def run(
        self,
        recorder: TranscriptRecorder,
        model: str,
        service_class: Optional[type] = None,
        service_name: Optional[str] = None,
        turn_indices: Optional[List[int]] = None,
        rehydration_turns: Optional[List[Dict[str, Any]]] = None,
        disable_vad: bool = False,
    ) -> None:
        if rehydration_turns is not None:
            raise NotImplementedError("PersonaPlex has no text-history input, so --rehydrate is unsupported")
        api_key = os.getenv(FAL_KEY_ENV)
        if not api_key:
            raise EnvironmentError(f"{FAL_KEY_ENV} must be set for PersonaPlex")
        self.recorder = recorder
        self._turn_indices = turn_indices or list(range(len(self.benchmark.turns)))
        self._prompt = build_persona_prompt(self.benchmark.system_instruction)

        headers = {"Authorization": f"Key {api_key}"}
        async with httpx.AsyncClient(headers=headers, timeout=REQUEST_TIMEOUT_S) as client:
            self._client = client
            await self._replay_call()

    async def _replay_call(self) -> None:
        turn_audio = [self._load_turn_audio(turn_index) for turn_index in self._turn_indices]
        self._sample_rate = turn_audio[0][1]
        assert all(sr == self._sample_rate for _, sr in turn_audio), "turn audio sample rates differ"

        call = np.zeros(0, dtype=np.float32)
        call, _ = await self._advance(
            call, turn_start_s=0.0, caller_end_s=0.0, start_deadline_s=GREETING_START_WINDOW_S
        )
        latest = await self._generate(call)

        for turn_index, (audio, _) in zip(self._turn_indices, turn_audio):
            self.recorder.start_turn(turn_index)
            turn_start_s = len(call) / self._sample_rate
            call = np.concatenate([call, audio])
            caller_end_s = len(call) / self._sample_rate
            call, first_voiced_s = await self._advance(
                call, turn_start_s, caller_end_s, start_deadline_s=caller_end_s + NO_RESPONSE_TIMEOUT_S
            )
            previous_text = latest.text
            latest = await self._generate(call)
            assert latest.text.startswith(previous_text), "PersonaPlex output was not prefix-stable; replay is invalid"
            if first_voiced_s is not None:
                self.recorder.record_ttfb(max(0.0, first_voiced_s - caller_end_s))
            self.recorder.write_turn(
                user_text=self.benchmark.turns[turn_index]["input"],
                assistant_text=latest.text[len(previous_text):].strip(),
            )

        sf.write(self.recorder.run_dir / "caller.wav", call, self._sample_rate)
        sf.write(self.recorder.run_dir / "personaplex_output.wav", latest.audio, latest.sample_rate)
        self._write_runtime_notes()

    async def _advance(
        self, call: np.ndarray, turn_start_s: float, caller_end_s: float, start_deadline_s: float
    ) -> tuple[np.ndarray, Optional[float]]:
        """Extend the call with silence until the harness would move on; return it and the reply start."""
        listen_s = max(NO_RESPONSE_TIMEOUT_S, start_deadline_s - caller_end_s)
        while True:
            probe = np.concatenate([call, np.zeros(int(listen_s * self._sample_rate), dtype=np.float32)])
            probe_end_s = len(probe) / self._sample_rate
            generation = await self._generate(probe)
            voiced_s = voiced_frame_starts(generation.audio, generation.sample_rate)
            turn_end_s = find_turn_end(voiced_s, turn_start_s, caller_end_s, start_deadline_s, probe_end_s)
            if turn_end_s is None and listen_s + LISTEN_EXTENSION_S <= MAX_LISTEN_S:
                listen_s += LISTEN_EXTENSION_S
                continue
            if turn_end_s is None:
                logger.warning(f"[PersonaPlex] Reply still running after {MAX_LISTEN_S}s of listening; cutting turn")
                turn_end_s = probe_end_s
            first_voiced_s = next((t for t in voiced_s if turn_start_s <= t < turn_end_s), None)
            return probe[: int(turn_end_s * self._sample_rate)], first_voiced_s

    async def _generate(self, call: np.ndarray) -> Generation:
        body = {
            "audio_url": encode_wav_data_uri(call, self._sample_rate),
            "prompt": self._prompt,
            "voice": VOICE,
            "seed": SEED,
        }
        for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
            try:
                response = await self._client.post(FAL_URL, json=body)
                response.raise_for_status()
                result = response.json()
                wav = (await self._client.get(result["audio"]["url"])).raise_for_status().content
                break
            except httpx.HTTPError as e:
                if attempt == MAX_REQUEST_ATTEMPTS:
                    raise
                logger.warning(f"[PersonaPlex] fal request failed (attempt {attempt}/{MAX_REQUEST_ATTEMPTS}): {e}")
        audio, sample_rate = sf.read(io.BytesIO(wav), dtype="float32")
        logger.info(f"[PersonaPlex] Generated {len(call) / self._sample_rate:.1f}s call")
        return Generation(audio=audio, sample_rate=sample_rate, text=result["text"])

    def _load_turn_audio(self, turn_index: int) -> tuple[np.ndarray, int]:
        path = self.benchmark.get_audio_path(turn_index)
        audio, sample_rate = sf.read(path, dtype="float32")
        assert audio.ndim == 1, f"expected mono turn audio: {path}"
        return audio, sample_rate

    def _write_runtime_notes(self) -> None:
        self.recorder.write_summary()
        runtime_path = self.recorder.run_dir / "runtime.json"
        runtime = json.loads(runtime_path.read_text())
        runtime["mode"] = "offline_prefix_replay"
        runtime["prompt"] = "benchmark system instruction without the tools section (PersonaPlex has no tools)"
        runtime["turn_taking_skip_reason"] = "Offline replay has no harness turn tags"
        runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
