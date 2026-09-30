import numpy as np
import pytest

from audio_arena.pipelines.personaplex import (
    TOOLS_SECTION_MARKER,
    build_persona_prompt,
    find_turn_end,
    voiced_frame_starts,
)
from benchmarks.appointment_bench.system import system_instruction as appointment_instruction
from benchmarks.conversation_bench.system import system_instruction as conversation_instruction


def frames(start_s: float, end_s: float) -> list[float]:
    return [round(t, 1) for t in np.arange(start_s, end_s, 0.1)]


def turn_end(voiced: list[float], turn_start_s: float = 0.0):
    """Caller audio ends at 11 s; the probe and no-response deadline end at 41 s."""
    return find_turn_end(voiced, turn_start_s, caller_end_s=11.0, start_deadline_s=41.0, probe_end_s=41.0)


def test_reply_after_caller_advances_after_stop_silence_and_drain():
    voiced = frames(12.0, 15.0)
    assert turn_end(voiced) == pytest.approx(17.5)


def test_mid_utterance_barge_in_waits_for_caller_audio():
    voiced = frames(6.0, 7.0)
    assert turn_end(voiced) == 11.0


def test_pause_shorter_than_stop_silence_keeps_reply_open():
    voiced = frames(12.0, 13.0) + frames(14.0, 15.0)
    assert turn_end(voiced) == pytest.approx(17.5)


def test_no_reply_gives_up_at_deadline():
    assert turn_end([]) == 41.0


def test_speech_before_turn_start_is_ignored():
    voiced = frames(2.0, 3.0)
    assert turn_end(voiced, turn_start_s=5.0) == 41.0


def test_reply_running_past_probe_needs_more_audio():
    voiced = frames(30.0, 41.0)
    assert turn_end(voiced) is None


def test_voiced_frame_starts_detects_tone():
    sample_rate = 24000
    audio = np.zeros(sample_rate, dtype=np.float32)
    audio[sample_rate // 2 :] = 0.5 * np.sin(np.arange(sample_rate // 2) * 0.1)
    assert voiced_frame_starts(audio, sample_rate) == pytest.approx(frames(0.5, 1.0))


def test_persona_prompt_drops_tools_section():
    prompt = build_persona_prompt(appointment_instruction)
    assert TOOLS_SECTION_MARKER not in prompt
    assert "end_session_function" not in prompt
    assert appointment_instruction.lstrip().startswith(prompt)


def test_persona_prompt_rejects_prompts_longer_than_context():
    with pytest.raises(ValueError, match="cannot be run faithfully"):
        build_persona_prompt(conversation_instruction)
