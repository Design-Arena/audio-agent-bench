# NVIDIA NemotronLabs VoiceChat

[NemotronLabs VoiceChat 11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
is a full-duplex speech-to-speech model with tool calling. There is no hosted API,
so it runs from NVIDIA's NIM container, which exposes an OpenAI Realtime-style
WebSocket. The `nemotron-voicechat` pipeline talks to that WebSocket.

## Serving limits (read before comparing scores)

The container ships with limits that no Audio Arena benchmark fits in:

| Limit | Stock | Why it breaks |
|---|---|---|
| LLM / TTS `max_model_len` | 6144 | The server prefills the system prompt **twice**. Benchmark prompts are 3.4k-14.8k tokens, i.e. 7k-30k positions before any audio. |
| Step buffer | 5000 steps (400 s) | Every 80 ms of session audio is one step; long benchmarks run 10-25 min. |
| `MAX_SESSION_DURATION` | 300 s | Same as above. |
| `MAX_INSTRUCTIONS_LENGTH` | 32000 chars | `conversation_bench` has a 53k-char prompt. |

`scripts/serve_nemotron_voicechat.sh` raises these by making each constant read from
an env var. Weights and model code are unchanged, but the model runs far past the
context length it was served at, so report VoiceChat scores with that caveat.

## Setup

1. GPU host with >= 80 GB VRAM (tested: AWS `g7e.4xlarge`, RTX PRO 6000 Blackwell 96 GB).
2. NGC access: `docker login nvcr.io` (user `$oauthtoken`, password = NGC API key).
3. Download the model with the NGC CLI:
   `ngc registry model download-version "nim/nvidia/nemotron-labs-voicechat:1.0.0" --dest ~`
4. On the GPU host: `bash scripts/serve_nemotron_voicechat.sh` (prints `SERVER_READY`).
5. From the machine running the benchmark:

```bash
ssh -f -N -L 9000:localhost:9000 ubuntu@<gpu-host>
export NEMOTRON_VOICECHAT_WS_URL=ws://localhost:9000/v1/realtime
uv run audio-arena run appointment_bench --model nemotron-voicechat
```

## Running

- **Run one session at a time.** At six concurrent sessions the server fell below
  real time and the container's vLLM scheduler crashed
  (`AttributeError: 'list' object has no attribute 'add'`), which kills every session.
- The model decides turn-taking itself; there is no VAD or silence setting. It can start
  replying during a pause in the caller's utterance, which the harness scores like any
  other realtime model's barge-in.
- Scores vary between runs: in some runs it calls `end_session` early, which ends the
  run under the standard harness rules. Repeat runs before drawing conclusions.
