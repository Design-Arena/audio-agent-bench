#!/usr/bin/env bash
# Serve NVIDIA NemotronLabs VoiceChat with serving limits large enough for Audio Arena prompts.
#
# Run on a GPU host (>= 80 GB VRAM) that is logged in to nvcr.io and has the model
# downloaded (see docs/nemotron-voicechat.md). The server binds to 127.0.0.1:9000;
# reach it over an SSH tunnel.
#
# NVIDIA hardcodes the limits below in the container. This script copies the three
# affected files out of the image and makes each limit read from an env var whose
# default is NVIDIA's shipped value, so model weights and all other code are untouched.
set -euo pipefail

IMAGE="nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest"
CONTAINER="nemotron-labs-voicechat"
MODEL_DIR="${MODEL_DIR:-$HOME/nemotron-labs-voicechat_v1.0.0}"
PATCH_DIR="${PATCH_DIR:-$HOME/vc_patch}"
BACKEND="/opt/tritonserver/backends/nemotron-voicechat"

# Stock values: 6144 / 6144 / 5000 steps (400 s) / 300 s / 32000 chars.
# The system prompt is prefilled twice, so conversation_bench needs ~30k LLM positions
# plus one position per 80 ms step of session audio.
VC_LLM_MAX_MODEL_LEN="${VC_LLM_MAX_MODEL_LEN:-81920}"
VC_TTS_MAX_MODEL_LEN="${VC_TTS_MAX_MODEL_LEN:-40960}"
VC_STEP_BUFFER="${VC_STEP_BUFFER:-40000}"
MAX_SESSION_DURATION_S="${MAX_SESSION_DURATION_S:-3600}"
MAX_INSTRUCTIONS_LENGTH="${MAX_INSTRUCTIONS_LENGTH:-80000}"

mkdir -p "$PATCH_DIR"
extractor=$(docker create "$IMAGE")
docker cp "$extractor:$BACKEND/checkpoint_utils/load_utils.py" "$PATCH_DIR/load_utils.py"
docker cp "$extractor:$BACKEND/sequence_manager.py" "$PATCH_DIR/sequence_manager.py"
docker cp "$extractor:/s2s/audio_server.py" "$PATCH_DIR/audio_server.py"
docker rm "$extractor" > /dev/null

sed -i \
  -e 's/max_model_len=6144,/max_model_len=int(os.environ.get("VC_LLM_MAX_MODEL_LEN", "6144")),/' \
  -e 's/"max_model_len": 6144,/"max_model_len": int(os.environ.get("VC_TTS_MAX_MODEL_LEN", "6144")),/' \
  "$PATCH_DIR/load_utils.py"
sed -i \
  -e '1i import os\nVC_STEP_BUFFER = int(os.environ.get("VC_STEP_BUFFER", "5000"))' \
  -e 's/torch.empty(1, 5000,/torch.empty(1, VC_STEP_BUFFER,/' \
  -e 's/^\( *\)1, 5000, self._num_audio_quantizers,/\11, VC_STEP_BUFFER, self._num_audio_quantizers,/' \
  "$PATCH_DIR/sequence_manager.py"
sed -i \
  -e 's/^MAX_INSTRUCTIONS_LENGTH = 32000/MAX_INSTRUCTIONS_LENGTH = int(os.environ.get("MAX_INSTRUCTIONS_LENGTH", "32000"))/' \
  "$PATCH_DIR/audio_server.py"

# Fail loudly if NVIDIA changed the source and a substitution silently missed.
[ "$(grep -c 'VC_LLM_MAX_MODEL_LEN\|VC_TTS_MAX_MODEL_LEN' "$PATCH_DIR/load_utils.py")" -eq 2 ]
[ "$(grep -c 'VC_STEP_BUFFER' "$PATCH_DIR/sequence_manager.py")" -eq 5 ]
grep -q 'os.environ.get("MAX_INSTRUCTIONS_LENGTH"' "$PATCH_DIR/audio_server.py"

docker rm -f "$CONTAINER" > /dev/null 2>&1 || true
docker run -d --name="$CONTAINER" --runtime=nvidia --gpus '"device=0"' --shm-size=8GB \
  -e NIM_HTTP_API_PORT=9000 -p 127.0.0.1:9000:9000 \
  -e VC_LLM_MAX_MODEL_LEN="$VC_LLM_MAX_MODEL_LEN" \
  -e VC_TTS_MAX_MODEL_LEN="$VC_TTS_MAX_MODEL_LEN" \
  -e VC_STEP_BUFFER="$VC_STEP_BUFFER" \
  -e MAX_SESSION_DURATION="$MAX_SESSION_DURATION_S" \
  -e MAX_INSTRUCTIONS_LENGTH="$MAX_INSTRUCTIONS_LENGTH" \
  -v "$MODEL_DIR:/data/models" \
  -v "$PATCH_DIR/load_utils.py:$BACKEND/checkpoint_utils/load_utils.py:ro" \
  -v "$PATCH_DIR/sequence_manager.py:$BACKEND/sequence_manager.py:ro" \
  -v "$PATCH_DIR/audio_server.py:/s2s/audio_server.py:ro" \
  --entrypoint /s2s/run_s2s_server.sh "$IMAGE"

echo "Waiting for server (model load takes ~5 min)..."
until curl -sf localhost:9000/v1/realtime/health | grep -q '"triton_status":"ready"'; do sleep 10; done
echo "SERVER_READY"
