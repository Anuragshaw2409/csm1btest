# CSM-1B streaming TTS server, containerized.
#
# Runs identically with or without a GPU: PyPI's torch==2.4.0 wheel bundles
# CUDA and uses it automatically when a GPU + nvidia-container-toolkit are
# present (`docker run --gpus all ...`), and transparently falls back to CPU
# otherwise (e.g. plain `docker run` on a Mac). No separate CPU/GPU image.
#
# Pinned to linux/amd64 (see FROM below) rather than building natively per
# host: moshi (a dependency) unconditionally requires bitsandbytes on any
# Linux platform, and bitsandbytes ships no linux/arm64 wheel in the pinned
# range -- so a native arm64 build (e.g. on Apple Silicon) fails outright.
# Real GPU servers are essentially always x86_64 anyway, so standardizing on
# amd64 for both Mac and server keeps one working image instead of two
# slightly-different ones. Docker Desktop on Apple Silicon transparently
# emulates amd64 via QEMU, so this still runs fine on a Mac, just slower
# than a native arm64 image would be (irrelevant on the GPU server).
#
# Build (same command on Mac or server):
#   docker build -t csm-tts-server .
#
# Run (Mac / no GPU):
#   docker run --rm -p 8000:8000 -e HF_TOKEN=hf_xxx \
#     -v csm-hf-cache:/root/.cache/huggingface \
#     csm-tts-server
#
# Run (GPU server, needs nvidia-container-toolkit installed on the host):
#   docker run --rm --gpus all -p 8000:8000 -e HF_TOKEN=hf_xxx \
#     -v csm-hf-cache:/root/.cache/huggingface \
#     csm-tts-server
#
# The volume mount caches the ~5-6GB of downloaded weights across container
# restarts -- omit it and every `docker run` re-downloads them.
#
# See server/README.md for the WebSocket wire protocol this serves on
# ws://<host>:8000/v1/tts/stream (or wss:// -- see the TLS section below).

FROM --platform=linux/amd64 python:3.11-slim

# ffmpeg: some torchaudio codecs need it. git: pip needs it to install
# silentcipher from GitHub (see requirements.txt). build-essential: a few
# deps (e.g. sentencepiece via transformers) build from source on slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi "uvicorn[standard]" websockets \
    # moshi's quantize.linear() unconditionally imports bitsandbytes even
    # when nothing is actually quantized -- same fix used by setup_livekit.sh.
    && pip install --no-cache-dir bitsandbytes \
    # fastapi/uvicorn/bitsandbytes can silently pull a newer huggingface_hub
    # than requirements.txt's ==0.28.1 pin; moshi/transformers/tokenizers
    # need that specific range, so re-pin it last (same fix used by
    # setup_server.sh / setup_livekit.sh).
    && pip install --no-cache-dir "huggingface_hub==0.28.1"

# Only what's needed to run CSM-1B + the TTS server -- no LiveKit code.
COPY generator.py models.py watermarking.py ./
COPY server/ ./server/

ENV NO_TORCH_COMPILE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

# CSM_TTS_TLS_CERTFILE / CSM_TTS_TLS_KEYFILE: unset by default (plain ws://).
# To serve wss://, mount a cert/key pair (e.g. the repo's own .certs/, or a
# real one) and point these at the mounted paths, e.g.:
#   docker run -v $PWD/.certs:/certs:ro \
#     -e CSM_TTS_TLS_CERTFILE=/certs/cert.pem -e CSM_TTS_TLS_KEYFILE=/certs/key.pem ...
ENV CSM_TTS_TLS_CERTFILE="" \
    CSM_TTS_TLS_KEYFILE=""

EXPOSE 8000

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]
