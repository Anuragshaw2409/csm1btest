#!/usr/bin/env bash
#
# One-shot setup + launch script for running the CSM-1B streaming TTS
# inference server (server/tts_server.py) on a remote GPU server
# (Ubuntu/Debian assumed). Idempotent: safe to re-run if it fails partway
# through, it skips steps already done. Modeled on setup_server.sh, which
# does the same for the Gradio webui.py demo -- this one launches the
# WebSocket TTS server instead, for a LiveKit agent (or any other client)
# running elsewhere to call remotely.
#
# What it does:
#   1. Checks for an NVIDIA GPU (warns if none found, continues on CPU).
#   2. Installs Python 3.11, ffmpeg, git if missing (apt).
#   3. Clones this repo (if not already present) and creates a venv.
#   4. Installs all Python deps (torch/CSM stack + fastapi/uvicorn/websockets).
#   5. Logs into Hugging Face (needed for the gated sesame/csm-1b repo).
#   6. Pre-downloads CSM-1B so the first request isn't slow.
#   7. Launches uvicorn serving server/tts_server.py over wss://, using the
#      self-signed cert already checked into .certs/.
#
# Usage:
#   ./setup_tts_server.sh                       # full setup + launch
#   ./setup_tts_server.sh --skip-setup          # just (re)launch the server
#   ./setup_tts_server.sh --no-tls              # plain ws:// (e.g. local testing)
#   HF_TOKEN=hf_xxx ./setup_tts_server.sh       # non-interactive HF login
#
set -euo pipefail

REPO_URL="https://github.com/SesameAILabs/csm.git"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
PYTHON_BIN="python3.11"
TTS_SERVER_PORT="${TTS_SERVER_PORT:-8000}"
TTS_SERVER_HOST="${TTS_SERVER_HOST:-0.0.0.0}"
SKIP_SETUP=0
TLS_ARGS=(--ssl-keyfile "$REPO_DIR/.certs/key.pem" --ssl-certfile "$REPO_DIR/.certs/cert.pem")

for arg in "$@"; do
  case "$arg" in
    --skip-setup) SKIP_SETUP=1 ;;
    --no-tls) TLS_ARGS=() ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }

if [ "$SKIP_SETUP" -eq 1 ]; then
  log "Skipping setup, launching TTS server directly"
  source "$VENV_DIR/bin/activate"
  cd "$REPO_DIR"
  exec uvicorn server.tts_server:app --host "$TTS_SERVER_HOST" --port "$TTS_SERVER_PORT" "${TLS_ARGS[@]}"
fi

log "Checking for NVIDIA GPU"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found or failed. This will run on CPU and be much slower." >&2
fi

log "Installing system packages (python3.11, ffmpeg, git, build tools)"
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -qq
  if ! command -v python3.11 >/dev/null 2>&1; then
    sudo apt-get install -y -qq software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
  fi
  sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev ffmpeg git curl build-essential
else
  echo "Non-Debian system detected -- install python3.11, ffmpeg, git, curl yourself, then re-run." >&2
  command -v python3.11 >/dev/null 2>&1 || { echo "python3.11 not found, aborting." >&2; exit 1; }
fi

if [ ! -f "$REPO_DIR/generator.py" ]; then
  log "Cloning CSM repo into $REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

log "Creating virtualenv at $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip -q

log "Installing Python dependencies (this can take a few minutes)"
pip install -q -r requirements.txt
pip install -q fastapi "uvicorn[standard]" websockets

if ! pip check >/dev/null 2>&1; then
  echo "WARNING: pip reports dependency conflicts, see below (may still work):" >&2
  pip check || true
fi

# Point CSM's tokenizer loader at an ungated Llama-3.2-1B mirror so setup
# doesn't block on Meta's gated-repo approval (only the tokenizer is needed,
# not the weights -- sesame/csm-1b ships its own weights).
if grep -q 'meta-llama/Llama-3.2-1B' generator.py; then
  log "Patching generator.py to use an ungated Llama-3.2-1B tokenizer mirror"
  sed -i.bak 's#meta-llama/Llama-3.2-1B#unsloth/Llama-3.2-1B#' generator.py
fi

log "Hugging Face login"
if ! python -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
  if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN"
  else
    echo "Enter your Hugging Face token (needs access to sesame/csm-1b, input hidden):"
    huggingface-cli login
  fi
else
  echo "Already logged in as: $(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")"
fi

log "Pre-downloading CSM-1B weights + tokenizer (this is the big one, ~a few GB)"
python - <<'PYEOF'
from generator import load_csm_1b
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
load_csm_1b(device=device)
print("CSM-1B cached.")
PYEOF

log "Setup complete. Launching TTS server on port $TTS_SERVER_PORT..."
exec uvicorn server.tts_server:app --host "$TTS_SERVER_HOST" --port "$TTS_SERVER_PORT" "${TLS_ARGS[@]}"
