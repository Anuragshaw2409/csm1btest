#!/usr/bin/env bash
#
# One-shot setup for the LiveKit-based local voice pipeline:
#   Silero VAD + LiveKit turn-detector -> faster-whisper STT -> Ollama LLM (llama3.2) -> CSM-1B TTS
# orchestrated by LiveKit Agents, with a React frontend + token-server for
# connecting over a real LiveKit room (LiveKit Cloud or self-hosted).
#
# What it does:
#   1. Installs Python 3.11, ffmpeg, git, Node.js if missing (apt; Ubuntu/Debian assumed).
#   2. Creates a venv in livekit/.venv and installs the agent's dependencies
#      (livekit-agents + local STT/LLM/TTS stack, reusing ../requirements.txt).
#   3. Installs Ollama and pulls the local LLM model.
#   4. Downloads Silero VAD / turn-detector model files and pre-warms CSM-1B /
#      Whisper caches.
#   5. npm installs the frontend + token-server, builds the frontend.
#
# Usage:
#   ./setup_livekit.sh                  # full setup
#   ./setup_livekit.sh --skip-setup     # skip installs, just verify + print run instructions
#   OLLAMA_MODEL=llama3.2:3b ./setup_livekit.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSM_ROOT="$(dirname "$ROOT_DIR")"
AGENT_DIR="$ROOT_DIR/agent"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="python3.11"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b}"
SKIP_SETUP=0

for arg in "$@"; do
  case "$arg" in
    --skip-setup) SKIP_SETUP=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }

if [ "$SKIP_SETUP" -eq 0 ]; then
  log "Checking for NVIDIA GPU"
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  else
    echo "WARNING: nvidia-smi not found or failed. This will run on CPU and be much slower." >&2
  fi

  log "Installing system packages (python3.11, ffmpeg, git, node, build tools)"
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update -qq
    if ! command -v python3.11 >/dev/null 2>&1; then
      sudo apt-get install -y -qq software-properties-common
      sudo add-apt-repository -y ppa:deadsnakes/ppa
      sudo apt-get update -qq
    fi
    sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev ffmpeg git curl build-essential
    if ! command -v node >/dev/null 2>&1; then
      curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
      sudo apt-get install -y -qq nodejs
    fi
  else
    echo "Non-Debian system detected -- install python3.11, ffmpeg, git, node yourself, then re-run." >&2
    command -v python3.11 >/dev/null 2>&1 || { echo "python3.11 not found, aborting." >&2; exit 1; }
  fi

  log "Creating virtualenv at $VENV_DIR"
  if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip -q

  log "Installing CSM-1B stack (shared with ../requirements.txt)"
  pip install -q -r "$CSM_ROOT/requirements.txt"

  log "Installing the agent (livekit-agents + local STT/LLM/TTS plugins)"
  pip install -q -e "$AGENT_DIR"

  # moshi's quantize.linear() unconditionally imports bitsandbytes even when
  # not actually quantizing -- same fix as the main csm-main project.
  pip install -q bitsandbytes

  log "Installing Ollama"
  if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  if ! pgrep -x ollama >/dev/null 2>&1; then
    nohup ollama serve > "$ROOT_DIR/ollama.log" 2>&1 &
    sleep 3
  fi
  log "Pulling Ollama model: $OLLAMA_MODEL"
  ollama pull "$OLLAMA_MODEL"

  log "Downloading Silero VAD + LiveKit turn-detector model files"
  (cd "$AGENT_DIR" && python -m livekit.agents download-files)

  log "Pre-warming CSM-1B / faster-whisper model caches"
  python - <<PYEOF
import sys
sys.path.insert(0, "$CSM_ROOT")
from generator import load_csm_1b
from faster_whisper import WhisperModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Downloading CSM-1B weights + voice prompts...")
load_csm_1b(device=device)
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="sesame/csm-1b", filename="prompts/conversational_a.wav")
hf_hub_download(repo_id="sesame/csm-1b", filename="prompts/conversational_b.wav")

print("Downloading faster-whisper base.en...")
WhisperModel("base.en", device=device, compute_type="float16" if device == "cuda" else "int8")
print("Done.")
PYEOF

  log "Installing + building frontend"
  (cd "$ROOT_DIR/frontend" && npm install --silent && npm run build --silent)

  log "Installing token-server"
  (cd "$ROOT_DIR/token-server" && npm install --silent)
fi

log "Setup complete."
cat <<EOF

Run these in three separate terminals:

  1) Voice agent (connects to LiveKit, runs the local STT/LLM/TTS pipeline):
       cd $AGENT_DIR && source $VENV_DIR/bin/activate && python src/agent.py dev

     Or test locally without a LiveKit room / browser at all:
       cd $AGENT_DIR && source $VENV_DIR/bin/activate && python src/agent.py console

  2) Token server (issues LiveKit tokens + serves the built frontend):
       cd $ROOT_DIR/token-server && npm start
     Open http://localhost:8787 in your browser, or tunnel that port
     (ngrok/cloudflared/ssh -L) to reach it from elsewhere.

  3) (Only if developing the frontend with hot reload instead of the built
     dist/ served by token-server):
       cd $ROOT_DIR/frontend && npm run dev
     Then open http://localhost:5173 (make sure VITE_TOKEN_SERVER_URL in
     frontend/.env.example / .env points at the token-server).

EOF
