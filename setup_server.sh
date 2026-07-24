#!/usr/bin/env bash
#
# One-shot setup + launch script for running the CSM-1B voice chat web UI on a
# remote GPU server (Ubuntu/Debian assumed). Idempotent: safe to re-run if it
# fails partway through, it skips steps already done.
#
# What it does:
#   1. Checks for an NVIDIA GPU (warns if none found, continues on CPU).
#   2. Installs Python 3.11, ffmpeg, git if missing (apt).
#   3. Clones this repo (if not already present) and creates a venv.
#   4. Installs all Python deps (torch/CSM stack + gradio/faster-whisper/onnxruntime).
#   5. Vendors the Silero VAD onnx weights (extracted from the livekit-plugins-silero
#      wheel, same approach used for local dev -- no livekit-agents dependency needed).
#   6. Installs Ollama and pulls the local LLM model.
#   7. Logs into Hugging Face (needed for the gated sesame/csm-1b repo).
#   8. Pre-downloads all models so the first web request isn't slow.
#   9. Launches webui.py, which prints a local URL -- point your tunnel
#      (ngrok/cloudflared/ssh -L/etc.) at that port from your local machine.
#
# Usage:
#   ./setup_server.sh                       # full setup + launch
#   ./setup_server.sh --skip-setup          # just (re)launch the web UI
#   ./setup_server.sh --https               # also serve over HTTPS (self-signed cert)
#   OLLAMA_MODEL=llama3.2:3b ./setup_server.sh
#   HF_TOKEN=hf_xxx ./setup_server.sh       # non-interactive HF login
#
set -euo pipefail

REPO_URL="https://github.com/SesameAILabs/csm.git"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
PYTHON_BIN="python3.11"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b}"
WEBUI_PORT="${WEBUI_PORT:-7860}"
WEBUI_HOST="${WEBUI_HOST:-127.0.0.1}"
VOICE_SPEAKER="${VOICE_SPEAKER:-conversational_a}"
SKIP_SETUP=0
HTTPS_FLAG=()

for arg in "$@"; do
  case "$arg" in
    --https) HTTPS_FLAG=(--https) ;;
    --skip-setup) SKIP_SETUP=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }

if [ "$SKIP_SETUP" -eq 1 ]; then
  log "Skipping setup, launching web UI directly"
  source "$VENV_DIR/bin/activate"
  exec python webui.py --host "$WEBUI_HOST" --port "$WEBUI_PORT" --speaker "$VOICE_SPEAKER" "${HTTPS_FLAG[@]}"
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

# faster-whisper / gradio / onnxruntime pull in a newer huggingface-hub by default,
# which conflicts with moshi/transformers/tokenizers pins in requirements.txt.
# Install them, then re-pin huggingface-hub to the compatible version.
pip install -q faster-whisper "gradio==5.25.2" onnxruntime requests
pip install -q "huggingface_hub==0.28.1"

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

if [ ! -f "$REPO_DIR/silero_vad.onnx" ]; then
  log "Vendoring Silero VAD onnx weights (from livekit-plugins-silero, no livekit-agents needed)"
  pip install -q --no-deps livekit-plugins-silero
  SILERO_PATH=$(python -c "
import glob
matches = glob.glob('$VENV_DIR/lib/python3.11/site-packages/livekit/plugins/silero/resources/silero_vad.onnx')
print(matches[0] if matches else '')
")
  if [ -z "$SILERO_PATH" ]; then
    echo "ERROR: could not locate silero_vad.onnx inside livekit-plugins-silero package." >&2
    exit 1
  fi
  cp "$SILERO_PATH" "$REPO_DIR/silero_vad.onnx"
  pip uninstall -y -q livekit-plugins-silero
fi

log "Installing Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
if ! pgrep -x ollama >/dev/null 2>&1; then
  nohup ollama serve > "$REPO_DIR/ollama.log" 2>&1 &
  sleep 3
fi
log "Pulling Ollama model: $OLLAMA_MODEL"
ollama pull "$OLLAMA_MODEL"

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

log "Pre-downloading models (CSM-1B, Silero-adjacent turn-detector, Whisper, tokenizer)"
python - <<'PYEOF'
from huggingface_hub import hf_hub_download
from faster_whisper import WhisperModel
from generator import load_csm_1b
import torch

print("Downloading turn-detector model + tokenizer...")
hf_hub_download(repo_id="livekit/turn-detector", filename="model_q8.onnx", subfolder="onnx", revision="v1.2.2-en")
hf_hub_download(repo_id="livekit/turn-detector", filename="languages.json", revision="v1.2.2-en")
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("livekit/turn-detector", revision="v1.2.2-en")

print("Downloading Llama-3.2-1B tokenizer mirror...")
AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")

print("Downloading CSM-1B voice prompts...")
hf_hub_download(repo_id="sesame/csm-1b", filename="prompts/conversational_a.wav")
hf_hub_download(repo_id="sesame/csm-1b", filename="prompts/conversational_b.wav")

print("Downloading faster-whisper base.en...")
device = "cuda" if torch.cuda.is_available() else "cpu"
WhisperModel("base.en", device=device, compute_type="float16" if device == "cuda" else "int8")

print("Downloading CSM-1B weights (this is the big one, ~a few GB)...")
load_csm_1b(device=device)

print("All models cached.")
PYEOF

log "Setup complete. Launching web UI..."
OLLAMA_MODEL="$OLLAMA_MODEL" python webui.py --host "$WEBUI_HOST" --port "$WEBUI_PORT" --speaker "$VOICE_SPEAKER" "${HTTPS_FLAG[@]}"
