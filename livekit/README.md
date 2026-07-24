# CSM Local Voice Agent (LiveKit Agents)

A fully local cascading voice pipeline built on [LiveKit Agents](https://github.com/livekit/agents),
using the same real-time architecture (WebRTC transport, Silero VAD, semantic
turn-detection, interruption handling) as a production LiveKit voice agent --
but every model in the pipeline runs on your own hardware, no cloud inference:

```
mic (browser, via LiveKit room)
  -> Silero VAD                              (local, onnx)
  -> LiveKit turn-detector (EnglishModel)    (local, onnx)
  -> faster-whisper                          (local STT)
  -> Ollama (llama3.2)                       (local LLM, via OpenAI-compatible API)
  -> CSM-1B                                  (local TTS, from the parent csm-main project)
  -> speakers (browser, via LiveKit room)
```

This reuses `../generator.py` / `../models.py` / `../watermarking.py` directly
(imported via a relative path, not duplicated) for the TTS step, and reuses the
LiveKit Cloud project + UI/API-key conventions from `../../voice agent`.

## Project layout

```
livekit/
  agent/            Python LiveKit Agents backend
    src/
      agent.py       entrypoint -- wires local STT/LLM/TTS into an AgentSession
      csm_tts.py      custom TTS plugin wrapping CSM-1B (imports ../../../generator.py)
      whisper_stt.py  custom STT plugin wrapping faster-whisper
    pyproject.toml
    .env.local        LiveKit credentials + local model config (gitignored)
  frontend/          React + Vite UI (mic button, orb visualizer, latency panel)
  token-server/      Express server: issues LiveKit tokens, serves the built frontend
  setup_livekit.sh   one-shot setup for a fresh (GPU) server
```

## How the pieces fit together

- **Transport**: a real LiveKit room (LiveKit Cloud, credentials reused from
  `../../voice agent/.env`). The frontend joins as a participant; the agent is
  dispatched into the same room by name (`AGENT_NAME`, must match between
  `agent/.env.local` and `token-server/.env`).
- **VAD + turn detection**: `silero.VAD` + `livekit.plugins.turn_detector.english.EnglishModel`
  -- both onnx models, downloaded once via `download-files`, no network calls
  at runtime. This is the same turn-detector model `../webui.py` and `../talk.py`
  use standalone; here it's the officially-supported LiveKit integration instead
  of a hand-rolled port.
- **STT**: `whisper_stt.py`'s `WhisperSTT` (faster-whisper, non-streaming),
  wrapped in `stt.StreamAdapter(stt=WhisperSTT(...), vad=silero_vad)` -- LiveKit's
  documented pattern for adapting a batch STT engine into the streaming shape
  `AgentSession` expects (VAD segments speech, each segment is transcribed as
  a batch).
- **LLM**: `livekit.plugins.openai.LLM.with_ollama(model="llama3.2:1b")` --
  Ollama exposes an OpenAI-compatible `/v1/chat/completions` endpoint, so this
  is just the standard OpenAI plugin pointed at `localhost:11434`.
- **TTS**: `csm_tts.py`'s `CSMTTS`, a non-streaming (`ChunkedStream`) plugin
  that calls `generator.Generator.generate()` from the parent project in a
  thread executor (CSM's generation is a blocking torch call) and pushes the
  result as raw 16-bit PCM. Keeps a pinned voice-prompt segment
  (`conversational_a` = female / `conversational_b` = male, set via
  `CSM_SPEAKER`) plus a rolling window of recent turns as context, same as
  `../webui.py` / `../talk.py`.

## Setup

On a fresh GPU server (Ubuntu/Debian assumed):

```bash
./setup_livekit.sh
```

This installs Python 3.11/ffmpeg/Node, creates `.venv`, installs the agent's
dependencies (including the CSM-1B stack from `../requirements.txt`),
installs Ollama and pulls `llama3.2:1b`, downloads the VAD/turn-detector
models, pre-warms the CSM-1B/whisper caches, and builds the frontend.

`agent/.env.local` and `token-server/.env` already have `LIVEKIT_URL` /
`LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` filled in (reused from
`../../voice agent/.env` -- same LiveKit Cloud project). Rotate these if you
don't want to share that project.

## Running

Three processes, each in its own terminal:

```bash
# 1. The agent -- connects to the LiveKit room, runs the local pipeline
cd agent && source ../.venv/bin/activate
python src/agent.py dev

# 2. Token server -- issues LiveKit tokens, serves the built frontend
cd token-server && npm start
# open http://localhost:8787, or tunnel that port to reach it remotely

# 3. (optional, for frontend dev with hot reload instead of the built dist/)
cd frontend && npm run dev
# open http://localhost:5173
```

**Quick local test without a browser or LiveKit room at all:**

```bash
cd agent && source ../.venv/bin/activate
python src/agent.py console
```

This runs the full pipeline against your local terminal mic/speakers,
bypassing LiveKit's network layer entirely -- useful for iterating on prompts
or the TTS voice without needing the frontend running.

## Known quirks / things to watch

- **`initialize_process_timeout`** is raised to 180s in `agent.py` (`AgentServer(...)`)
  because loading CSM-1B + Whisper + VAD + turn-detector in `prewarm()` blows
  past the 10s default, especially on CPU or a cold model cache. If your GPU
  server is fast enough you can lower this.
- **`bitsandbytes`** must be installed even though nothing here is actually
  quantized -- `moshi`'s `quantize.linear()` unconditionally imports it. Same
  fix as the main `csm-main` project; `setup_livekit.sh` handles it.
- **Turn-detector deprecation**: `livekit.plugins.turn_detector.english.EnglishModel`
  logs a deprecation warning pointing at `livekit.agents.inference.TurnDetector`.
  That newer path lives in LiveKit's `inference` namespace (the same one used
  for their *hosted* STT/LLM/TTS gateway elsewhere in the SDK) and its local-only
  status wasn't verified before this was built -- don't switch to it without
  confirming it doesn't route through LiveKit's cloud inference gateway, which
  would violate the "fully local" goal.
- **Occasional literal "assistant" text in LLM output**: observed once in
  testing (`"assistant\n\nHi there! ..."`). Likely a chat-template quirk from
  `llama3.2:1b` via Ollama's OpenAI-compat endpoint, not a pipeline bug. Watch
  for it in real use; if it recurs, tuning the system prompt or trying
  `llama3.2:3b`/an explicitly `-instruct` tagged model is the likely fix.
- **CPU vs GPU**: `agent.py` auto-detects `torch.cuda.is_available()`. On CPU
  (e.g. local dev on a Mac), expect noticeably slower TTS generation than the
  target GPU server.
