# CSM-1B TTS server

Standalone WebSocket server that runs CSM-1B on a GPU box and, per utterance,
generates the complete reply before sending it back as a single PCM blob.
Meant to be called by `../livekit/agent/src/csm_tts_remote.py`, but the
protocol is plain WebSocket + JSON/binary frames so any client can use it.

## Running it

```bash
./setup_tts_server.sh                # fresh GPU box: full setup + launch
./setup_tts_server.sh --skip-setup   # just (re)launch, deps already installed
./setup_tts_server.sh --no-tls       # plain ws:// instead of wss:// (local testing)
```

This installs the CSM-1B stack (same as `../requirements.txt`) plus
`fastapi`, `uvicorn[standard]`, `websockets`, pre-downloads CSM-1B, and
launches `uvicorn server.tts_server:app` on port 8000, terminated with the
repo's self-signed cert at `../.certs/` (rotate these for anything beyond
local testing/a private network).

Env vars (all optional):
- `CSM_MAX_CONTEXT_SEGMENTS` (default `6`) -- how many recent turns to keep as generation context per connection.
- `CSM_MAX_REPLY_AUDIO_MS` (default `15000`) -- cap on a single reply's length.

## Wire protocol

One WebSocket connection per conversation (e.g. per LiveKit room), opened
once and reused for every turn in that conversation, so the server can keep
rolling context (recent turns + a pinned voice-prompt segment) the same way
`csm_tts.py`'s in-process `_voice_context` does today. Calls on one
connection are strictly sequential -- send one `synthesize`, drain its
response (`done`/`error`), then send the next. `request_id` is for
client-side correlation/logging only, not for demuxing.

Endpoint: `wss://<host>:8000/v1/tts/stream` (or `ws://` with `--no-tls`).

**Client -> server** (JSON text frames):

```jsonc
// once, right after connecting
{"type": "configure", "speaker": 0, "voice_prompt_text": "...", "voice_prompt_audio_b64": "<base64 float32 PCM mono @ 24kHz>"}

// once per utterance to synthesize
{"type": "synthesize", "request_id": "abc123", "text": "Hello there."}
```

**Server -> client:**

```jsonc
{"type": "ready"}                              // after configure is applied
// one binary frame: the complete reply as PCM16LE mono @ 24kHz
{"type": "done", "request_id": "abc123"}       // this utterance's audio is complete
{"type": "error", "request_id": "abc123", "message": "..."}
```

## Watermarking

The complete reply is watermarked as a whole
(`watermarking.py`'s `watermark()`, same key as the rest of this repo --
**keep it in place**, see the top-level README's misuse policy) before being
sent to the client.

## GPU concurrency

CSM's `Model` instance holds mutable KV-cache buffers that get reset/written
on every call to `generate_frame`. One `Generator` is loaded once at server
startup and shared across all connections; a single global `asyncio.Lock`
serializes `generate` calls across connections so two simultaneous rooms
can't corrupt each other's cache state. This costs nothing but queuing,
since one GPU serializes the actual compute anyway.
