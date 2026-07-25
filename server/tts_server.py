
"""
Standalone CSM-1B TTS inference server, meant to run on a remote GPU box.

Exposes a single WebSocket endpoint (`/v1/tts/stream`) that a LiveKit agent
(or any other client) can connect to once per conversation and drive with a
small JSON control protocol, receiving the complete reply audio back as a
single binary PCM16LE blob once generation finishes. See ../server/README.md
for the full wire protocol.

Run with:
    uvicorn server.tts_server:app --host 0.0.0.0 --port 8000 \
        --ssl-keyfile .certs/key.pem --ssl-certfile .certs/cert.pem
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# server/tts_server.py -> parents[1] == csm-main (repo root), so `generator`
# etc. resolve the same way they do for webui.py/talk.py/run_csm.py.
_CSM_ROOT = Path(__file__).resolve().parents[1]
if str(_CSM_ROOT) not in sys.path:
    sys.path.insert(0, str(_CSM_ROOT))

from generator import Segment, load_csm_1b  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("csm-tts-server")

MAX_CONTEXT_SEGMENTS = int(os.environ.get("CSM_MAX_CONTEXT_SEGMENTS", "6"))
MAX_REPLY_AUDIO_MS = float(os.environ.get("CSM_MAX_REPLY_AUDIO_MS", "15000"))

app = FastAPI()

_device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading CSM-1B on device={_device} ...")
_generator = load_csm_1b(device=_device)
logger.info("CSM-1B loaded.")

# CSM's Model instance carries mutable KV-cache buffers that generate()
# resets/writes on every call -- serialize access across concurrent
# connections so two rooms can't race on the same buffers. One GPU serializes
# the actual compute anyway, so this costs nothing but queuing.
_gpu_lock = asyncio.Lock()


class _ConnectionState:
    def __init__(self) -> None:
        self.speaker_id: int = 0
        self.voice_prompt_segment: Segment | None = None
        self.context: list[Segment] = []

    def configure(self, *, speaker: int, voice_prompt_text: str | None, voice_prompt_audio_b64: str | None) -> None:
        self.speaker_id = speaker
        if voice_prompt_text and voice_prompt_audio_b64:
            audio = torch.frombuffer(bytearray(base64.b64decode(voice_prompt_audio_b64)), dtype=torch.float32)
            self.voice_prompt_segment = Segment(text=voice_prompt_text, speaker=speaker, audio=audio)
            self.context = [self.voice_prompt_segment]

    def append_turn(self, text: str, audio: torch.Tensor) -> None:
        new_segment = Segment(text=text, speaker=self.speaker_id, audio=audio)
        prompt = self.voice_prompt_segment
        generated = [s for s in self.context if s is not prompt]
        generated.append(new_segment)
        if prompt is not None:
            self.context = [prompt] + generated[-(MAX_CONTEXT_SEGMENTS - 1) :]
        else:
            self.context = generated[-MAX_CONTEXT_SEGMENTS:]


def _pcm16_bytes(audio_tensor: torch.Tensor) -> bytes:
    return audio_tensor.clamp(-1.0, 1.0).mul(32767.0).to(torch.int16).cpu().numpy().tobytes()


@app.websocket("/v1/tts/stream")
async def tts_stream(ws: WebSocket) -> None:
    await ws.accept()
    state = _ConnectionState()
    loop = asyncio.get_event_loop()

    try:
        while True:
            message = await ws.receive_json()
            msg_type = message.get("type")

            if msg_type == "configure":
                state.configure(
                    speaker=message.get("speaker", 0),
                    voice_prompt_text=message.get("voice_prompt_text"),
                    voice_prompt_audio_b64=message.get("voice_prompt_audio_b64"),
                )
                await ws.send_json({"type": "ready"})
                continue

            if msg_type == "synthesize":
                request_id = message.get("request_id", "")
                text = message.get("text", "")
                try:
                    async with _gpu_lock:
                        # generate() is a blocking torch call, so it must run
                        # in a thread. Unlike generate_stream(), it only
                        # returns once the whole reply has been generated,
                        # decoded and watermarked as a single tensor.
                        full_audio = await loop.run_in_executor(
                            None,
                            lambda: _generator.generate(
                                text=text,
                                speaker=state.speaker_id,
                                context=state.context,
                                max_audio_length_ms=MAX_REPLY_AUDIO_MS,
                            ),
                        )

                    state.append_turn(text, full_audio)
                    await ws.send_bytes(_pcm16_bytes(full_audio))
                    await ws.send_json({"type": "done", "request_id": request_id})
                except Exception as exc:  # noqa: BLE001
                    logger.exception("synthesize failed")
                    try:
                        await ws.send_json({"type": "error", "request_id": request_id, "message": str(exc)})
                    except Exception:  # noqa: BLE001
                        # The client is already gone (e.g. disconnected mid-stream) --
                        # nothing left to notify, don't let this mask the real error above.
                        logger.warning("could not deliver error to client, connection already closed")
                continue

            logger.warning(f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info("client disconnected")
