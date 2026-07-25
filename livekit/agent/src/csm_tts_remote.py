"""
Custom LiveKit Agents TTS plugin that calls a remote CSM-1B inference server
(../../../server/tts_server.py) over a WebSocket instead of loading the model
in-process. Use this instead of csm_tts.py when CSM runs on a separate GPU
box from the agent process. Still a ChunkedStream (one full sentence in per
call, same as csm_tts.py) -- but unlike csm_tts.py, audio is pushed to the
output as each chunk arrives from the server, not all at once at the end, so
playback starts well before the whole reply has finished generating.

See ../../../server/README.md for the wire protocol this speaks.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import ssl

import torch
import websockets
from websockets.protocol import State as _WSState

from livekit.agents import APIConnectionError
from livekit.agents import tts
from livekit.agents.tts.tts import AudioEmitter
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

SAMPLE_RATE = 24_000  # fixed by CSM's Mimi codec, matches generator.Generator.sample_rate

logger = logging.getLogger("csm-tts-remote")


class CSMRemoteTTS(tts.TTS):
    def __init__(
        self,
        *,
        server_url: str,
        speaker_id: int = 0,
        voice_prompt_text: str | None = None,
        voice_prompt_audio: torch.Tensor | None = None,
        insecure_tls: bool = False,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        self._server_url = server_url
        self._insecure_tls = insecure_tls
        self._speaker_id = speaker_id
        self._voice_prompt_text = voice_prompt_text
        self._voice_prompt_audio = voice_prompt_audio

        self._ws: websockets.ClientConnection | None = None
        self._connect_lock = asyncio.Lock()
        # AgentSession(preemptive_generation=True) can start a new synthesize()
        # call before a previous one has finished (speculative next-turn TTS).
        # The wire protocol (server/tts_server.py) only ever handles one
        # synthesize/response cycle at a time per connection, and raw send/recv
        # on one websockets.ClientConnection isn't safe to interleave from
        # multiple tasks -- so serialize full request/response cycles here.
        # One GPU serializes generation server-side anyway, so this doesn't
        # cost real concurrency, just stops the two calls' bytes from
        # corrupting each other on the wire.
        self._request_lock = asyncio.Lock()

    def set_voice_prompt(self, *, text: str, audio: torch.Tensor) -> None:
        """(Re)sets the pinned voice-prompt segment sent to the server on
        (re)connect. `audio` must already be at `self.sample_rate`, mono."""
        self._voice_prompt_text = text
        self._voice_prompt_audio = audio
        # Force a reconnect + re-configure on the next synth call.
        self._ws = None

    @property
    def model(self) -> str:
        return "csm-1b"

    @property
    def provider(self) -> str:
        return "sesame"

    def _ssl_context(self) -> ssl.SSLContext | None:
        if not self._server_url.startswith("wss://"):
            return None
        ctx = ssl.create_default_context()
        if self._insecure_tls:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    async def _ensure_connected(self) -> websockets.ClientConnection:
        async with self._connect_lock:
            if self._ws is not None and self._ws.state is _WSState.OPEN:
                return self._ws

            try:
                # A synthesize can legitimately take a long time (slow/CPU/
                # emulated hardware, or just a long reply) while the GPU-bound
                # work runs in a server-side thread that starves its asyncio
                # loop's ability to answer keepalive pings promptly. Disable
                # the client's own ping/pong entirely rather than have
                # `websockets` kill a healthy-but-slow connection out from
                # under an in-flight generation.
                ws = await websockets.connect(
                    self._server_url, ssl=self._ssl_context(), ping_interval=None
                )
                if self._voice_prompt_text and self._voice_prompt_audio is not None:
                    audio_b64 = base64.b64encode(
                        self._voice_prompt_audio.to(torch.float32).cpu().numpy().tobytes()
                    ).decode("ascii")
                    await ws.send(
                        json.dumps(
                            {
                                "type": "configure",
                                "speaker": self._speaker_id,
                                "voice_prompt_text": self._voice_prompt_text,
                                "voice_prompt_audio_b64": audio_b64,
                            }
                        )
                    )
                    reply = json.loads(await ws.recv())
                    if reply.get("type") != "ready":
                        raise APIConnectionError(f"unexpected configure reply: {reply}")
            except OSError as exc:
                raise APIConnectionError(f"failed to connect to CSM TTS server: {exc}") from exc

            self._ws = ws
            return ws

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return _CSMRemoteChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _CSMRemoteChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: CSMRemoteTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._csm_tts = tts

    async def _run(self, output_emitter: AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=self._csm_tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        request_id = shortuuid()
        try:
            async with self._csm_tts._request_lock:
                ws = await self._csm_tts._ensure_connected()
                await ws.send(
                    json.dumps({"type": "synthesize", "request_id": request_id, "text": self.input_text})
                )

                while True:
                    message = await ws.recv()
                    if isinstance(message, (bytes, bytearray)):
                        output_emitter.push(bytes(message))
                        continue

                    payload = json.loads(message)
                    if payload.get("type") == "done":
                        break
                    if payload.get("type") == "error":
                        raise APIConnectionError(f"CSM TTS server error: {payload.get('message')}")
        except (websockets.exceptions.WebSocketException, OSError) as exc:
            self._csm_tts._ws = None  # force reconnect next call
            raise APIConnectionError(f"CSM TTS server connection failed: {exc}") from exc
