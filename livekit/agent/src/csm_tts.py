"""
Custom LiveKit Agents TTS plugin that wraps CSM-1B (Sesame's conversational
speech model) from the parent csm-main project. Non-streaming (ChunkedStream):
CSM generates the whole reply as one shot, same as talk.py/webui.py.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import torch

# csm-main/livekit/agent/src/csm_tts.py -> parents[3] == csm-main
_CSM_ROOT = Path(__file__).resolve().parents[3]
if str(_CSM_ROOT) not in sys.path:
    sys.path.insert(0, str(_CSM_ROOT))

from generator import Segment, load_csm_1b  # noqa: E402

from livekit.agents import tts
from livekit.agents.tts.tts import AudioEmitter
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

MAX_CONTEXT_SEGMENTS = 6  # keep recent turns only, so prompts stay under CSM's seq len
MAX_REPLY_AUDIO_MS = 15_000


class CSMTTS(tts.TTS):
    def __init__(
        self,
        *,
        device: str | None = None,
        speaker_id: int = 0,
        voice_prompt_text: str | None = None,
        voice_prompt_audio: torch.Tensor | None = None,
    ) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._generator = load_csm_1b(device=device)
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=self._generator.sample_rate,
            num_channels=1,
        )
        self._speaker_id = speaker_id
        self._voice_prompt_segment: Segment | None = None
        if voice_prompt_text and voice_prompt_audio is not None:
            self._voice_prompt_segment = Segment(
                text=voice_prompt_text, speaker=speaker_id, audio=voice_prompt_audio
            )
        self._voice_context: list[Segment] = (
            [self._voice_prompt_segment] if self._voice_prompt_segment else []
        )

    def set_voice_prompt(self, *, text: str, audio: torch.Tensor) -> None:
        """(Re)sets the pinned voice-prompt segment used as context for every
        synthesis, so replies stay consistent in this voice. `audio` must
        already be at `self.sample_rate`, mono."""
        self._voice_prompt_segment = Segment(text=text, speaker=self._speaker_id, audio=audio)
        self._voice_context = [self._voice_prompt_segment]

    @property
    def model(self) -> str:
        return "csm-1b"

    @property
    def provider(self) -> str:
        return "sesame"

    def _generate_sync(self, text: str) -> torch.Tensor:
        return self._generator.generate(
            text=text,
            speaker=self._speaker_id,
            context=self._voice_context,
            max_audio_length_ms=MAX_REPLY_AUDIO_MS,
        )

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return _CSMChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _CSMChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: CSMTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._csm_tts = tts

    async def _run(self, output_emitter: AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=self._csm_tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        loop = asyncio.get_event_loop()
        audio_tensor = await loop.run_in_executor(
            None, self._csm_tts._generate_sync, self.input_text
        )

        # Keep voice context anchored: pin the original voice prompt (if any) at
        # the front, then the most recent generated turns -- mirrors talk.py/webui.py.
        new_segment = Segment(
            text=self.input_text, speaker=self._csm_tts._speaker_id, audio=audio_tensor
        )
        prompt = self._csm_tts._voice_prompt_segment
        generated = [s for s in self._csm_tts._voice_context if s is not prompt]
        generated.append(new_segment)
        if prompt is not None:
            self._csm_tts._voice_context = [prompt] + generated[-(MAX_CONTEXT_SEGMENTS - 1) :]
        else:
            self._csm_tts._voice_context = generated[-MAX_CONTEXT_SEGMENTS:]

        pcm_bytes = (
            audio_tensor.clamp(-1.0, 1.0).mul(32767.0).to(torch.int16).cpu().numpy().tobytes()
        )
        output_emitter.push(pcm_bytes)
