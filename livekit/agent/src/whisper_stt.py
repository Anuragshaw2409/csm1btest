"""
Custom LiveKit Agents STT plugin wrapping faster-whisper (local, offline).
Batch/non-streaming -- wrap with stt.StreamAdapter(stt=WhisperSTT(...), vad=...)
to get streaming-shaped behavior (VAD segments speech, each segment is then
transcribed as a batch). This is LiveKit's documented pattern for adapting a
non-streaming STT engine.
"""

from __future__ import annotations

import asyncio

import numpy as np
from faster_whisper import WhisperModel
from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer, shortuuid


class WhisperSTT(stt.STT):
    def __init__(self, *, model_size: str = "base.en", device: str | None = None, language: str = "en") -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._language = language
        compute_type = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(model_size, device=device or "cpu", compute_type=compute_type)

    @property
    def model(self) -> str:
        return "faster-whisper"

    @property
    def provider(self) -> str:
        return "local"

    def _transcribe_sync(self, samples: np.ndarray) -> str:
        segments, _ = self._model.transcribe(samples, language=self._language, vad_filter=False)
        return " ".join(seg.text.strip() for seg in segments).strip()

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)
        samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        if frame.num_channels > 1:
            samples = samples.reshape(-1, frame.num_channels).mean(axis=1)
        if frame.sample_rate != 16000:
            import torch
            import torchaudio

            wav = torch.from_numpy(samples)
            wav = torchaudio.functional.resample(wav, frame.sample_rate, 16000)
            samples = wav.numpy()

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, self._transcribe_sync, samples)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=shortuuid(),
            alternatives=[stt.SpeechData(language=self._language, text=text)],
        )
