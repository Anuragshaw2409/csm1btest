"""
Fully local voice conversation with LiveKit-style turn detection:

  mic -> Silero VAD (speech segmentation)
      -> LiveKit turn-detector model (semantic "are they actually done talking?")
      -> faster-whisper (transcription)
      -> Ollama LLM (reply)
      -> CSM-1B (speech synthesis)
      -> speakers

Usage:
    python talk.py
"""

import collections
import json
import os
import subprocess
import tempfile

os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import onnxruntime as ort
import requests
import sounddevice as sd
import torch
import torchaudio
from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from generator import Segment, load_csm_1b

SILERO_VAD_ONNX_PATH = os.path.join(os.path.dirname(__file__), "silero_vad.onnx")

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:1b"
SPEAKER_ID = 0
MAX_AUDIO_MS = 15_000
MAX_CONTEXT_SEGMENTS = 6  # keep recent turns only, so prompts stay under CSM's seq len

# CSM-1B's built-in "conversational_a" voice prompt (female, ~207Hz est. pitch).
# Used as a persistent context segment so every reply is spoken in this voice.
VOICE_PROMPT_NAME = "conversational_a"
VOICE_PROMPT_TEXT = (
    "like revising for an exam I'd have to try and like keep up the momentum because I'd "
    "start really early I'd be like okay I'm gonna start revising now and then like "
    "you're revising for ages and then I just like start losing steam I didn't do that "
    "for the exam we had recently to be fair that was a more of a last minute scenario "
    "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
    "sort of start the day with this not like a panic but like a"
)

SYSTEM_PROMPT = (
    "You are a friendly, concise voice assistant speaking out loud. "
    "Keep replies short (1-3 sentences), conversational, and avoid lists, "
    "markdown, emojis, or anything that doesn't read naturally as speech."
)

# --- Silero VAD settings (mirrors livekit-plugins-silero defaults) ---
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 512  # Silero's fixed window size at 16kHz (32ms)
FRAME_SEC = WINDOW_SAMPLES / SAMPLE_RATE
ACTIVATION_THRESHOLD = 0.5
DEACTIVATION_THRESHOLD = max(ACTIVATION_THRESHOLD - 0.15, 0.01)
MIN_SPEECH_DURATION = 0.05
MIN_SILENCE_DURATION = 0.55
PREFIX_PADDING_DURATION = 0.5
MAX_UTTERANCE_SEC = 20.0

# --- LiveKit turn-detector (semantic end-of-utterance) settings ---
EOU_REPO = "livekit/turn-detector"
EOU_REVISION = "v1.2.2-en"
EOU_MAX_HISTORY_TOKENS = 128
EOU_MAX_HISTORY_TURNS = 6
EOU_MAX_GRACE_ROUNDS = 3  # how many "probably not done" extensions before forcing the turn to end


class SileroVAD:
    """Standalone port of livekit-plugins-silero's OnnxModel wrapper (Apache-2.0),
    using the same silero_vad.onnx weights, without depending on livekit-agents."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            SILERO_VAD_ONNX_PATH, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.sample_rate = sample_rate
        self.window_size_samples = 512 if sample_rate == 16000 else 256
        self.context_size = 64 if sample_rate == 16000 else 32
        self._sample_rate_nd = np.array(sample_rate, dtype=np.int64)
        self.reset()

    def reset(self):
        self._context = np.zeros((1, self.context_size), dtype=np.float32)
        self._rnn_state = np.zeros((2, 1, 128), dtype=np.float32)

    def prob(self, frame_f32: np.ndarray) -> float:
        input_buffer = np.concatenate([self._context, frame_f32[np.newaxis, :]], axis=1)
        out, self._rnn_state = self.session.run(
            None,
            {"input": input_buffer, "state": self._rnn_state, "sr": self._sample_rate_nd},
        )
        self._context = input_buffer[:, -self.context_size :]
        return float(out.item())


class EOUModel:
    """Reimplements livekit-plugins-turn-detector's inference logic standalone,
    without the livekit-agents worker/executor machinery."""

    def __init__(self):
        onnx_path = hf_hub_download(
            repo_id=EOU_REPO, filename="model_q8.onnx", subfolder="onnx", revision=EOU_REVISION
        )
        lang_path = hf_hub_download(repo_id=EOU_REPO, filename="languages.json", revision=EOU_REVISION)
        self.tokenizer = AutoTokenizer.from_pretrained(
            EOU_REPO, revision=EOU_REVISION, truncation_side="left"
        )
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        with open(lang_path) as f:
            self.threshold = json.load(f)["en"]["threshold"]

    def _format_chat_ctx(self, chat_ctx: list[dict]) -> str:
        combined = []
        for msg in chat_ctx:
            if not msg["content"]:
                continue
            if combined and combined[-1]["role"] == msg["role"]:
                combined[-1] = {"role": msg["role"], "content": combined[-1]["content"] + " " + msg["content"]}
            else:
                combined.append(dict(msg))

        text = self.tokenizer.apply_chat_template(
            combined, add_generation_prompt=False, add_special_tokens=False, tokenize=False
        )
        ix = text.rfind("<|im_end|>")
        return text[:ix]

    def probability(self, chat_ctx: list[dict]) -> float:
        messages = [m for m in chat_ctx if m["role"] in ("user", "assistant")][-EOU_MAX_HISTORY_TURNS:]
        text = self._format_chat_ctx(messages)
        inputs = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="np",
            max_length=EOU_MAX_HISTORY_TOKENS,
            truncation=True,
        )
        outputs = self.session.run(None, {"input_ids": inputs["input_ids"].astype("int64")})
        return float(outputs[0].flatten()[-1])


def transcribe(stt: WhisperModel, audio_f32: np.ndarray) -> str:
    segments, _ = stt.transcribe(audio_f32, language="en", vad_filter=False)
    return " ".join(seg.text.strip() for seg in segments).strip()


def record_utterance(vad: SileroVAD, eou: EOUModel, stt: WhisperModel, chat_history: list[dict]) -> str:
    """Blocks until it captures one full user turn, using Silero VAD to segment
    speech and the LiveKit turn-detector model to decide whether a detected pause
    is genuinely the end of the turn (vs. a mid-thought pause). Returns transcribed text."""
    vad.reset()
    prefix_padding_frames = max(1, int(PREFIX_PADDING_DURATION / FRAME_SEC))
    pre_buffer: collections.deque = collections.deque(maxlen=prefix_padding_frames)

    speaking = False
    speech_frames: list[np.ndarray] = []
    speech_run = 0.0
    silence_run = 0.0
    grace_rounds_used = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=WINDOW_SAMPLES
    ) as stream:
        print("Listening...")
        while True:
            block, _ = stream.read(WINDOW_SAMPLES)
            frame_i16 = block.flatten()
            frame_f32 = frame_i16.astype(np.float32) / 32768.0
            p = vad.prob(frame_f32)

            if not speaking:
                pre_buffer.append(frame_i16)

            is_active = p >= ACTIVATION_THRESHOLD or (speaking and p > DEACTIVATION_THRESHOLD)

            if is_active:
                silence_run = 0.0
                speech_run += FRAME_SEC
                if not speaking and speech_run >= MIN_SPEECH_DURATION:
                    speaking = True
                    speech_frames = list(pre_buffer)
                if speaking:
                    speech_frames.append(frame_i16)
                continue

            speech_run = 0.0
            if not speaking:
                continue

            speech_frames.append(frame_i16)
            silence_run += FRAME_SEC

            utterance_sec = len(speech_frames) * FRAME_SEC
            if silence_run < MIN_SILENCE_DURATION and utterance_sec < MAX_UTTERANCE_SEC:
                continue

            # Candidate pause: transcribe what we have and ask the semantic
            # turn-detector whether the user is actually finished.
            audio = np.concatenate(speech_frames).astype(np.float32) / 32768.0
            text = transcribe(stt, audio)

            if not text:
                speaking = False
                speech_frames = []
                silence_run = 0.0
                grace_rounds_used = 0
                continue

            candidate_ctx = chat_history + [{"role": "user", "content": text}]
            eou_prob = eou.probability(candidate_ctx)
            print(f"  (transcript so far: {text!r}, eou_probability={eou_prob:.4f}, threshold={eou.threshold})")

            done_talking = eou_prob >= eou.threshold
            timed_out = utterance_sec >= MAX_UTTERANCE_SEC or grace_rounds_used >= EOU_MAX_GRACE_ROUNDS

            if done_talking or timed_out:
                return text

            # Probably mid-thought: give them more time to keep talking.
            grace_rounds_used += 1
            silence_run = 0.0


def get_llm_reply(history: list[dict]) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "messages": history, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def main():
    device = "cpu"
    print("Loading Silero VAD...")
    vad = SileroVAD()

    print("Loading LiveKit turn-detector (semantic end-of-turn model)...")
    eou = EOUModel()

    print("Loading faster-whisper (base.en)...")
    stt = WhisperModel("base.en", device="cpu", compute_type="int8")

    print(f"Loading CSM-1B on {device} (this can take ~20s)...")
    generator = load_csm_1b(device=device)

    print(f"Loading voice prompt '{VOICE_PROMPT_NAME}'...")
    voice_prompt_path = hf_hub_download(
        repo_id="sesame/csm-1b", filename=f"prompts/{VOICE_PROMPT_NAME}.wav"
    )
    prompt_wav, prompt_sr = torchaudio.load(voice_prompt_path)
    prompt_wav = torchaudio.functional.resample(
        prompt_wav.mean(0), orig_freq=prompt_sr, new_freq=generator.sample_rate
    )
    voice_prompt_segment = Segment(text=VOICE_PROMPT_TEXT, speaker=SPEAKER_ID, audio=prompt_wav)

    print(f"Ready. Talking to Ollama model '{OLLAMA_MODEL}'. Speak after 'Listening...'. Ctrl+C to quit.\n")

    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    voice_context: list[Segment] = [voice_prompt_segment]

    while True:
        try:
            user_text = record_utterance(vad, eou, stt, chat_history)
        except KeyboardInterrupt:
            print("\nBye.")
            break

        if not user_text:
            print("(didn't catch that, try again)")
            continue

        print(f"You: {user_text}")
        chat_history.append({"role": "user", "content": user_text})

        reply_text = get_llm_reply(chat_history)
        print(f"Bot: {reply_text}")
        chat_history.append({"role": "assistant", "content": reply_text})

        audio_out = generator.generate(
            text=reply_text,
            speaker=SPEAKER_ID,
            context=voice_context,
            max_audio_length_ms=MAX_AUDIO_MS,
        )

        voice_context.append(Segment(text=reply_text, speaker=SPEAKER_ID, audio=audio_out))
        # keep the voice prompt pinned at the front so the voice doesn't drift,
        # plus the most recent generated turns
        voice_context = [voice_prompt_segment] + voice_context[1:][-(MAX_CONTEXT_SEGMENTS - 1) :]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        torchaudio.save(wav_path, audio_out.unsqueeze(0).cpu(), generator.sample_rate)
        subprocess.run(["afplay", wav_path])
        os.remove(wav_path)


if __name__ == "__main__":
    main()
