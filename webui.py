"""
Web UI for the local voice conversation pipeline, meant to run on a GPU server
and be reached over a tunnel (ngrok / cloudflared / ssh -L / etc.) from your
local machine's browser.

Pipeline (same as talk.py, but audio comes from the browser mic instead of a
local microphone, and replies are played back in the browser instead of via
afplay):

  browser mic (streamed)
      -> Silero VAD (speech segmentation)
      -> LiveKit turn-detector model (semantic "are they actually done talking?")
      -> faster-whisper (transcription)
      -> Ollama LLM (reply)
      -> CSM-1B (speech synthesis)
      -> browser audio playback

Usage:
    python webui.py [--port 7860] [--speaker conversational_a]

Prints a local URL (e.g. http://0.0.0.0:7860) once models are loaded. Point
your tunnel of choice at that port to reach it from elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field

os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr
import numpy as np
import onnxruntime as ort
import requests
import torch
import torchaudio
from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from generator import Segment, load_csm_1b

SILERO_VAD_ONNX_PATH = os.path.join(os.path.dirname(__file__), "silero_vad.onnx")

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
SPEAKER_ID = 0
MAX_AUDIO_MS = 15_000
MAX_CONTEXT_SEGMENTS = 6  # keep recent turns only, so prompts stay under CSM's seq len

SYSTEM_PROMPT = (
    "You are a friendly, concise voice assistant speaking out loud. "
    "Keep replies short (1-3 sentences), conversational, and avoid lists, "
    "markdown, emojis, or anything that doesn't read naturally as speech."
)

# CSM-1B's built-in voice prompts (see prompts/ in the sesame/csm-1b HF repo).
# conversational_a tests out as female (~207Hz est. pitch), conversational_b as
# male (~118Hz). Pick with --speaker.
VOICE_PROMPTS = {
    "conversational_a": (
        "like revising for an exam I'd have to try and like keep up the momentum because I'd "
        "start really early I'd be like okay I'm gonna start revising now and then like "
        "you're revising for ages and then I just like start losing steam I didn't do that "
        "for the exam we had recently to be fair that was a more of a last minute scenario "
        "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
        "sort of start the day with this not like a panic but like a"
    ),
    "conversational_b": (
        "like a super Mario level. Like it's very like high detail. And like, once you get "
        "into the park, it just like, everything looks like a computer game and they have all "
        "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
        "will have like a question block. And if you like, you know, punch it, a coin will "
        "come out. So like everyone, when they come into the park, they get like this little "
        "bracelet and then you can go punching question blocks around."
    ),
}

# --- Silero VAD settings (mirrors livekit-plugins-silero defaults) ---
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 512  # Silero's fixed window size at 16kHz (32ms)
FRAME_SEC = WINDOW_SAMPLES / SAMPLE_RATE
ACTIVATION_THRESHOLD = 0.5
DEACTIVATION_THRESHOLD = max(ACTIVATION_THRESHOLD - 0.15, 0.01)
MIN_SPEECH_DURATION = 0.05
MIN_SILENCE_DURATION = 0.55
PREFIX_PADDING_DURATION = 0.5
MAX_UTTERANCE_SEC = 8.0  # hard safety cap: force a turn to resolve even if VAD never sees clean silence

# --- LiveKit turn-detector (semantic end-of-utterance) settings ---
EOU_REPO = "livekit/turn-detector"
EOU_REVISION = "v1.2.2-en"
EOU_MAX_HISTORY_TOKENS = 128
EOU_MAX_HISTORY_TURNS = 6
EOU_MAX_GRACE_ROUNDS = 3  # how many "probably not done" extensions before forcing the turn to end


class VadOnnx:
    """Standalone port of livekit-plugins-silero's OnnxModel wrapper (Apache-2.0),
    using the same silero_vad.onnx weights, without depending on livekit-agents.
    The onnxruntime session is shared/global; per-session recurrent state
    (context/rnn_state) is kept externally in SessionState so this is safe to
    call concurrently across browser sessions."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            SILERO_VAD_ONNX_PATH, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.sample_rate = sample_rate
        self.context_size = 64 if sample_rate == 16000 else 32
        self._sample_rate_nd = np.array(sample_rate, dtype=np.int64)

    def new_state(self):
        context = np.zeros((1, self.context_size), dtype=np.float32)
        rnn_state = np.zeros((2, 1, 128), dtype=np.float32)
        return context, rnn_state

    def prob(self, frame_f32: np.ndarray, context: np.ndarray, rnn_state: np.ndarray):
        input_buffer = np.concatenate([context, frame_f32[np.newaxis, :]], axis=1)
        out, new_rnn_state = self.session.run(
            None,
            {"input": input_buffer, "state": rnn_state, "sr": self._sample_rate_nd},
        )
        new_context = input_buffer[:, -self.context_size :]
        return float(out.item()), new_context, new_rnn_state


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


@dataclass
class SessionState:
    """Small, cheap-to-copy per-browser-session mutable state. Heavy models
    (CSM, whisper, EOU, VAD sessions) are module-level globals, shared and
    stateless across sessions."""

    vad_context: np.ndarray
    vad_rnn_state: np.ndarray
    voice_context: list  # list[Segment], starts with the pinned voice prompt
    chat_history: list = field(default_factory=lambda: [{"role": "system", "content": SYSTEM_PROMPT}])
    pcm_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    pre_buffer: list = field(default_factory=list)
    speaking: bool = False
    speech_frames: list = field(default_factory=list)
    speech_run: float = 0.0
    silence_run: float = 0.0
    grace_rounds_used: int = 0


def resample_int16(data_i16: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or data_i16.size == 0:
        return data_i16
    wav = torch.from_numpy(data_i16.astype(np.float32) / 32768.0)
    wav = torchaudio.functional.resample(wav, orig_sr, target_sr)
    return (wav.clamp(-1, 1) * 32767).to(torch.int16).numpy()


def transcribe(stt: WhisperModel, audio_f32: np.ndarray) -> str:
    segments, _ = stt.transcribe(audio_f32, language="en", vad_filter=False)
    return " ".join(seg.text.strip() for seg in segments).strip()


def get_llm_reply(history: list[dict]) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "messages": history, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


class Pipeline:
    """Holds the heavy, shared, load-once models and exposes the per-chunk
    processing function used by the Gradio streaming callback."""

    def __init__(self, device: str, speaker_name: str):
        print("Loading Silero VAD...")
        self.vad = VadOnnx()

        print("Loading LiveKit turn-detector (semantic end-of-turn model)...")
        self.eou = EOUModel()

        print("Loading faster-whisper (base.en)...")
        compute_type = "float16" if device == "cuda" else "int8"
        self.stt = WhisperModel("base.en", device=device, compute_type=compute_type)

        print(f"Loading CSM-1B on {device}...")
        self.generator = load_csm_1b(device=device)

        print(f"Loading voice prompt '{speaker_name}'...")
        prompt_path = hf_hub_download(repo_id="sesame/csm-1b", filename=f"prompts/{speaker_name}.wav")
        prompt_wav, prompt_sr = torchaudio.load(prompt_path)
        prompt_wav = torchaudio.functional.resample(
            prompt_wav.mean(0), orig_freq=prompt_sr, new_freq=self.generator.sample_rate
        )
        self.voice_prompt_segment = Segment(
            text=VOICE_PROMPTS[speaker_name], speaker=SPEAKER_ID, audio=prompt_wav
        )

    def new_session_state(self) -> SessionState:
        context, rnn_state = self.vad.new_state()
        return SessionState(
            vad_context=context,
            vad_rnn_state=rnn_state,
            voice_context=[self.voice_prompt_segment],
        )

    def process_chunk(self, chunk, sess: SessionState, chat_display: list):
        """Feeds one streamed audio chunk through VAD/turn-detection. When a
        full user turn is detected, runs STT -> LLM -> TTS and returns the
        reply audio + updated chat display. Otherwise leaves the audio output
        untouched (gr.skip()) rather than clearing it -- since this callback
        fires every ~100-200ms, returning None here would reset the Audio
        component back to empty right after a reply was just set, cutting
        playback off before the browser even starts it."""
        reply_audio = gr.skip()
        last_vad_prob = None

        if sess is None:
            # Session state hasn't finished initializing yet (demo.load() race);
            # nothing to do with this chunk.
            return sess, chat_display, reply_audio, "Session still initializing, one sec..."

        if chunk is None:
            return sess, chat_display, reply_audio, "Listening... (no audio received yet)"

        sr, data = chunk
        data = np.asarray(data)
        if data.ndim > 1:
            data = data.mean(axis=1)

        raw_dtype = str(data.dtype)
        raw_min, raw_max = (float(np.min(data)), float(np.max(data))) if data.size else (0.0, 0.0)

        if np.issubdtype(data.dtype, np.floating):
            # float chunks are typically already in [-1, 1]
            data = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            data = data.astype(np.int16)
        data = resample_int16(data, sr, SAMPLE_RATE)

        sess.pcm_buffer = np.concatenate([sess.pcm_buffer, data])
        chunk_peak = int(np.max(np.abs(data))) if data.size else 0

        print(
            f"[chunk] sr={sr} dtype={raw_dtype} shape={data.shape} "
            f"raw_range=({raw_min:.4f},{raw_max:.4f}) peak_i16={chunk_peak} "
            f"buffer_samples={len(sess.pcm_buffer)}"
        )

        status = None

        while len(sess.pcm_buffer) >= WINDOW_SAMPLES:
            frame_i16 = sess.pcm_buffer[:WINDOW_SAMPLES]
            sess.pcm_buffer = sess.pcm_buffer[WINDOW_SAMPLES:]
            frame_f32 = frame_i16.astype(np.float32) / 32768.0

            p, sess.vad_context, sess.vad_rnn_state = self.vad.prob(
                frame_f32, sess.vad_context, sess.vad_rnn_state
            )
            last_vad_prob = p

            if not sess.speaking:
                sess.pre_buffer.append(frame_i16)
                max_prefix_frames = max(1, int(PREFIX_PADDING_DURATION / FRAME_SEC))
                sess.pre_buffer = sess.pre_buffer[-max_prefix_frames:]

            is_active = p >= ACTIVATION_THRESHOLD or (sess.speaking and p > DEACTIVATION_THRESHOLD)

            if is_active:
                sess.silence_run = 0.0
                sess.speech_run += FRAME_SEC
                if not sess.speaking and sess.speech_run >= MIN_SPEECH_DURATION:
                    sess.speaking = True
                    sess.speech_frames = list(sess.pre_buffer)
                    print(f"[vad] speech START (p={p:.3f})")
                if sess.speaking:
                    sess.speech_frames.append(frame_i16)
                continue

            sess.speech_run = 0.0
            if not sess.speaking:
                continue

            sess.speech_frames.append(frame_i16)
            sess.silence_run += FRAME_SEC

            utterance_sec = len(sess.speech_frames) * FRAME_SEC
            if sess.silence_run < MIN_SILENCE_DURATION and utterance_sec < MAX_UTTERANCE_SEC:
                continue

            print(f"[vad] pause detected after {utterance_sec:.2f}s of speech, transcribing...")
            audio = np.concatenate(sess.speech_frames).astype(np.float32) / 32768.0
            text = transcribe(self.stt, audio)
            print(f"[stt] {text!r}")

            if not text:
                sess.speaking = False
                sess.speech_frames = []
                sess.silence_run = 0.0
                sess.grace_rounds_used = 0
                status = "Heard silence/noise, no speech recognized. Listening..."
                continue

            candidate_ctx = sess.chat_history + [{"role": "user", "content": text}]
            eou_prob = self.eou.probability(candidate_ctx)
            done_talking = eou_prob >= self.eou.threshold
            timed_out = utterance_sec >= MAX_UTTERANCE_SEC or sess.grace_rounds_used >= EOU_MAX_GRACE_ROUNDS
            print(f"[eou] prob={eou_prob:.4f} threshold={self.eou.threshold} done={done_talking or timed_out}")

            if not (done_talking or timed_out):
                sess.grace_rounds_used += 1
                sess.silence_run = 0.0
                status = f'Still listening ("{text}"...) eou_prob={eou_prob:.4f} < {self.eou.threshold}'
                continue

            # Finalize the turn.
            sess.speaking = False
            sess.speech_frames = []
            sess.silence_run = 0.0
            sess.grace_rounds_used = 0

            sess.chat_history.append({"role": "user", "content": text})
            print("[llm] querying Ollama...")
            reply_text = get_llm_reply(sess.chat_history)
            print(f"[llm] {reply_text!r}")
            sess.chat_history.append({"role": "assistant", "content": reply_text})

            print("[tts] generating CSM audio...")
            audio_out = self.generator.generate(
                text=reply_text,
                speaker=SPEAKER_ID,
                context=sess.voice_context,
                max_audio_length_ms=MAX_AUDIO_MS,
            )
            sess.voice_context.append(Segment(text=reply_text, speaker=SPEAKER_ID, audio=audio_out))
            sess.voice_context = [sess.voice_context[0]] + sess.voice_context[1:][-(MAX_CONTEXT_SEGMENTS - 1) :]

            chat_display = chat_display + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": reply_text},
            ]
            reply_audio = (self.generator.sample_rate, audio_out.cpu().numpy())
            print(f"[tts] done, {audio_out.shape[0] / self.generator.sample_rate:.2f}s of audio generated")
            status = f'You said: "{text}" -> Bot: "{reply_text}"'
            break

        if status is None:
            vad_str = f"{last_vad_prob:.3f}" if last_vad_prob is not None else "n/a"
            status = (
                f"Listening... speaking={sess.speaking} vad_prob={vad_str} "
                f"peak_i16={chunk_peak} silence_run={sess.silence_run:.2f}s "
                f"speech_run={sess.speech_run:.2f}s"
            )

        print(f"[status] {status}")
        return sess, chat_display, reply_audio, status


def ensure_self_signed_cert(cert_dir: str, hostnames: list[str]) -> tuple[str, str]:
    """Generates a temporary self-signed TLS cert/key (via openssl) if one
    doesn't already exist, valid for the given hostnames/IPs. Browsers will
    show a one-time "not secure" warning to click through -- that's expected
    for a self-signed cert, there's no CA behind it."""
    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "cert.pem")
    key_path = os.path.join(cert_dir, "key.pem")

    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path

    print(f"Generating a temporary self-signed TLS certificate at {cert_dir}...")
    san = ",".join(
        f"IP:{h}" if h.replace(".", "").isdigit() else f"DNS:{h}" for h in hostnames
    )
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:4096",
            "-sha256", "-days", "365", "-nodes",
            "-keyout", key_path, "-out", cert_path,
            "-subj", f"/CN={hostnames[0]}",
            "-addext", f"subjectAltName={san}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return cert_path, key_path


def build_ui(pipeline: Pipeline) -> gr.Blocks:
    with gr.Blocks(title="Local CSM Voice Chat") as demo:
        gr.Markdown(
            "# Local Voice Chat (CSM-1B + Ollama)\n"
            "Click the microphone to start listening. Speak naturally — turn-taking "
            "is detected automatically (Silero VAD + semantic end-of-turn model), no "
            "push-to-talk needed. Everything runs locally on this server."
        )
        chatbot = gr.Chatbot(label="Conversation", height=420, type="messages")
        status_box = gr.Textbox(label="Status", interactive=False, value="Click the mic to start.")
        reply_audio = gr.Audio(label="Reply", autoplay=True, visible=True)
        mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Mic (click to start/stop)")
        session_state = gr.State()

        def init_session():
            return pipeline.new_session_state()

        demo.load(fn=init_session, outputs=[session_state])

        mic.stream(
            fn=pipeline.process_chunk,
            inputs=[mic, session_state, chatbot],
            outputs=[session_state, chatbot, reply_audio, status_box],
            show_progress="hidden",
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address. 127.0.0.1 (default) is what tunnel tools (ngrok, cloudflared, ssh -L) "
        "expect to connect to on the same machine. Use 0.0.0.0 only if you need it reachable "
        "directly on the server's network interfaces.",
    )
    parser.add_argument(
        "--speaker", default="conversational_a", choices=list(VOICE_PROMPTS.keys()),
        help="Which built-in CSM-1B voice prompt to use (conversational_a=female, conversational_b=male).",
    )
    parser.add_argument(
        "--https", action="store_true",
        help="Serve over HTTPS using a temporary self-signed certificate (generated on first "
        "run, reused after). Browsers will show a one-time 'not secure' warning to click "
        "through since it's self-signed, not CA-issued.",
    )
    parser.add_argument(
        "--cert-dir", default=os.path.join(os.path.dirname(__file__), ".certs"),
        help="Where to store/look for the self-signed cert+key (only used with --https).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pipeline = Pipeline(device=device, speaker_name=args.speaker)
    demo = build_ui(pipeline)
    demo.queue()

    ssl_certfile = ssl_keyfile = None
    if args.https:
        hostnames = [args.host]
        if args.host in ("0.0.0.0", "127.0.0.1"):
            hostnames = ["127.0.0.1", "localhost"]
        ssl_certfile, ssl_keyfile = ensure_self_signed_cert(args.cert_dir, hostnames)
        scheme = "https"
    else:
        scheme = "http"

    if args.https:
        print(f"\n(Your browser will warn about an untrusted certificate -- that's expected "
              f"for a self-signed cert, proceed anyway. URL: {scheme}://{args.host}:{args.port})\n")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_verify=False,  # self-signed: nothing to verify against a CA
    )


if __name__ == "__main__":
    main()
