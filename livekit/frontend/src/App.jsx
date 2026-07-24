import { useCallback, useEffect, useRef, useState } from "react";
import { Room, RoomEvent, Track } from "livekit-client";

const STAGE_ORDER = ["eou", "stt", "llm", "tts"];
const STAGE_LABEL = { eou: "Turn detection", stt: "Speech to text", llm: "LLM", tts: "Text to speech" };
const decoder = new TextDecoder();

function primaryMs(stage, data) {
  if (stage === "eou") return data.end_of_utterance_delay_ms;
  if (stage === "stt") return data.duration_ms;
  if (stage === "llm") return data.ttft_ms;
  if (stage === "tts") return data.ttfb_ms;
  return null;
}

// In production the frontend is served by the same Express process as /token
// (see token-server/server.js), so this defaults to a same-origin relative path.
// Only `vite dev` (separate port) needs an explicit VITE_TOKEN_SERVER_URL.
const TOKEN_SERVER_URL =
  import.meta.env.VITE_TOKEN_SERVER_URL ?? (import.meta.env.DEV ? "http://localhost:8787" : "");

const STATUS_COPY = {
  idle: "Ready to call",
  connecting: "Connecting…",
  connected: "Listening",
  agent: "Agent speaking",
  error: "Call failed",
};

export default function App() {
  const [callState, setCallState] = useState("idle");
  const [muted, setMuted] = useState(false);
  const [error, setError] = useState(null);
  const [roomName, setRoomName] = useState(null);
  const [localLevel, setLocalLevel] = useState(0);
  const [agentLevel, setAgentLevel] = useState(0);
  const [agentSpeaking, setAgentSpeaking] = useState(false);
  const [latency, setLatency] = useState({});

  const roomRef = useRef(null);
  const audioElRef = useRef(null);
  const audioCtxRef = useRef(null);
  const rafRef = useRef(null);
  const localAnalyserRef = useRef(null);
  const agentAnalyserRef = useRef(null);

  const stopVisualizer = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
    localAnalyserRef.current = null;
    agentAnalyserRef.current = null;
    setLocalLevel(0);
    setAgentLevel(0);
  }, []);

  const runVisualizerLoop = useCallback(() => {
    const readLevel = (analyser) => {
      if (!analyser) return 0;
      const data = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) {
        const v = (data[i] - 128) / 128;
        sum += v * v;
      }
      return Math.min(1, Math.sqrt(sum / data.length) * 4);
    };

    const tick = () => {
      setLocalLevel(readLevel(localAnalyserRef.current));
      const aLevel = readLevel(agentAnalyserRef.current);
      setAgentLevel(aLevel);
      setAgentSpeaking(aLevel > 0.04);
      rafRef.current = requestAnimationFrame(tick);
    };
    tick();
  }, []);

  const attachAnalyser = useCallback((mediaStream, target) => {
    if (!audioCtxRef.current) {
      audioCtxRef.current = new AudioContext();
    }
    const ctx = audioCtxRef.current;
    const source = ctx.createMediaStreamSource(mediaStream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    if (target === "local") localAnalyserRef.current = analyser;
    if (target === "agent") agentAnalyserRef.current = analyser;
  }, []);

  const disconnect = useCallback(async () => {
    if (roomRef.current) {
      await roomRef.current.disconnect();
      roomRef.current = null;
    }
    stopVisualizer();
    setCallState("idle");
    setRoomName(null);
    setMuted(false);
    setLatency({});
  }, [stopVisualizer]);

  const connect = useCallback(async () => {
    setError(null);
    setCallState("connecting");
    try {
      const identity = `caller-${Math.random().toString(36).slice(2, 8)}`;
      const res = await fetch(
        `${TOKEN_SERVER_URL}/token?identity=${encodeURIComponent(identity)}`
      );
      if (!res.ok) throw new Error("Could not reach the token server");
      const { token, url, room: joinedRoomName } = await res.json();

      const room = new Room({ adaptiveStream: true, dynacast: true });
      roomRef.current = room;

      room.on(RoomEvent.Disconnected, () => {
        stopVisualizer();
        setCallState("idle");
        setRoomName(null);
        setLatency({});
      });

      room.on(RoomEvent.DataReceived, (payload, _participant, _kind, topic) => {
        if (topic !== "latency_metrics") return;
        try {
          const data = JSON.parse(decoder.decode(payload));
          if (!data.stage) return;
          setLatency((prev) => ({ ...prev, [data.stage]: data }));
        } catch (err) {
          console.error("Failed to parse latency metrics", err);
        }
      });

      room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
        if (track.kind === Track.Kind.Audio && participant.identity !== room.localParticipant.identity) {
          const el = track.attach();
          el.autoplay = true;
          document.body.appendChild(el);
          audioElRef.current = el;
          attachAnalyser(new MediaStream([track.mediaStreamTrack]), "agent");
        }
      });

      room.on(RoomEvent.TrackUnsubscribed, (track) => {
        track.detach().forEach((el) => el.remove());
      });

      await room.connect(url, token);
      await room.localParticipant.setMicrophoneEnabled(true);

      const micPub = room.localParticipant.getTrackPublication(Track.Source.Microphone);
      if (micPub?.track?.mediaStreamTrack) {
        attachAnalyser(new MediaStream([micPub.track.mediaStreamTrack]), "local");
      }

      runVisualizerLoop();
      setRoomName(joinedRoomName);
      setCallState("connected");
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong starting the call");
      setCallState("error");
      if (roomRef.current) {
        await roomRef.current.disconnect();
        roomRef.current = null;
      }
      stopVisualizer();
    }
  }, [attachAnalyser, runVisualizerLoop, stopVisualizer]);

  const toggleMute = useCallback(async () => {
    if (!roomRef.current) return;
    const next = !muted;
    await roomRef.current.localParticipant.setMicrophoneEnabled(!next);
    setMuted(next);
  }, [muted]);

  useEffect(() => {
    return () => {
      if (roomRef.current) roomRef.current.disconnect();
      stopVisualizer();
    };
  }, [stopVisualizer]);

  const isActive = callState === "connected";
  const statusText = isActive && agentSpeaking ? STATUS_COPY.agent : STATUS_COPY[callState];
  const ringLevel = isActive ? (agentSpeaking ? agentLevel : localLevel) : 0;
  const ringColor = agentSpeaking ? "var(--listen)" : "var(--live)";

  return (
    <div className="page">
      <div className="card">
        <div className="eyebrow">
          <span className={`dot ${isActive ? "live" : ""}`} />
          <span>CSM local voice line</span>
        </div>
        <h1>Talk to your agent</h1>

        <div className="orb-wrap">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="orb-ring"
              style={{
                width: 100 + i * 40,
                height: 100 + i * 40,
                transform: `scale(${1 + ringLevel * (0.12 + i * 0.05)})`,
                borderColor: isActive ? ringColor : "var(--line)",
                opacity: isActive ? 0.6 - i * 0.15 : 0.3,
              }}
            />
          ))}
          <div
            className={`orb-core ${agentSpeaking ? "agent-speaking" : ""}`}
            style={{
              boxShadow: isActive
                ? `0 0 ${20 + ringLevel * 60}px ${ringLevel * 8}px ${
                    agentSpeaking ? "rgba(87,217,163,0.35)" : "rgba(255,106,61,0.3)"
                  }`
                : "none",
            }}
          />
        </div>

        <div className="status-row">
          <div className="status-label">
            <span className="accent">{statusText}</span>
          </div>
        </div>

        <div className="controls">
          <button
            className={`btn-call ${isActive ? "end" : ""}`}
            onClick={isActive ? disconnect : connect}
            disabled={callState === "connecting"}
          >
            {isActive ? "End call" : callState === "connecting" ? "Connecting…" : "Start call"}
          </button>
          <button
            className={`btn-mute ${muted ? "muted" : ""}`}
            onClick={toggleMute}
            disabled={!isActive}
            aria-label={muted ? "Unmute microphone" : "Mute microphone"}
            title={muted ? "Unmute" : "Mute"}
          >
            {muted ? "🔇" : "🎙"}
          </button>
        </div>

        {error && <div className="error-banner mono">{error}</div>}

        {isActive && (
          <div className="latency-panel">
            <div className="latency-title">Pipeline latency</div>
            {STAGE_ORDER.map((stage) => {
              const data = latency[stage];
              const ms = data ? primaryMs(stage, data) : null;
              return (
                <div className="latency-row" key={stage}>
                  <span className="latency-label">{STAGE_LABEL[stage]}</span>
                  <span className={`latency-value mono ${data ? "seen" : ""}`}>
                    {ms != null ? `${ms} ms` : "—"}
                  </span>
                </div>
              );
            })}
          </div>
        )}

        <div className="meta mono">
          <span>room: {roomName || "—"}</span>
          <span>{TOKEN_SERVER_URL ? TOKEN_SERVER_URL.replace(/^https?:\/\//, "") : "same origin"}</span>
        </div>
      </div>
    </div>
  );
}
