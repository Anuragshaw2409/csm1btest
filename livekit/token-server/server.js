import "dotenv/config";
import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import cors from "cors";
import { AccessToken, RoomAgentDispatch, RoomConfiguration } from "livekit-server-sdk";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const {
  LIVEKIT_URL,
  LIVEKIT_API_KEY,
  LIVEKIT_API_SECRET,
  PORT = 8787,
  ALLOWED_ORIGIN = "*",
  AGENT_NAME = "Anurag",
  FRONTEND_DIST = "../frontend/dist",
} = process.env;

if (!LIVEKIT_URL || !LIVEKIT_API_KEY || !LIVEKIT_API_SECRET) {
  console.error("Missing LIVEKIT_URL, LIVEKIT_API_KEY, or LIVEKIT_API_SECRET in environment");
  process.exit(1);
}

const app = express();
app.use(cors({ origin: ALLOWED_ORIGIN }));

app.get("/health", (_req, res) => res.json({ ok: true }));

// Issues a fresh room + access token so the frontend can join and talk to the agent.
app.get("/token", async (req, res) => {
  try {
    const identity = req.query.identity?.toString() || `user-${Date.now()}`;
    const room = req.query.room?.toString() || `voice-agent-${Date.now()}`;

    const at = new AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET, {
      identity,
      ttl: "10m",
    });
    at.addGrant({
      room,
      roomJoin: true,
      canPublish: true,
      canSubscribe: true,
      canPublishData: true,
    });
    // Explicit dispatch: this agent uses agent_name="Anurag" in @server.rtc_session,
    // so it only joins rooms that request it by name.
    at.roomConfig = new RoomConfiguration({
      agents: [new RoomAgentDispatch({ agentName: AGENT_NAME })],
    });

    const token = await at.toJwt();
    res.json({ token, url: LIVEKIT_URL, room, identity });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to create token" });
  }
});

// Serve the built React frontend from the same origin/port as the API, so a
// single ngrok tunnel covers both the UI and the token endpoint (no CORS to manage).
const distPath = path.resolve(__dirname, FRONTEND_DIST);
app.use(express.static(distPath));
app.get("*", (req, res, next) => {
  if (req.path.startsWith("/token") || req.path.startsWith("/health")) return next();
  res.sendFile(path.join(distPath, "index.html"));
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
