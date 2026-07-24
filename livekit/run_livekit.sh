#!/usr/bin/env bash
#
# Starts both the voice agent and the token-server in the background (for
# when you only have one terminal/SSH session available), streams both
# logs interleaved to this terminal, and cleanly stops both on Ctrl+C.
#
# Assumes ./setup_livekit.sh has already been run.
#
# Usage:
#   ./run_livekit.sh              # dev mode (connects to the LiveKit room)
#   ./run_livekit.sh console      # agent runs in console mode instead (no token-server)
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$ROOT_DIR/agent"
VENV_DIR="$ROOT_DIR/.venv"
TOKEN_SERVER_DIR="$ROOT_DIR/token-server"
LOG_DIR="$ROOT_DIR/logs"
AGENT_MODE="${1:-dev}"

mkdir -p "$LOG_DIR"

PIDS=()

# Recursively signals a PID and all of its descendants (catches subprocesses
# the agent spawns, e.g. its multiprocessing job workers, which a plain
# `kill $pid` on the top-level wrapper process would leave running).
kill_tree() {
  local pid="$1" sig="$2" child
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    kill_tree "$child" "$sig"
  done
  kill -"$sig" "$pid" 2>/dev/null || true
}

cleanup() {
  echo
  echo "Stopping..."
  for pid in "${PIDS[@]}"; do
    kill_tree "$pid" TERM
  done
  sleep 2
  for pid in "${PIDS[@]}"; do
    kill_tree "$pid" KILL
  done
  wait 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

echo "==> Starting voice agent ($AGENT_MODE mode), logging to $LOG_DIR/agent.log"
(
  cd "$AGENT_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  exec python src/agent.py "$AGENT_MODE"
) > "$LOG_DIR/agent.log" 2>&1 &
PIDS+=("$!")

if [ "$AGENT_MODE" != "console" ]; then
  echo "==> Starting token-server, logging to $LOG_DIR/token-server.log"
  (
    cd "$TOKEN_SERVER_DIR"
    exec npm start
  ) > "$LOG_DIR/token-server.log" 2>&1 &
  PIDS+=("$!")
fi

sleep 2
echo
echo "Both services starting up (PIDs: ${PIDS[*]}). Tailing logs -- Ctrl+C stops everything."
echo "----------------------------------------------------------------------"

if [ "$AGENT_MODE" != "console" ]; then
  tail -f "$LOG_DIR/agent.log" "$LOG_DIR/token-server.log" &
else
  tail -f "$LOG_DIR/agent.log" &
fi
TAIL_PID=$!
PIDS+=("$TAIL_PID")

wait
