#!/usr/bin/env bash
# Launches the CSM TTS server, adding --ssl-keyfile/--ssl-certfile only if
# CSM_TTS_TLS_KEYFILE/CSM_TTS_TLS_CERTFILE are set (see Dockerfile).
set -euo pipefail

TLS_ARGS=()
if [ -n "${CSM_TTS_TLS_CERTFILE:-}" ] && [ -n "${CSM_TTS_TLS_KEYFILE:-}" ]; then
  TLS_ARGS=(--ssl-certfile "$CSM_TTS_TLS_CERTFILE" --ssl-keyfile "$CSM_TTS_TLS_KEYFILE")
fi

exec uvicorn server.tts_server:app --host 0.0.0.0 --port 8000 "${TLS_ARGS[@]}"
