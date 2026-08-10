#!/bin/sh
set -e

PORT="${PORT:-8000}"
APP_MODE="${APP_MODE:-ui}"

case "$APP_MODE" in
  api)
    echo "Starting FastAPI on port $PORT"
    exec uvicorn src.app:app --host 0.0.0.0 --port "$PORT"
    ;;
  ui)
    # Deliberately no --server.address: Streamlit already binds 0.0.0.0 by
    # default, and setting it explicitly makes the startup banner advertise
    # http://0.0.0.0:$PORT, which no browser can open.
    echo "Starting Streamlit on port $PORT"
    exec streamlit run src/streamlit_app.py \
      --server.port "$PORT" \
      --server.headless true \
      --browser.gatherUsageStats false
    ;;
  *)
    echo "Unknown APP_MODE '$APP_MODE' (expected 'api' or 'ui')" >&2
    exit 1
    ;;
esac
