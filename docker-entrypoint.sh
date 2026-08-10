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
    # Streamlit's own banner prints http://0.0.0.0:$PORT, which is the bind
    # address and is NOT browsable. Print the real one first.
    echo "Starting Streamlit -> open http://localhost:$PORT in your browser"
    echo "(ignore the 0.0.0.0 URL Streamlit prints below - it will not load)"
    exec streamlit run src/streamlit_app.py \
      --server.port "$PORT" \
      --server.address 0.0.0.0 \
      --server.headless true \
      --browser.gatherUsageStats false
    ;;
  *)
    echo "Unknown APP_MODE '$APP_MODE' (expected 'api' or 'ui')" >&2
    exit 1
    ;;
esac
