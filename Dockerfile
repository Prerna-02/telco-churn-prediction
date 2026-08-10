FROM python:3.11-slim

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1
# Keeps Python from buffering stdout and stderr, so container logs appear
# immediately in CloudWatch / Render rather than being lost on a crash.
ENV PYTHONUNBUFFERED=1
# So that `from src.predictor import ...` resolves no matter how the process
# is launched (uvicorn from /app, or streamlit from /app/src).
ENV PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Run as a non-root user — ECS task definitions and Render both prefer this.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

# APP_MODE=ui   -> Streamlit dashboard (used for the public demo)
# APP_MODE=api  -> FastAPI JSON service (used for the ECS Fargate service)
ENV APP_MODE=ui
# Render injects its own PORT; this is the fallback for local/ECS runs.
ENV PORT=8000

EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]
