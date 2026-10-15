# WhatsApp AI Assistant - production image.
# Slim Python 3.12, dependencies installed as root, the app runs as an unprivileged user.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

# Install the package and its runtime dependencies (no dev tools in the image).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data ./data
RUN mkdir -p /app/var && chown app:app /app/var

ENV PORT=8000 \
    WA_ENV=prod \
    WA_DATA_DIR=/app/data \
    WA_DB_PATH=/app/var/wa_assistant.db

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT', '8000'), timeout=4)"

# Render (and most PaaS) inject $PORT; default 8000 locally. No access log: Meta's
# verification request carries the verify token in its query string.
CMD ["sh", "-c", "exec uvicorn wa_assistant.app:create_app --factory --host 0.0.0.0 --port \"${PORT:-8000}\" --no-access-log"]
