# Multi-stage build für optimale Image-Größe
FROM python:3.11-slim AS builder

# UV installieren
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Arbeitsverzeichnis erstellen
WORKDIR /app

# Projekt-Dateien kopieren
COPY pyproject.toml uv.lock ./

# Virtuelle Umgebung erstellen und Abhängigkeiten installieren
RUN uv sync --frozen --no-cache

# Production Stage
FROM python:3.11-slim

# System-Abhängigkeiten installieren
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis erstellen
WORKDIR /app

# UV installieren (für Runtime)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Virtuelle Umgebung aus Builder-Stage kopieren
COPY --from=builder /app/.venv /app/.venv

# Projekt-Code kopieren
COPY . .

# Start-Skript ausführbar machen
RUN chmod +x docker-entrypoint.sh

# Virtuelle Umgebung aktivieren durch PATH Update
ENV PATH="/app/.venv/bin:$PATH"

# Verzeichnisse für Daten erstellen
RUN mkdir -p /app/models /app/scenarios /app/data

# Port freigeben
EXPOSE 7860

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

# Container als non-root User ausführen
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Startkommando
CMD ["./docker-entrypoint.sh"]
