# Multi-stage Dockerfile для Sales Agent сервисов
# Использует uv для быстрой установки зависимостей с кешированием

# Stage 1: Builder - установка зависимостей
FROM python:3.12-slim as builder

# Установка системных зависимостей для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка uv из официального образа
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Установка рабочей директории
WORKDIR /app

# Настройка uv для копирования (для поддержки cache mounts)
ENV UV_LINK_MODE=copy

# Копирование файлов зависимостей
COPY pyproject.toml uv.lock ./

# Установка зависимостей с кешированием
# Используем --no-install-project для оптимизации кеша слоев
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Копирование исходного кода
COPY src/ ./src/
COPY scripts/ ./scripts/

# Установка проекта
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Stage 2: Runtime - минимальный образ
FROM python:3.12-slim

# Установка системных зависимостей для runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание непривилегированного пользователя
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Копирование uv из builder (для возможного использования в runtime)
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

# Копирование установленных зависимостей из builder
COPY --from=builder /app/.venv /app/.venv

# Копирование исходного кода
WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Переключение на непривилегированного пользователя
USER appuser

# Установка PATH для использования .venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check (порт можно переопределить через ARG)
ARG PORT=8001
ENV PORT=${PORT}
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose порт
EXPOSE ${PORT}

# Точка входа (переопределяется в docker-compose или при запуске)
# По умолчанию запускает ASR Gateway
CMD ["python", "-m", "src.asr_gateway.main"]
