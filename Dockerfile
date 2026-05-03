FROM ghcr.io/astral-sh/uv:0.11.8-python3.14-trixie-slim AS builder

ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

FROM python:3.14-slim AS stage

ENV APP_HOME=/app
ENV HOME=/home/app
ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --home-dir "$HOME" --shell /bin/bash app

COPY --from=builder /app/.venv /app/.venv

WORKDIR $APP_HOME

COPY . .

RUN chown -R app:app .

USER app
