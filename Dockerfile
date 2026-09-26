FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

WORKDIR /usr/src/app
# The venv lives outside /usr/src/app, which docker compose mounts over with the working copy.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# dependencies first, so that code changes don't invalidate this layer
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --python-preference only-system
COPY . .
RUN uv sync --frozen --python-preference only-system
