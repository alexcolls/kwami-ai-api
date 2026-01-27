# Kwami LK API - Token Endpoint Server

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8080

ENV APP_ENV=production
ENV API_HOST=0.0.0.0
ENV API_PORT=8080

CMD ["uv", "run", "kwami-api"]
