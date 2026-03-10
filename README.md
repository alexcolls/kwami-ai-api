# Kwami LiveKit API

Backend API for **Kwami** voice agents: LiveKit token issuance, model and voice catalogs, Zep memory operations, and credits (Stripe). Used by [Kwami App](https://github.com/kwami-labs/kwami-app) and the Kwami LiveKit agent.

## Features

- **LiveKit tokens** — Issue JWT tokens for app/agent participants; agent dispatch is handled by LiveKit Cloud
- **Models** — STT, LLM, TTS model lists derived from LiveKit plugins (OpenAI, Anthropic, Deepgram, ElevenLabs, etc.)
- **Voices & languages** — Voice and language catalogs for the app
- **Memory** — Zep-backed memory endpoints (sessions, search, graph operations)
- **Credits** — Balance, usage, and Stripe checkout for credit purchases
- **Auth** — Supabase JWT verification for protected routes

## Prerequisites

- **Python** 3.11+
- **uv** (recommended) or pip

## Setup

```bash
# Install dependencies (uv)
uv sync

# Or with pip
pip install -e .

# Configure environment
cp .env.sample .env
# Edit .env with your keys (see Environment variables)
```

## Running

```bash
# Development (reload on change)
make dev
# or: uv run python -m src.main

# Production (e.g. in Docker)
make docker-build
make docker-up
```

API listens on `API_HOST:API_PORT` (default `0.0.0.0:8080`). OpenAPI docs at `/docs` and `/redoc` when `ENABLE_DOCS=true`.

## API overview

| Prefix    | Description                |
|----------|----------------------------|
| `/`      | Health                     |
| `/token` | LiveKit token generation   |
| `/memory`| Zep memory (sessions, etc.)|
| `/models`| STT/LLM/TTS model lists    |
| `/voices`| Voice catalog              |
| `/languages` | Language catalog      |
| `/credits`   | Balance, usage, Stripe |

The **token** endpoint expects a POST body with `roomName`, optional `participantName` / `participantIdentity`, permissions, and optional `kwamiId`. It returns a JWT for connecting to LiveKit.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LIVEKIT_URL` | Yes | LiveKit WebSocket URL (e.g. `wss://your-project.livekit.cloud`) |
| `LIVEKIT_API_KEY` | Yes | LiveKit API key |
| `LIVEKIT_API_SECRET` | Yes | LiveKit API secret |
| `ZEP_API_KEY` | For memory | Zep Cloud API key |
| `SUPABASE_URL` | For auth | Supabase project URL (JWKS verification) |
| `SUPABASE_SECRET_KEY` | For credits/DB | Supabase service role key |
| `STRIPE_SECRET_KEY` | For credits | Stripe secret key |
| `STRIPE_WEBHOOK_SECRET` | For credits | Stripe webhook signing secret |
| `STRIPE_PUBLISHABLE_KEY` | Optional | Stripe publishable key |
| `KWAMI_API_KEY` | For agent | Shared secret for agent usage reporting (X-API-Key) |
| `CORS_ORIGINS` | No | Comma-separated origins (default `*` in dev) |
| `API_HOST` / `API_PORT` | No | Bind address and port (default `0.0.0.0:8080`) |
| `APP_ENV` | No | `development` \| `staging` \| `production` |
| `ENABLE_DOCS` | No | Set to `true` to expose `/docs` and `/redoc` in production |

See `.env.sample` for a full list and comments.

## Commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies (`uv sync`) |
| `make dev` | Run API in dev mode |
| `make test` | Run tests |
| `make lint` | Run Ruff linter |
| `make format` | Format and fix with Ruff |
| `make docker-build` | Build Docker image |
| `make docker-up` / `make docker-down` | Run or stop container |

## Project structure

```
src/
├── main.py           # FastAPI app, CORS, routes
├── api/
│   ├── routes/       # health, token, memory, models, voices, languages, credits
│   └── deps.py       # Auth dependencies
├── core/
│   ├── config.py     # Pydantic settings
│   └── security.py    # JWT / auth helpers
├── services/
│   ├── livekit.py    # Token creation
│   ├── models.py     # Model list from LiveKit plugins
│   ├── voices.py     # Voice catalog
│   ├── languages.py  # Language catalog
│   ├── credits.py    # Balance, usage, Stripe
│   └── ...
config/               # LiveKit plugin YAML (inference, voices, languages)
migrations/           # SQL migrations (credits, user kwamis)
tests/
```

## Deployment

- **Fly.io** — `fly.toml` is included. Set secrets with `fly secrets set` and deploy with `fly deploy`.
- **Docker** — Use `Dockerfile` and set env via `--env-file` or environment.

## License

Apache-2.0. See [LICENSE](LICENSE).
