.PHONY: help install dev test lint format clean docker-build docker-up docker-down deploy

help:
	@echo "Kwami AI API - Development Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install       - Install dependencies"
	@echo "  make dev           - Run API server (dev mode)"
	@echo "  make test          - Run tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          - Run linter"
	@echo "  make format        - Format code"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-up     - Start container"
	@echo "  make docker-down   - Stop container"
	@echo ""
	@echo "Deploy:"
	@echo "  make deploy        - Deploy to Fly.io (fly deploy)"

# =============================================================================
# Development
# =============================================================================

install:
	uv sync

dev:
	uv run python -m src.main

test:
	uv run python -m pytest tests/ -v

# =============================================================================
# Code Quality
# =============================================================================

lint:
	uv run ruff check .

format:
	uv run ruff format . && uv run ruff check --fix .

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build -t kwami-lk-api .

docker-up:
	docker run -d --name kwami-lk-api -p 8080:8080 --env-file .env kwami-lk-api

docker-down:
	docker stop kwami-lk-api && docker rm kwami-lk-api

# =============================================================================
# Deploy
# =============================================================================

deploy:
	fly deploy

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf .venv __pycache__ **/__pycache__
	rm -rf .pytest_cache .ruff_cache
