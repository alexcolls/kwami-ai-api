.PHONY: help install dev test lint format clean docker-build docker-up docker-down

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
	docker build -t kwami-ai-api .

docker-up:
	docker run -d --name kwami-ai-api -p 8080:8080 --env-file .env kwami-ai-api

docker-down:
	docker stop kwami-ai-api && docker rm kwami-ai-api

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf .venv __pycache__ **/__pycache__
	rm -rf .pytest_cache .ruff_cache
