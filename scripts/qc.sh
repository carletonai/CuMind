#!/usr/bin/env sh

set +e

uv sync

echo "=== Ruff Linting ==="
uv run ruff check .

echo "=== Ty Type Checking ==="
uv run ty check .

echo "=== Mypy Type Checking ==="
uv run mypy src

echo "=== Pytest Unit Tests ==="
uv run pytest -q