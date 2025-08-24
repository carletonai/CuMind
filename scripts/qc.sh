#!/usr/bin/env sh

set +e

uv sync

echo "=== Ruff Linting ==="
uv run python -m ruff check .

echo "=== Ty Type Checking ==="
uv run python -m ty check .

echo "=== Mypy Type Checking ==="
uv run python -m mypy src

echo "=== Pytest Unit Tests ==="
uv run python -m pytest -q