#!/usr/bin/env sh

set -o pipefail

set +e

uv sync

echo "=== Ruff Linting ==="
uv run ruff check .

echo "=== Mypy Type Checking ==="
uv run mypy src

echo "=== Pytest Unit Tests ==="
uv run pytest -q

wait