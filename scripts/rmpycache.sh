#!/usr/bin/env bash

# Remove all __pycache__ directories recursively
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true