#!/usr/bin/env sh

set -e

if [ -d src/cumind ]; then
    uv run pydeps --max-bacon 2 --show-deps --cluster --rankdir TB -o CuMind.svg -T svg src/cumind
else
    echo "src/cumind not found"
    exit 1
fi