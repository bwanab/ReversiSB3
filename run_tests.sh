#!/bin/bash
# Run tests with proper PYTHONPATH setup (project root = this script's directory)
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT" && PYTHONPATH="$ROOT" uv run python tests/run_tests.py "$@"
