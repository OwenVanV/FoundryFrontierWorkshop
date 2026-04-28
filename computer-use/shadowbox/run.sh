#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$SCRIPT_DIR/.venv"

# Ensure uv is installed
if ! command -v uv &>/dev/null; then
    echo "uv not found -- installing via Homebrew..."
    brew install uv
fi

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    uv venv "$VENV_DIR"
fi

# Sync / install dependencies
echo "Installing dependencies..."
cd "$SCRIPT_DIR"
uv pip install -e . --quiet

# Run from the parent so Python sees "shadowbox" as a package
cd "$PARENT_DIR"

echo ""
echo "==========================================="
echo "  Shadowbox - Screen Record > Analyze > CUA"
echo "==========================================="
echo ""

"$VENV_DIR/bin/python" -m shadowbox "$@"
