#!/bin/sh
# One-time setup for the local DA3 depth server. Usage:  npm run server:setup   (or: sh server/setup.sh)
#   VENV=<dir>  override the environment location (default server/.venv)
set -e
cd "$(dirname "$0")"
VENV="${VENV:-.venv}"
# Depth-Anything-3 commit this server was developed and tested against.
DA3_REF="3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: '$1' is required but was not found. $2" >&2
    exit 1
  fi
}
need uv "Install it with:  brew install uv   (or see https://docs.astral.sh/uv/)"
need ffmpeg "Install it with:  brew install ffmpeg"
need git "Needed to fetch Depth-Anything-3 from GitHub."

echo "==> Creating Python 3.12 environment in server/$VENV (uv downloads Python 3.12 if missing; PyTorch has no 3.14 wheels)"
uv venv --python 3.12 "$VENV"

echo "==> Installing dependencies (PyTorch is a large download the first time)"
uv pip install --python "$VENV/bin/python" -r requirements.txt
# --no-deps: the package pins xformers, which is CUDA-only and fails to build on macOS; its other deps are in requirements.txt.
uv pip install --python "$VENV/bin/python" --no-deps "git+https://github.com/ByteDance-Seed/Depth-Anything-3@$DA3_REF"

echo "==> Downloading the DA3-BASE model weights (~0.5 GB, cached in ~/.cache/huggingface)"
"$VENV/bin/python" prefetch_weights.py

echo
echo "Setup complete. Start the viewer with:  npm run serve   (from the repository root)"
