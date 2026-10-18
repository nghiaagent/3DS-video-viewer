#!/bin/sh
# Creates server/.venv (Python 3.12; torch has no 3.14 wheels) with Depth-Anything-3 minus xformers.
set -e
cd "$(dirname "$0")"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps "git+https://github.com/ByteDance-Seed/Depth-Anything-3"
