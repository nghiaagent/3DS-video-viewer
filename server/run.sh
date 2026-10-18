#!/bin/sh
# Builds the viewer and starts the local DA3 depth server. Usage:  npm run serve   (or: sh server/run.sh)
cd "$(dirname "$0")/.."
if [ ! -x server/.venv/bin/python ]; then
  echo "The depth server is not set up yet. Run this once first:  npm run server:setup" >&2
  exit 1
fi
npm run build || exit 1
exec server/.venv/bin/python server/serve.py
