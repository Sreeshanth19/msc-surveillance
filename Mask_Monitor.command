#!/bin/bash
# Double-click to launch the Face Mask & Social Distance Monitor.
cd "$(dirname "$0")"
if [ -d ".venv-arm" ]; then
  source .venv-arm/bin/activate
elif [ -d ".venv" ]; then
  source .venv/bin/activate
fi
python -m scripts.desktop_app
