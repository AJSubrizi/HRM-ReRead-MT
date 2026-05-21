#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

echo "Environment ready. Copy .env.example to .env and set DEEPSEEK_API_KEY before augmentation."
