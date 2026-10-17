#!/usr/bin/env bash
# Thin wrapper: run from the repo root with the project venv. Logs are tee'd to logs/<model>.run.log.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
.venv/bin/python -u run.py "$@" 2>&1 | tee -a "logs/${1:-run}.run.log"
