#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/baseline/NavReg/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${CONDA_EXE:-conda}" run --no-capture-output -n "${HERO_CONDA_ENV:-Hero}" \
  python -B -m navreg run "$@"
