#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/setup_wsl_conda_mem_v2.sh "$@"
