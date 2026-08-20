#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: scripts/bootstrap_vm.sh [--output-dir PATH]"
  echo "Sync and validate the pinned Linux GPU VM runtime."
  exit 0
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "bootstrap_vm requires Linux; current host is $(uname -s)." >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install the pinned project tool before bootstrap." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export GIT_LFS_SKIP_SMUDGE=1
export GIT_CONFIG_COUNT=3
export GIT_CONFIG_KEY_0=filter.lfs.smudge
export GIT_CONFIG_VALUE_0=cat
export GIT_CONFIG_KEY_1=filter.lfs.process
export GIT_CONFIG_VALUE_1=""
export GIT_CONFIG_KEY_2=filter.lfs.required
export GIT_CONFIG_VALUE_2=false
export WANDB_MODE=disabled
export WANDB_DISABLED=true

uv sync --frozen --extra gpu
exec uv run python scripts/bootstrap_runtime.py \
  --config configs/platform/gpu_vm.yaml "$@"
