#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: scripts/bootstrap_colab.sh"
  echo "Bootstrap the pinned Colab smoke environment (implemented in M1)."
  exit 0
fi

echo "bootstrap_colab is intentionally unavailable until M1 runtime pins are validated." >&2
exit 2
