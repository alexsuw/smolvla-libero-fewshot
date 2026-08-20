#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: scripts/bootstrap_vm.sh"
  echo "Bootstrap the pinned Linux GPU VM environment (implemented in M1)."
  exit 0
fi

echo "bootstrap_vm is intentionally unavailable until M1 runtime pins are validated." >&2
exit 2
