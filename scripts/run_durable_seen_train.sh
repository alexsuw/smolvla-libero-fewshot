#!/usr/bin/env bash
# Resume-safe seen training launcher for a crashy host.
# Paths come from flags and VLA_* environment variables, never from hostnames.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_durable_seen_train.sh --config PATH --output-dir PATH [--log-freq N]

Requires a sourced runtime.env (WANDB off, HF/TORCH caches) plus:
  VLA_DATASETS_DIR   pinned dataset revision root parent
  PATH including uv

Run this under tmux, not an IDE terminal. The same command is idempotent:
  - missing output-dir → fresh start
  - completed manifest → exit 0
  - complete checkpoint → --resume-from that directory
  - stale run.lock whose PID is dead → removed
  - existing run without a complete checkpoint → refuse (no overwrite)
EOF
}

CONFIG=""
OUTPUT_DIR=""
LOG_FREQ="50"
PROFILE="full"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --config)
      CONFIG="${2:?}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?}"
      shift 2
      ;;
    --log-freq)
      LOG_FREQ="${2:?}"
      shift 2
      ;;
    --profile)
      PROFILE="${2:?}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CONFIG" || -z "$OUTPUT_DIR" ]]; then
  usage >&2
  exit 2
fi
if [[ -z "${VLA_DATASETS_DIR:-}" ]]; then
  echo "VLA_DATASETS_DIR is required; no GPU training was started." >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required on PATH." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$(dirname "$OUTPUT_DIR")"
CONSOLE_LOG="${OUTPUT_DIR}.console.log"
mkdir -p "$(dirname "$CONSOLE_LOG")"

resolve_resume() {
  uv run python - "$OUTPUT_DIR" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
lock = run_dir / "run.lock"
if lock.is_file():
    raw = lock.read_text(encoding="utf-8").strip()
    pid = int(raw) if raw.isdigit() else None
    alive = pid is not None and pid > 1 and Path(f"/proc/{pid}").exists()
    if alive:
        print(f"LOCK_HELD {pid}", file=sys.stderr)
        sys.exit(3)
    lock.unlink()
    print(f"removed stale run.lock pid={raw}", file=sys.stderr)

if not run_dir.exists():
    print("FRESH")
    raise SystemExit(0)

manifest_path = run_dir / "manifest.json"
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") == "completed":
        print("COMPLETED")
        raise SystemExit(0)

checkpoints = run_dir / "checkpoints"
complete: list[tuple[int, Path]] = []
if checkpoints.is_dir():
    for child in checkpoints.iterdir():
        if not child.is_dir() or not child.name.startswith("step_"):
            continue
        if not (child / "COMPLETED.json").is_file():
            continue
        try:
            step = int(child.name.split("_", 1)[1])
        except ValueError:
            continue
        complete.append((step, child))

if not complete:
    print(
        f"existing run {run_dir} has no complete checkpoint; "
        "refusing to overwrite. no GPU training was started.",
        file=sys.stderr,
    )
    raise SystemExit(4)

complete.sort()
chosen = complete[-1][1]
pointer = checkpoints / "latest.json"
if pointer.is_file():
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    named = checkpoints / str(payload.get("directory", ""))
    if named.is_dir() and (named / "COMPLETED.json").is_file():
        chosen = named
print(str(chosen.resolve()))
PY
}

set +e
RESUME_OUT="$(resolve_resume)"
STATUS=$?
set -e
if [[ $STATUS -ne 0 ]]; then
  exit "$STATUS"
fi

CMD=(
  uv run python scripts/train_seen.py
  --config "$CONFIG"
  --profile "$PROFILE"
  --output-dir "$OUTPUT_DIR"
  --output-root "$VLA_DATASETS_DIR"
  --log-freq "$LOG_FREQ"
)

case "$RESUME_OUT" in
  FRESH)
    echo "starting fresh run at $OUTPUT_DIR"
    ;;
  COMPLETED)
    echo "run already completed: $OUTPUT_DIR"
    exit 0
    ;;
  *)
    echo "resuming $OUTPUT_DIR from $RESUME_OUT"
    CMD+=(--resume-from "$RESUME_OUT")
    ;;
esac

{
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) pid=$$ ====="
  echo "cmd=${CMD[*]}"
} | tee -a "$CONSOLE_LOG"

exec "${CMD[@]}" >>"$CONSOLE_LOG" 2>&1
