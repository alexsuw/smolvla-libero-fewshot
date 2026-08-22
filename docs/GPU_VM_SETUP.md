# GPU and Colab handoff after M1 implementation

GPU пока покупать не нужно. Этот документ используется только после завершения
доступных CPU/CI milestones и перед первым hardware smoke.

## Требования к RTX VM

- Ubuntu 22.04/24.04 x86-64;
- RTX PRO 6000 Blackwell или совместимая CUDA GPU;
- NVIDIA driver `>=570.86.10`;
- system FFmpeg exactly `7.1.1` с AV1 encoder/decoder;
- 64+ GB RAM и persistent SSD с резервом не менее 50 GB;
- `uv` и Git;
- runtime roots на persistent storage.

Doctor записывает фактические GPU name, VRAM, compute capability, supported
Torch architectures и BF16 support. Модель не предполагает объём VRAM в коде.

Host notes from the first RTX VM:

- Ubuntu 24.04 apt FFmpeg is `6.1.1`. Full doctor requires exact `7.1.1` with
  `libaom-av1`. A static-only build makes TorchCodec fall back to slow PyAV.
  `scripts/build_shared_ffmpeg.sh` builds the pinned shared libraries under
  `VLA_CACHE_DIR`; put both its `bin` and `lib` on the runtime paths.
- Headless CUDA images often omit NVIDIA EGL. Install `libnvidia-gl-<driver>`
  so `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` exists; otherwise
  `libero_two_camera` fails even when MuJoCo EGL passes.
- The Linux user must be in groups `video` and `render` to open `/dev/dri`.
  New shells pick this up after login; in an existing session use
  `sg render` / `sg video`.
- Put the 7.1.1 `ffmpeg` on `PATH` before any distro binary.

## VM validation

```bash
git clone https://github.com/alexsuw/smolvla-libero-fewshot.git
cd smolvla-libero-fewshot
git checkout <PINNED_PROJECT_COMMIT>

export VLA_DATA_ROOT=/mnt/vla
export VLA_SCRATCH_DIR=/local_nvme/vla_scratch

bash scripts/bootstrap_vm.sh
# Выполнить напечатанную bootstrap команду `source .../runtime.env`.

uv run python scripts/doctor.py \
  --config configs/platform/gpu_vm.yaml \
  --profile full

VLA_RUN_GPU_TESTS=1 VLA_RUN_STORAGE_TESTS=1 \
  uv run pytest -q -m "gpu or integration"
```

Bootstrap загружает LIBERO assets только по revision из
`configs/revisions.lock.yaml`, не model weights и не training dataset.
Он никогда не удаляет существующие данные и не перезаписывает другой
`~/.libero/config.yaml`.

Metadata-only dataset download (M2) можно выполнить до покупки GPU:

```bash
export VLA_DATASETS_DIR=/mnt/vla/datasets
uv sync --frozen --extra data
uv run python scripts/download_dataset.py --output-root "$VLA_DATASETS_DIR"
uv run python scripts/inspect_dataset.py --output-root "$VLA_DATASETS_DIR" \
  --output-dir artifacts/validation/M2
```

## Colab Drive validation

Открыть `notebooks/colab_smoke.ipynb`, смонтировать Google Drive и выполнить
M1 cells. Durable root должен находиться внутри `/content/drive`; иначе doctor
запишет `ephemeral=true` и hardware acceptance останется incomplete.

## Закрытие M1

После успешного full doctor сохранить весь output directory на persistent
disk/Drive и передать `doctor.json` обратно в проект. Затем:

1. проверить required checks и hardware identity;
2. изменить revision status с `resolved_m1_pending_hardware` на `validated_m1`;
3. повторить full doctor на том же host;
4. обновить `STATUS.md` отдельным commit.

## Seen-pretrain on the VM (TODO 23)

Код trainer готов. 100k `seen_expert` только после M4 full smoke и 200-step
`configs/train/smoke.yaml`. Длительный run — CLI в `tmux`, не IDE terminal.
Данные и checkpoints только на persistent `VLA_DATA_ROOT`.

```bash
export PATH="$HOME/.local/bin:$PATH"
set -a
source "$VLA_DATA_ROOT/bootstrap/<stamp>/runtime.env"
set +a
export VLA_DATASETS_DIR="${VLA_DATA_ROOT}/datasets"
export VLA_RUNS_DIR="${VLA_DATA_ROOT}/runs"

# One-time idempotent build; required for fast TorchCodec video decode.
bash scripts/build_shared_ffmpeg.sh
FFMPEG_SHARED="${VLA_CACHE_DIR}/ffmpeg-shared-7.1.1/install"
export PATH="${FFMPEG_SHARED}/bin:$PATH"
export LD_LIBRARY_PATH="${FFMPEG_SHARED}/lib:${LD_LIBRARY_PATH:-}"

# 200-step SmolVLA smoke (config already has max_steps=200)
tmux new-session -d -s vla-seen-smoke \
  "sg render -c 'sg video -c \"cd $(pwd) && bash scripts/run_durable_seen_train.sh --config configs/train/smoke.yaml --output-dir $VLA_RUNS_DIR/seen_smoke_200 --log-freq 1\"'"

# Primary 100k seen-pretrain after the 200-step smoke succeeds.
# auto_fit runs only on a fresh output-dir; crash resume reuses frozen batch.
tmux new-session -d -s vla-seen-100k \
  "sg render -c 'sg video -c \"cd $(pwd) && bash scripts/run_durable_seen_train.sh --config configs/train/seen_expert.yaml --output-dir $VLA_RUNS_DIR/seen_expert_100k --log-freq 50\"'"
```

The same `run_durable_seen_train.sh` command after a host crash resumes from
`checkpoints/latest.json` (complete `COMPLETED.json` only). SIGTERM/SIGINT
save a checkpoint before exit. `kill -9` or power loss can lose up to
`every_steps` (5000 on 100k). Console logs are `$OUTPUT_DIR.console.log` on
the same durable disk. `lerobot-train` is never called.

On the RTX PRO 6000, measured full-dataset throughput improved from
`1.338 s/step` (`physical=4`, PyAV) to `0.365 s/step` (`physical=32`,
TorchCodec, ordered one-batch prefetch). The 100k forecast is about 10.15 h.
The optimization keeps every `libero_90` episode/task and effective batch 32;
it does not select scenes or use target success.

`--profile static` остаётся CPU toy smoke.
`--stop-after` можно использовать только как resume-allowlist override на уже
замороженном `max_steps`; для 200-step GPU smoke берите `configs/train/smoke.yaml`.

## TensorBoard

Run TensorBoard only on localhost:

```bash
tmux new-session -d -s vla-tensorboard \
  "cd $(pwd) && uv run tensorboard --logdir $VLA_RUNS_DIR --host 127.0.0.1 --port 6006"
```

From the local machine, open an SSH tunnel and then visit
`http://127.0.0.1:6006`:

```bash
ssh -L 6006:127.0.0.1:6006 <vm-user>@<vm-host>
```

Use Scalars for `train/loss`, `train/learning_rate`, `train/grad_norm`, and
`train/samples_per_second`. Do not expose port 6006 on `0.0.0.0`.

## Where weights are saved

Training writes atomic `weights.pt`, optimizer/RNG state, checksums, and
`COMPLETED.json` under
`$VLA_RUNS_DIR/<run>/checkpoints/step_<N>/`. The seen run saves every 5,000
steps. It does **not** upload to Hugging Face implicitly. A private Hugging
Face model repository is optional only after final provenance/checksum
verification. Until an object-storage or private-HF destination and
credentials are configured, the durable VM disk is the only checkpoint copy.

## Seen probes and checkpoint freeze (TODO 24)

После seen-pretrain. Только `libero_90` probe tasks; target eval закрыт, пока YAML
не `status: frozen`.

```bash
# All complete checkpoints × three frozen probe slugs
uv run python scripts/eval_seen.py \
  --config configs/eval/seen_probe.yaml \
  --profile full \
  --run-dir "$VLA_RUNS_DIR/seen_expert_100k" \
  --output-dir "$VLA_RUNS_DIR/seen_probes" \
  --output-root "$VLA_DATASETS_DIR"

# Dry-run selection, then write the freeze
uv run python scripts/select_seen_checkpoint.py \
  --run-dir "$VLA_RUNS_DIR/seen_expert_100k" \
  --probe-root "$VLA_RUNS_DIR/seen_probes"

uv run python scripts/select_seen_checkpoint.py \
  --run-dir "$VLA_RUNS_DIR/seen_expert_100k" \
  --probe-root "$VLA_RUNS_DIR/seen_probes" \
  --write
```

Zero-shot from the frozen seen checkpoint (TODO 26). Empty train list, 3×20,
same hash. `--task` omitted runs all three:

```bash
uv run python scripts/eval_zero_shot.py --print-grid

uv run python scripts/eval_zero_shot.py \
  --profile full \
  --output-dir "$VLA_RUNS_DIR/zero_shot" \
  --output-root "$VLA_DATASETS_DIR"
```

Paired language control (TODO 27). Same frozen hash, same seeds/states, only the
instruction string changes:

```bash
uv run python scripts/eval_language_control.py --print-grid

uv run python scripts/eval_language_control.py \
  --profile full \
  --output-dir "$VLA_RUNS_DIR/language_control" \
  --output-root "$VLA_DATASETS_DIR"
```

Seen LoRA (TODO 25) is skipped. After zero-shot and language control, run the
18-cell baseline (no LoRA, no replay). Print the grid first:

```bash
uv run python scripts/train_target.py --print-grid

uv run python scripts/eval_target.py --print-grid

uv run python scripts/verify_baseline_eval.py --print-grid
```

Every complete checkpoint of a cell (≥20 rollouts, failure videos):

```bash
uv run python scripts/eval_target.py \
  --config configs/eval/final.yaml \
  --profile full \
  --run-dir "$VLA_RUNS_DIR/target_baseline/drawer_middle_n05_s42" \
  --task drawer_middle \
  --n-demos 5 \
  --seed 42 \
  --output-dir "$VLA_RUNS_DIR/target_eval/drawer_middle_n05_s42" \
  --output-root "$VLA_DATASETS_DIR"

uv run python scripts/verify_baseline_eval.py \
  --train-dir "$VLA_RUNS_DIR/target_baseline/drawer_middle_n05_s42" \
  --eval-dir "$VLA_RUNS_DIR/target_eval/drawer_middle_n05_s42" \
  --task drawer_middle \
  --n-demos 5 \
  --seed 42
```

Example train cell:

```bash
uv run python scripts/train_target.py \
  --config configs/train/target_baseline.yaml \
  --task drawer_middle \
  --n-demos 5 \
  --seed 42 \
  --profile full \
  --output-dir "$VLA_RUNS_DIR/target_baseline/drawer_middle_n05_s42" \
  --log-freq 50
```

Target LoRA ablation uses the same 18 cells after the baseline exists:

```bash
uv run python scripts/train_target.py \
  --config configs/train/target_lora.yaml \
  --print-grid

uv run python scripts/eval_target.py \
  --train-config configs/train/target_lora.yaml \
  --print-grid

uv run python scripts/verify_baseline_eval.py --method lora --print-grid
```

Example LoRA cell (same origin/episodes as the matching baseline cell):

```bash
uv run python scripts/train_target.py \
  --config configs/train/target_lora.yaml \
  --task drawer_middle \
  --n-demos 5 \
  --seed 42 \
  --profile full \
  --output-dir "$VLA_RUNS_DIR/target_lora/drawer_middle_n05_s42" \
  --log-freq 50
```

Replay-LoRA is the same 18 cells plus 25% `libero_90` mix:

```bash
uv run python scripts/train_target.py \
  --config configs/train/target_replay_lora.yaml \
  --print-grid

uv run python scripts/eval_target.py \
  --train-config configs/train/target_replay_lora.yaml \
  --print-grid

uv run python scripts/verify_baseline_eval.py --method replay_lora --print-grid

uv run python scripts/train_target.py \
  --config configs/train/target_replay_lora.yaml \
  --task drawer_middle \
  --n-demos 5 \
  --seed 42 \
  --profile full \
  --output-dir "$VLA_RUNS_DIR/target_replay_lora/drawer_middle_n05_s42" \
  --log-freq 50
```
