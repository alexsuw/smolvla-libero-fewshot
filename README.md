# SmolVLA LIBERO Few-shot

Воспроизводимый pipeline для few-shot adaptation модели
`lerobot/smolvla_base` на задачах LIBERO. Техническим контрактом проекта
является [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

## Текущий статус

M0–M5 реализация завершена на CPU; hardware acceptance M1/M3/M4/M5 остаётся
pending. Eval protocol, object-storage sync, `predictions.md`, pseudo-target
freeze и project-owned SmolVLA trainer (`train_seen.py --profile full`) готовы.
Seen-probe selection/freeze (`eval_seen --profile full`, `select_seen_checkpoint.py`)
готовы на CPU. 100k seen-pretrain и live rollouts ждут Linux CUDA VM.
`lerobot-train` не вызывается. Target eval закрыт, пока seen checkpoint не frozen.
Актуальный прогресс и evidence перечислены в [`STATUS.md`](STATUS.md).

## Быстрый старт для разработки

Требования: `uv` и Python 3.12.

```bash
uv sync --frozen --extra data
uv run pytest -q
make check-reporting
```

Посмотреть интерфейс команд можно без запуска GPU:

```bash
uv run python scripts/train_seen.py --help
uv run python scripts/train_seen.py --config configs/train/smoke.yaml \
  --profile static --protocol resume-compare --output-dir /tmp/vla-m5
```

`--profile full` на Linux + CUDA запускает project-owned SmolVLA trainer
(без `lerobot-train` / W&B). На macOS и без CUDA команда сразу завершается
с `no GPU training was started`.

```bash
# VM, после videos + gpu extra. Сначала короткий smoke:
uv run python scripts/train_seen.py \
  --config configs/train/smoke.yaml --profile full \
  --output-dir "$VLA_RUNS_DIR/seen_smoke"

# Затем 100k libero_90 (TODO 23):
uv run python scripts/train_seen.py \
  --config configs/train/seen_expert.yaml --profile full \
  --output-dir "$VLA_RUNS_DIR/seen_expert"
```

## Dataset metadata (M2)

Metadata-only download не декодирует videos и пишет revision-encoded путь
`<datasets>/nvidia_LIBERO_LeRobot_v3/<40-char SHA>/`.

```bash
export VLA_DATASETS_DIR="$HOME/.cache/vla-fewshot/datasets"
uv run python scripts/download_dataset.py --output-root "$VLA_DATASETS_DIR"
uv run python scripts/inspect_dataset.py --output-root "$VLA_DATASETS_DIR" \
  --output-dir artifacts/validation/M2
uv run python scripts/verify_split.py --output-root "$VLA_DATASETS_DIR"
uv run python scripts/verify_no_leakage.py --output-root "$VLA_DATASETS_DIR"
```

Videos скачиваются только с `--include-videos` и ровно одним `--suite`.
Python-код не содержит `/mnt/vla` или `/content/drive`.

## Принципы

- Model и dataset всегда загружаются по полным revision SHA.
- W&B отключён; metrics пишутся в TensorBoard, CSV и JSONL.
- Пути задаются environment variables и platform configs.
- Финальные runs должны переживать preemption через атомарные checkpoints.
- Dataset, checkpoints, videos, caches и secrets не попадают в Git.
- Long GPU runs выполняются на Linux VM под `tmux`, а notebooks остаются
  тонкими launcher/analysis слоями.

## RTX 6000 Blackwell

Финальный VM overlay будет использовать auto-detection, BF16 после smoke-test
и фиксированный effective batch. Код не предполагает конкретный объём VRAM:
physical batch подбирается до создания final run, затем фиксируется в manifest.
Linux cu128/LeRobot/MuJoCo runtime уже закреплён в universal `uv.lock`.
Фактическая проверка GPU, BF16, EGL и VRAM выполняется позднее по
[`docs/GPU_VM_SETUP.md`](docs/GPU_VM_SETUP.md); static CI не считается
hardware evidence.

## Основные каталоги

- `configs/` — tracked experiment contracts, без credentials.
- `src/vla_fewshot/` — importable production logic.
- `scripts/` — стабильные project-owned CLI.
- `notebooks/` — визуальная проверка и анализ готовых artifacts.
- `artifacts/` — локальное validation evidence, игнорируется Git.
- Runtime data находятся вне worktree в `VLA_DATA_ROOT`.

## Лицензия и upstream

Project code использует Apache-2.0. Лицензии и условия model, dataset,
LeRobot и LIBERO проверяются отдельно по pinned upstream revisions.
