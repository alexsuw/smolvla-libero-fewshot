# SmolVLA LIBERO Few-shot

Воспроизводимый pipeline для few-shot adaptation модели
`lerobot/smolvla_base` на задачах LIBERO. Техническим контрактом проекта
является [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

## Текущий статус

M0 завершён. M1 runtime/doctor реализуется без аренды GPU; hardware acceptance
останется pending до финального RTX/Colab этапа.
Обучение и платные GPU-запуски до прохождения smoke-gates M1–M5 запрещены.
Актуальный прогресс и evidence перечислены в [`STATUS.md`](STATUS.md).

## Быстрый старт для разработки

Требования: `uv` и Python 3.12.

```bash
uv sync
uv run pytest -q
make check-m1-static
```

Посмотреть интерфейс будущей команды можно без запуска вычислений:

```bash
uv run python scripts/train_seen.py --help
```

До реализации соответствующего milestone вычислительные entry points
завершаются с понятной ошибкой. Это защищает от случайного запуска training
на неподготовленной машине.

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
