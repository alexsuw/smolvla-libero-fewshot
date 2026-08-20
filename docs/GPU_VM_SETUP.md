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

До этого момента training не запускается.
