# SmolVLA LIBERO few-shot

Reproducible few-shot adaptation of [SmolVLA](https://huggingface.co/lerobot/smolvla_base)
on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), using the NVIDIA
LeRobot v3 conversion of the benchmark.

## Task

After domain-adapting SmolVLA on **seen** `libero_90` demonstrations, how many
**held-out** `libero_goal` expert episodes (`N = 0, 1, 2, 5, 10, 25`) does a
naive target-only continuation need to solve three language-conditioned
manipulation tasks?

Held-out instructions (exact dataset text):

| slug | instruction |
|---|---|
| `drawer_middle` | open the middle drawer of the cabinet |
| `bowl_stove` | put the bowl on the stove |
| `wine_cabinet` | put the wine bottle on top of the cabinet |

## Sources (pinned)

| What | Where | Pin |
|---|---|---|
| Base policy | [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) | `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| SmolVLA paper | [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) | — |
| Dataset | [`nvidia/LIBERO_LeRobot_v3`](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3) | `e5907374380b8f96511957e6ba5582be52a1e179` |
| LIBERO paper | [arXiv:2306.03310](https://arxiv.org/abs/2306.03310) | — |
| LIBERO code | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | via `hf-libero` |
| LeRobot | [huggingface/lerobot](https://github.com/huggingface/lerobot) | `d451fe4f1f1b00a812f95aa9534389b5e42ab155` |
| Weights on Hub | [seen 100k](https://huggingface.co/alexsuw/smolvla-libero-fewshot-seen-expert-100k) · [naive family](https://huggingface.co/alexsuw/smolvla-libero-fewshot-naive-baseline) · [N=1 LoRA](https://huggingface.co/alexsuw/smolvla-libero-fewshot-lora-n1) · [N=1 stability](https://huggingface.co/alexsuw/smolvla-libero-fewshot-stability-n1) · [collection](https://huggingface.co/collections/alexsuw/smolvla-libero-few-shot-6a8b009357482d2b4b9d3c2f) | see Hub cards |

Dataset suites used here:

- **Seen pretrain:** `libero_90` only (no held-out `libero_goal` tasks).
- **Target few-shot:** the three `libero_goal` texts above. Subsets are nested
  prefixes of one frozen episode-id list per task (`configs/splits/target_splits.json`).

## Checkpoints on Hugging Face

Weights are **not** stored in Git. Each naive cell is an independent fine-tune
from the frozen seen checkpoint (N=10 does not continue N=5).

| Hub repo | Contents |
|---|---|
| [`alexsuw/smolvla-libero-fewshot-seen-expert-100k`](https://huggingface.co/alexsuw/smolvla-libero-fewshot-seen-expert-100k) | Frozen seen policy, 100k steps, `weights.pt` SHA-256 `2cd510a594a87580f7368b782ca9b37332c0e5002d807093c759e95fbfb57c88` |
| [`alexsuw/smolvla-libero-fewshot-naive-baseline`](https://huggingface.co/alexsuw/smolvla-libero-fewshot-naive-baseline) | 30 naive target finals: 3 tasks × N∈{1,2,5,10,25} × seeds {42,123} |
| [`alexsuw/smolvla-libero-fewshot-lora-n1`](https://huggingface.co/alexsuw/smolvla-libero-fewshot-lora-n1) | 12 N=1 finals: Target-LoRA and Replay-LoRA × 3 tasks × 2 seeds; full weights and adapters |
| [`alexsuw/smolvla-libero-fewshot-stability-n1`](https://huggingface.co/alexsuw/smolvla-libero-fewshot-stability-n1) | 12 N=1 finals: Frozen-Stats FT and Anchored FT (L2-SP) × 3 tasks × 2 seeds |

Download one few-shot cell:

```python
from huggingface_hub import hf_hub_download

weights = hf_hub_download(
    "alexsuw/smolvla-libero-fewshot-naive-baseline",
    "drawer_middle_n01_s42/weights.pt",
)
stats = hf_hub_download(
    "alexsuw/smolvla-libero-fewshot-naive-baseline",
    "drawer_middle_n01_s42/normalization_stats.json",
)
```

Target evaluation must use that cell’s **subset overlay** MEAN_STD, not
suite-wide `libero_goal` statistics. Seen-probe / forgetting measurements must
use **`libero_90` suite** statistics. Mixing the two is a deployment bug, not
a forgetting metric.

## Naive target success (20 eval seeds, not used for HP search)

| N | success | notes |
|---|---|---|
| 0 | 1/60 (0.017) | zero-shot from frozen seen + `libero_90` stats |
| 1 | 109/120 (0.908) | entire run is inside cosine warmup |
| 2 | 100/120 (0.833) | entire run is inside cosine warmup |
| 5 | 107/120 (0.892) | official `DEMO_BUDGETS` |
| 10 | 116/120 (0.967) | official `DEMO_BUDGETS` |
| 25 | 114/120 (0.950) | official `DEMO_BUDGETS` |

Official spec budgets remain `{5, 10, 25}`. N=1/2 are a ceiling-extension of
the same nested prefixes; `warmup_steps=1000` was not retuned.

## Quick start (code)

Python 3.12 and `uv`:

```bash
uv sync --frozen --extra data
uv run pytest -q
make check
```

Full GPU training/eval needs Linux CUDA and `uv sync --frozen --extra gpu`.
Paths come from environment variables (`VLA_DATA_ROOT`, `VLA_DATASETS_DIR`,
`VLA_RUNS_DIR`), never from hardcoded `/mnt/vla` in production modules.

```bash
uv run python scripts/train_seen.py --config configs/train/seen_expert.yaml --help
uv run python scripts/train_target.py --config configs/train/target_baseline.yaml --print-grid
```

`lerobot-train` is not used. W&B is disabled; logs are TensorBoard + CSV/JSONL.

## Repository layout

- `configs/` — frozen experiment contracts (no credentials)
- `src/vla_fewshot/` — production library
- `scripts/` — project CLI
- `tests/` — unit contracts
- `artifacts/` — local evidence (gitignored)
- Runtime datasets, checkpoints, and videos stay outside the worktree

## License

Project **code** is [Apache-2.0](LICENSE). Model weights are derivatives of
`lerobot/smolvla_base` and were trained on `nvidia/LIBERO_LeRobot_v3`; follow
those upstream licenses and dataset terms as well.
