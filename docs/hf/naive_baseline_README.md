---
license: other
library_name: lerobot
pipeline_tag: robotics
datasets:
  - nvidia/LIBERO_LeRobot_v3
tags:
  - smolvla
  - libero
  - robotics
  - imitation-learning
  - vision-language-action
  - few-shot
base_model: lerobot/smolvla_base
---

# SmolVLA naive few-shot family on LIBERO-Goal

Thirty **independent** naive target fine-tunes of the frozen seen-expert
checkpoint. Each cell starts from the same origin weights and sees only a
prefix of one frozen `libero_goal` demonstration list. N=10 does **not**
continue N=5.

**Code:** [github.com/alexsuw/smolvla-libero-fewshot](https://github.com/alexsuw/smolvla-libero-fewshot)  
**Origin seen checkpoint:**
[`alexsuw/smolvla-libero-fewshot-seen-expert-100k`](https://huggingface.co/alexsuw/smolvla-libero-fewshot-seen-expert-100k)
(`weights.pt` SHA-256 `2cd510a594a87580f7368b782ca9b37332c0e5002d807093c759e95fbfb57c88`)  
**Collection:**
[`alexsuw/smolvla-libero-few-shot-6a8b009357482d2b4b9d3c2f`](https://huggingface.co/collections/alexsuw/smolvla-libero-few-shot-6a8b009357482d2b4b9d3c2f)

## Task and data

Held-out LIBERO-Goal instructions (exact NVIDIA LeRobot v3 text):

| slug | instruction | dataset `task_index` |
|---|---|---|
| `drawer_middle` | open the middle drawer of the cabinet | 9 |
| `bowl_stove` | put the bowl on the stove | 7 |
| `wine_cabinet` | put the wine bottle on top of the cabinet | 4 |

- Dataset: [`nvidia/LIBERO_LeRobot_v3`](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3) revision `e5907374380b8f96511957e6ba5582be52a1e179`
- Suite for these runs: **`libero_goal`**
- Episode IDs: nested prefixes of `configs/splits/target_splits.json` (first 1 / 2 / 5 / 10 / 25)
- Train seeds: `{42, 123}`
- Base policy: [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`
- Papers: [SmolVLA](https://arxiv.org/abs/2506.01844), [LIBERO](https://arxiv.org/abs/2306.03310)

## Layout

```text
<task>_n<NN>_s<seed>/
  weights.pt
  normalization_stats.json   # subset overlay MEAN_STD used at train/eval
  COMPLETED.json
  checksums.json
  config.resolved.yaml
  trainable_parameters.txt
```

Example:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    "alexsuw/smolvla-libero-fewshot-naive-baseline",
    "bowl_stove_n10_s42/weights.pt",
)
hf_hub_download(
    "alexsuw/smolvla-libero-fewshot-naive-baseline",
    "bowl_stove_n10_s42/normalization_stats.json",
)
```

`optimizer.pt` is not uploaded. `index.json` lists Hub prefixes, training
steps, and `weights.pt` SHA-256 for every cell.

## Target success (20 eval seeds / cell)

Pooled over 3 tasks × 2 seeds. N=0 is zero-shot from the seen checkpoint
(not in this repo). Official spec budgets are `{5,10,25}`; `{1,2}` use the
same prefixes and the same `warmup_steps=1000` (those short runs stay inside
warmup — a recorded limitation, not a retune).

| N | naive target success |
|---|---|
| 0 | 1/60 (0.017) |
| 1 | 109/120 (0.908) |
| 2 | 100/120 (0.833) |
| 5 | 107/120 (0.892) |
| 10 | 116/120 (0.967) |
| 25 | 114/120 (0.950) |

Target success was **not** used to choose hyperparameters or the seen checkpoint.

## Normalization (read this)

- **Target eval:** use this folder’s `normalization_stats.json` (train overlay).
- **Seen-probe / retention:** use **`libero_90` suite** stats with these
  adapted weights if you want a weight-forgetting comparison.
- Evaluating adapted weights on seen probes with the target overlay is a
  **deployment mismatch**. That number is not catastrophic forgetting.

## License

Derivatives of `lerobot/smolvla_base` trained on `nvidia/LIBERO_LeRobot_v3`.
Project code is Apache-2.0; follow upstream model and dataset terms for weights.
