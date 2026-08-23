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
  - lora
base_model: alexsuw/smolvla-libero-fewshot-seen-expert-100k
---

# SmolVLA N=1 Target-LoRA and Replay-LoRA on LIBERO-Goal

Twelve independently trained checkpoints: two methods, three held-out target
tasks, and train seeds 42/123. Every cell starts from the same frozen
[`seen-expert-100k`](https://huggingface.co/alexsuw/smolvla-libero-fewshot-seen-expert-100k)
checkpoint (SHA-256 `2cd510a594a87580f7368b782ca9b37332c0e5002d807093c759e95fbfb57c88`)
and uses only the first registered target demonstration.

**Code:** [github.com/alexsuw/smolvla-libero-fewshot](https://github.com/alexsuw/smolvla-libero-fewshot)<br>
**Collection:** [SmolVLA LIBERO Few-shot](https://huggingface.co/collections/alexsuw/smolvla-libero-few-shot-6a8b009357482d2b4b9d3c2f)

## Methods and results

| Method | Target success | Corrected seen retention | Trainable parameters |
|---|---:|---:|---:|
| Naive N=1 reference | 109/120 (90.8%) | 37/180 (20.6%) | 99,880,992 |
| Target-LoRA N=1 | 99/120 (82.5%) | 19/180 (10.6%) | 4,215,632 |
| Replay-LoRA N=1 | 67/120 (55.8%) | 2/180 (1.1%) | 4,215,632 |

Replay-LoRA uses a fixed 75% target / 25% seen mixture. Seen replay comes only
from `libero_90` and was not selected using the three retention probes. No
method was tuned or rerun using success.

## Layout and loading

```text
<target_lora|replay_lora>/<task>_n01_s<seed>/
  weights.pt
  adapter/adapter_model.pt
  adapter/adapter_config.json
  normalization_stats.json
  config.resolved.yaml
  COMPLETED.json
  checksums.json
  trainable_parameters.txt
  run/                         # provenance and training metrics, no optimizer
```

`weights.pt` is the complete inference checkpoint. The `adapter/` files are
also provided for adapter-specific analysis. `index.json` contains exact
weights/adapter hashes and per-cell target/retention results.

```python
from huggingface_hub import hf_hub_download

weights = hf_hub_download(
    "alexsuw/smolvla-libero-fewshot-lora-n1",
    "target_lora/drawer_middle_n01_s42/weights.pt",
)
adapter = hf_hub_download(
    "alexsuw/smolvla-libero-fewshot-lora-n1",
    "target_lora/drawer_middle_n01_s42/adapter/adapter_model.pt",
)
```

For target deployment use the cell's `normalization_stats.json`. Corrected
seen retention must instead use the frozen `libero_90` statistics with digest
`b159b6fed3e52edf25bd39b377dd64940221b7a030362daf7f726b1c2ecb30cf`.

Optimizer/RNG state, raw rollouts, datasets, traces, videos, and credentials
are intentionally excluded. Full tables and limitations are in
`results/results.md`.

## License

Derivatives of `lerobot/smolvla_base` trained on `nvidia/LIBERO_LeRobot_v3`.
Project code is Apache-2.0; follow upstream model and dataset terms for weights.
