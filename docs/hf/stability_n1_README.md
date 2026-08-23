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
  - continual-learning
base_model: alexsuw/smolvla-libero-fewshot-seen-expert-100k
---

# SmolVLA N=1 Frozen-Stats FT and Anchored FT (L2-SP)

Twelve independently trained checkpoints from a matched stability experiment:
two methods, three held-out LIBERO-Goal tasks, and train seeds 42/123. All
cells start from the same frozen
[`seen-expert-100k`](https://huggingface.co/alexsuw/smolvla-libero-fewshot-seen-expert-100k)
checkpoint and use the first registered target demonstration.

**Code:** [github.com/alexsuw/smolvla-libero-fewshot](https://github.com/alexsuw/smolvla-libero-fewshot)<br>
**Collection:** [SmolVLA LIBERO Few-shot](https://huggingface.co/collections/alexsuw/smolvla-libero-few-shot-6a8b009357482d2b4b9d3c2f)

## Frozen protocol and results

Both methods use the original `libero_90` normalization during target training
and deployment. No target-overlay statistics are fitted. Anchored FT adds an
FP32 raw-sum L2-SP penalty with preregistered `lambda=0.01` on the trainable
Action Expert/projection parameters relative to their frozen initialization.

| Method | Target success | Corrected seen retention | Trainable parameters | Peak VRAM |
|---|---:|---:|---:|---:|
| Naive N=1 reference | 109/120 (90.8%) | 37/180 (20.6%) | 99,880,992 | 7,540 MiB |
| Frozen-Stats FT N=1 | 109/120 (90.8%) | 39/180 (21.7%) | 99,880,992 | 7,540 MiB |
| Anchored FT N=1 | 105/120 (87.5%) | 57/180 (31.7%) | 99,880,992 | 8,036 MiB |

No method or anchoring strength was selected, tuned, or rerun using target or
retention success.

## Layout and loading

```text
<frozen_stats|anchored_l2sp>/<task>_n01_s<seed>/
  weights.pt
  normalization_stats.json   # canonical frozen libero_90 statistics
  config.resolved.yaml
  COMPLETED.json
  checksums.json
  trainable_parameters.txt
  run/                        # provenance and training metrics, no optimizer
```

```python
from huggingface_hub import hf_hub_download

weights = hf_hub_download(
    "alexsuw/smolvla-libero-fewshot-stability-n1",
    "anchored_l2sp/drawer_middle_n01_s42/weights.pt",
)
stats = hf_hub_download(
    "alexsuw/smolvla-libero-fewshot-stability-n1",
    "anchored_l2sp/drawer_middle_n01_s42/normalization_stats.json",
)
```

Every published normalization file has SHA-256
`b159b6fed3e52edf25bd39b377dd64940221b7a030362daf7f726b1c2ecb30cf`.
`index.json` records exact checkpoint hashes and per-cell evaluation results.
Full tables and limitations are in `results/results.md`.

Optimizer/RNG state, raw rollouts, datasets, traces, videos, and credentials
are intentionally excluded.

## License

Derivatives of `lerobot/smolvla_base` trained on `nvidia/LIBERO_LeRobot_v3`.
Project code is Apache-2.0; follow upstream model and dataset terms for weights.
