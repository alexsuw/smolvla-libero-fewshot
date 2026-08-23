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

# SmolVLA seen-expert on LIBERO-90 (100k)

Frozen **seen-domain** SmolVLA checkpoint after 100k imitation steps on
`libero_90`. This is the immutable origin for every target few-shot run in
the project. It has **not** been trained on the three held-out `libero_goal`
tasks.

**Code:** [github.com/alexsuw/smolvla-libero-fewshot](https://github.com/alexsuw/smolvla-libero-fewshot)  
**Few-shot family (30 naive FT checkpoints):**
[`alexsuw/smolvla-libero-fewshot-naive-baseline`](https://huggingface.co/alexsuw/smolvla-libero-fewshot-naive-baseline)  
**Collection:**
[`alexsuw/smolvla-libero-few-shot-6a8b009357482d2b4b9d3c2f`](https://huggingface.co/collections/alexsuw/smolvla-libero-few-shot-6a8b009357482d2b4b9d3c2f)

## Research question

How many held-out expert demonstrations does SmolVLA need, after LIBERO-90
domain adaptation, to solve a new language-conditioned manipulation task?

## Pinned sources

| Artifact | Link | Revision / id |
|---|---|---|
| Base policy | [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base) | `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| SmolVLA | [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) | paper |
| Dataset | [nvidia/LIBERO_LeRobot_v3](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3) | `e5907374380b8f96511957e6ba5582be52a1e179` |
| LIBERO | [arXiv:2306.03310](https://arxiv.org/abs/2306.03310) · [code](https://github.com/Lifelong-Robot-Learning/LIBERO) | benchmark |
| LeRobot | [huggingface/lerobot](https://github.com/huggingface/lerobot) | `d451fe4f1f1b00a812f95aa9534389b5e42ab155` |

Training suite: **`libero_90` only**. Vision encoder and VLM backbone frozen;
Action Expert + state/action projections trained.

## This file

| Field | Value |
|---|---|
| `run_id` | `seen__expert__libero90__nall__s42__20260822T010019Z__gd4b8fb8` |
| step | 100000 |
| `weights.pt` SHA-256 | `2cd510a594a87580f7368b782ca9b37332c0e5002d807093c759e95fbfb57c88` |
| seen-probe success | 24/30 = 0.80 on three frozen `libero_90` probes, seeds 1000–1009, horizon 300 |
| `optimizer.pt` | not uploaded (eval does not need it) |

Selected from `libero_90` probes only. Target-task success was **not** used
to pick this checkpoint, the learning rate, or the training length.

## Normalization

Deploy this checkpoint with **suite-wide `libero_90` MEAN_STD**. Do not attach
held-out `libero_goal` overlay statistics. That mix zeros seen-probe success
and is a deployment incompatibility, not a measure of parameter forgetting.

## License

Weights are a derivative of `lerobot/smolvla_base` trained on
`nvidia/LIBERO_LeRobot_v3`. Project code is Apache-2.0; follow upstream model
and dataset terms for the weights.
