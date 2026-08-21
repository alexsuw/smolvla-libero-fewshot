# Experiment decisions

Решения о pseudo-target calibration, training schedule и checkpoint selection
записываются сюда до запуска real target grid.

## Frozen 2026-08-21 (TODO 22)

Pseudo-target tasks are the three diverse `libero_90` language tasks already
listed in the expert-replay gate:

- `black_bowl_plate`: put the middle black bowl on the plate
- `drawer_bowl`: open the top drawer of the cabinet and put the bowl in it
- `book_caddy`: pick up the book and place it in the front compartment of the caddy

These texts are not the three held-out `libero_goal` targets. Episode prefixes
follow first-N in ascending `episode_index` once pinned metadata is present;
IDs are not invented offline.

Hyperparameters frozen from this set, not from real targets:

- seen expert: AdamW `1e-4`, cosine warmup 1000, min LR `1e-5`, 100k steps,
  effective batch 32, Action Expert + projections, VLM frozen
- target baseline: same optimizer, `min(100 epochs, 12000 steps)`, final
  checkpoint, no target-success early stopping
- LoRA: r=64, alpha=64, dropout 0, LR `1e-3`, min LR `1e-4`
  (target LoRA only; seen-FT LoRA / TODO 25 is skipped so the mandatory
  baseline is not delayed)
- Replay-LoRA: 75% target / 25% `libero_90` only
- eval: 20 rollouts, seeds 1000–1019, hard reset
- seen checkpoint: earliest probe-best within 0.02, else step 100000; hash
  remains unset until TODO 24

## Skip seen LoRA (TODO 25)

Seen-FT LoRA is not part of the mandatory path. Primary seen-pretrain stays
Action Expert + projections. Target baseline is target-only continuation from
the frozen seen checkpoint with the same scope: no LoRA, no replay. Target LoRA
train/eval **code** is in `train_target.py --config configs/train/target_lora.yaml`;
GPU LoRA cells wait until the 18-cell baseline exists on the VM. Replay-LoRA
**code** is `--config configs/train/target_replay_lora.yaml` (75/25 `libero_90`).

