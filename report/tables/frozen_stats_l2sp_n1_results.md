# Matched N=1: Frozen-Stats FT and Anchored FT (L2-SP)

Status: complete on 2026-08-23.

## Frozen protocol

- Both methods start independently from the same frozen seen checkpoint
  `step_100000`, SHA-256
  `2cd510a594a87580f7368b782ca9b37332c0e5002d807093c759e95fbfb57c88`.
- Dataset revision: `nvidia/LIBERO_LeRobot_v3` at
  `e5907374380b8f96511957e6ba5582be52a1e179`.
- The first target demonstration is fixed before training: drawer episode 20,
  bowl episode 13, and wine episode 6.
- Train seeds are 42 and 123. Target evaluation uses seeds 1000--1019.
- Corrected retention uses the frozen `libero_90` probes
  `black_bowl_plate`, `drawer_bowl`, and `book_caddy`, with seeds 1000--1009.
- Training and deployment use only the canonical `libero_90` normalization
  digest
  `b159b6fed3e52edf25bd39b377dd64940221b7a030362daf7f726b1c2ecb30cf`.
  No target-overlay statistics are fitted or loaded.
- Both methods train the same Action Expert and projection scope: 99,880,992
  parameters. Anchored FT adds the preregistered FP32 raw-sum L2-SP penalty
  with `lambda=0.01` relative to the frozen seen initialization.
- Naive N=1 is reused from the existing validated artifacts; it is not rerun.
  No method is selected, tuned, or rerun using target or retention success.

## Aggregate comparison

Training time is observed process wall-clock per cell. It includes contention
from the four-job grid and is therefore not an isolated microbenchmark.

| Method | Target success | Seen retention | Delta target | Delta retention | Train wall/cell | Trainable params | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive N=1 | 109/120 (90.8%) | 37/180 (20.6%) | -- | -- | 162.8 s | 99,880,992 | 7,540 MiB |
| Frozen-Stats FT N=1 | 109/120 (90.8%) | 39/180 (21.7%) | +0.0 pp | +1.1 pp | 311.3 s | 99,880,992 | 7,540 MiB |
| Anchored FT (L2-SP) N=1 | 105/120 (87.5%) | 57/180 (31.7%) | -3.3 pp | +11.1 pp | 297.3 s | 99,880,992 | 8,036 MiB |

The complete 12-cell grid took 47m39s: training finished in 15m57s,
target evaluation in a further 8m58s, and corrected retention in a further
22m44s. There were no failed stages. The training concurrency benchmark
measured 194.0 aggregate samples/s with two jobs and 202.7 samples/s with
four jobs (+4.4%), so the grid used four-way concurrency. Four-way training
reserved approximately 33--35 GiB of the 97.9 GiB GPU.

## Per-task and per-seed results

Target is out of 20 rollouts. Retention is out of 30 rollouts: three probes
times ten fixed seeds.

| Method | Target task | Seed | Target | Retention | Train wall | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| Naive | drawer_middle | 42 | 17/20 | 4/30 | 199.1 s | 7,540 MiB |
| Naive | drawer_middle | 123 | 19/20 | 2/30 | 199.4 s | 7,540 MiB |
| Naive | bowl_stove | 42 | 20/20 | 10/30 | 162.2 s | 7,540 MiB |
| Naive | bowl_stove | 123 | 20/20 | 10/30 | 162.0 s | 7,540 MiB |
| Naive | wine_cabinet | 42 | 16/20 | 6/30 | 127.2 s | 7,540 MiB |
| Naive | wine_cabinet | 123 | 17/20 | 5/30 | 126.9 s | 7,540 MiB |
| Frozen-Stats FT | drawer_middle | 42 | 20/20 | 4/30 | 378.3 s | 7,540 MiB |
| Frozen-Stats FT | drawer_middle | 123 | 20/20 | 4/30 | 376.9 s | 7,540 MiB |
| Frozen-Stats FT | bowl_stove | 42 | 20/20 | 4/30 | 326.1 s | 7,540 MiB |
| Frozen-Stats FT | bowl_stove | 123 | 20/20 | 5/30 | 325.5 s | 7,540 MiB |
| Frozen-Stats FT | wine_cabinet | 42 | 15/20 | 12/30 | 230.4 s | 7,540 MiB |
| Frozen-Stats FT | wine_cabinet | 123 | 14/20 | 10/30 | 230.4 s | 7,540 MiB |
| Anchored FT (L2-SP) | drawer_middle | 42 | 20/20 | 7/30 | 392.3 s | 8,036 MiB |
| Anchored FT (L2-SP) | drawer_middle | 123 | 20/20 | 7/30 | 392.3 s | 8,036 MiB |
| Anchored FT (L2-SP) | bowl_stove | 42 | 19/20 | 7/30 | 314.8 s | 8,036 MiB |
| Anchored FT (L2-SP) | bowl_stove | 123 | 17/20 | 10/30 | 314.7 s | 8,036 MiB |
| Anchored FT (L2-SP) | wine_cabinet | 42 | 14/20 | 12/30 | 185.2 s | 8,036 MiB |
| Anchored FT (L2-SP) | wine_cabinet | 123 | 15/20 | 14/30 | 184.5 s | 8,036 MiB |

Retention totals by frozen probe:

| Method | black_bowl_plate | drawer_bowl | book_caddy |
|---|---:|---:|---:|
| Naive | 21/60 | 14/60 | 2/60 |
| Frozen-Stats FT | 15/60 | 22/60 | 2/60 |
| Anchored FT (L2-SP) | 21/60 | 33/60 | 3/60 |

## Integrity evidence

- Final driver summary reports `integrity_ok=true`; all 36 stage jobs returned
  zero and all 12 final checkpoints have distinct weight hashes.
- Independent recount from raw JSONL agrees with the driver: 109/120 and
  39/180 for Frozen-Stats FT; 105/120 and 57/180 for Anchored FT.
- Raw records contain exactly the registered train/eval seeds, target tasks,
  retention probes, checkpoint hashes, and one canonical normalization digest.
- Preflight action audit found finite values and normalized gripper range
  `[-1.059968, 0.943425]`; the drawer demonstration's constant open-gripper
  value maps to `0.943425` and is valid data rather than clipping or leakage.
- Implementation commit: `f77c469c5f90a2b7bba37988b39848b0b7101abe`.
- Current full test suite: 304 passed, 8 skipped, 0 failed.

Evidence roots:

- summary: `/mnt/vla/validation/TODO32/summary.json`
- status and timing: `/mnt/vla/validation/TODO32/grid_status.json`
- checkpoints: `/mnt/vla/runs/target_matched_n1`
- target evaluation: `/mnt/vla/eval/target_matched_n1`
- corrected retention: `/mnt/vla/eval/seen_retention_libero90_matched_n1`

## Research interpretation

Frozen statistics do not explain the main forgetting effect. They preserve
the pooled Naive target score exactly, but improve retention by only 1.1
percentage points. The pooled equality also hides a task redistribution:
Drawer improves by four target successes while Wine loses four.

L2-SP produces a clear target--retention tradeoff at the preregistered
strength: relative to Naive it gains 20 retention successes (+11.1 pp) at a
cost of four target successes (-3.3 pp) and 496 MiB peak VRAM. The retention
gain is not uniform, however: 19 of the 20 additional successes over Naive
come from `drawer_bowl`; `black_bowl_plate` is unchanged and `book_caddy`
improves by only one. The result is promising evidence for anchoring, not yet
evidence of broad seen-task preservation.

Recommended next steps are to keep this result untouched, expand retention to
a preregistered broader set of seen tasks, and add train seeds before making a
method claim. Any future lambda comparison should use separate calibration
tasks rather than these target tasks or three probes. Useful mechanistic
follow-ups are parameter-drift measurements by Action Expert component and a
matched function-space distillation baseline on independently sampled
`libero_90` observations.
