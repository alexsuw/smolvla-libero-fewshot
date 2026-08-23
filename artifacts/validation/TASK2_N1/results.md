# Task 2: N=1 Target-LoRA and Replay-LoRA

Status: complete on 2026-08-23.

## Frozen protocol

- Base checkpoint:
  /mnt/vla/runs/seen__expert__libero90__nall__s42__20260822T010019Z__gd4b8fb8/checkpoints/step_100000
  with SHA-256
  2cd510a594a87580f7368b782ca9b37332c0e5002d807093c759e95fbfb57c88.
- Dataset: nvidia/LIBERO_LeRobot_v3 at
  e5907374380b8f96511957e6ba5582be52a1e179.
- Target demonstrations: the first selected target episode only:
  drawer 20, bowl 13, wine 6.
- Train seeds: 42, 123. No success-based tuning or reruns.
- Target evaluation: seeds 1000..1019, 20 rollouts per cell,
  libero_goal target stats and the same init-state fingerprints as Naive.
- Corrected retention: three frozen probes
  black_bowl_plate, drawer_bowl, book_caddy; seeds
  1000..1009; original frozen probe states; libero_90 stats digest
  b159b6fed3e52edf25bd39b377dd64940221b7a030362daf7f726b1c2ecb30cf.
- Replay-LoRA used a fixed 75% target / 25% seen mixture. Seen replay came
  only from the full libero_90 source and was not selected using the probes.
- The existing Naive N=1 artifacts were reused and were not rerun.

## Aggregate comparison

| Method | Target success | Corrected seen retention | Trainable parameters | Mean train wall/cell | Sum train wall | Peak reserved VRAM/cell |
|---|---:|---:|---:|---:|---:|---:|
| Naive N=1 | 109/120 (90.8%) | 37/180 (20.6%) | 99,880,992 | 162.8 s | 976.8 s | 7,540 MiB |
| Target-LoRA N=1 | 99/120 (82.5%) | 19/180 (10.6%) | 4,215,632 | 244.0 s | 1,463.9 s | 6,944 MiB |
| Replay-LoRA N=1 | 67/120 (55.8%) | 2/180 (1.1%) | 4,215,632 | 309.3 s | 1,856.0 s | 6,944 MiB |

LoRA trains 4.22% as many parameters as Naive (23.69x fewer; 95.78%
reduction) and reduced per-process peak reserved VRAM by 596 MiB (7.9%).
Observed wall times include concurrent load and are not isolated
microbenchmarks. The official 12-cell LoRA training grid took 1,007.7 s
(16m48s) from first start to last finish at concurrency four.

## Per-task and per-seed results

Target is out of 20. Retention is out of 30. Wall is manifest
finished_at_utc - started_at_utc.

| Method | Target task | Seed | Target | Retention | Train wall | Peak reserved VRAM |
|---|---|---:|---:|---:|---:|---:|
| Naive | drawer_middle | 42 | 17/20 | 4/30 | 199.1 s | 7,540 MiB |
| Naive | drawer_middle | 123 | 19/20 | 2/30 | 199.4 s | 7,540 MiB |
| Naive | bowl_stove | 42 | 20/20 | 10/30 | 162.2 s | 7,540 MiB |
| Naive | bowl_stove | 123 | 20/20 | 10/30 | 162.0 s | 7,540 MiB |
| Naive | wine_cabinet | 42 | 16/20 | 6/30 | 127.2 s | 7,540 MiB |
| Naive | wine_cabinet | 123 | 17/20 | 5/30 | 126.9 s | 7,540 MiB |
| Target-LoRA | drawer_middle | 42 | 15/20 | 0/30 | 314.7 s | 6,944 MiB |
| Target-LoRA | drawer_middle | 123 | 8/20 | 1/30 | 315.0 s | 6,938 MiB |
| Target-LoRA | bowl_stove | 42 | 20/20 | 9/30 | 229.7 s | 6,944 MiB |
| Target-LoRA | bowl_stove | 123 | 20/20 | 7/30 | 229.3 s | 6,944 MiB |
| Target-LoRA | wine_cabinet | 42 | 19/20 | 1/30 | 187.0 s | 6,882 MiB |
| Target-LoRA | wine_cabinet | 123 | 17/20 | 1/30 | 188.2 s | 6,944 MiB |
| Replay-LoRA | drawer_middle | 42 | 3/20 | 0/30 | 435.9 s | 6,944 MiB |
| Replay-LoRA | drawer_middle | 123 | 3/20 | 0/30 | 436.0 s | 6,944 MiB |
| Replay-LoRA | bowl_stove | 42 | 20/20 | 1/30 | 332.3 s | 6,944 MiB |
| Replay-LoRA | bowl_stove | 123 | 20/20 | 1/30 | 332.3 s | 6,944 MiB |
| Replay-LoRA | wine_cabinet | 42 | 8/20 | 0/30 | 159.7 s | 6,944 MiB |
| Replay-LoRA | wine_cabinet | 123 | 13/20 | 0/30 | 159.9 s | 6,944 MiB |

Retention probe totals across all six cells:

| Method | black_bowl_plate | drawer_bowl | book_caddy |
|---|---:|---:|---:|
| Naive | 21/60 | 14/60 | 2/60 |
| Target-LoRA | 12/60 | 7/60 | 0/60 |
| Replay-LoRA | 2/60 | 0/60 | 0/60 |

## Concurrency benchmark

The benchmark was bounded to the requested 15--20 minutes. Existing safe
settings were retained: batch 32, four data workers, bf16, no compile, no
fused optimizer, and videos/traces disabled for bulk evaluation. Batch
64/128 would change the frozen update/warmup semantics; compile was not
compatible; fused AdamW had no measured gain.

Forty-step smoke runs took about 15.6 s for isolated Target-LoRA and 23.0 s
for isolated Replay-LoRA. Four-way runs took about 24.4 s and 30.4 s per
process respectively, while improving aggregate throughput by about 1.6x
over concurrency two. Four processes reserved approximately 27,776 MiB
in total on the 97 GB RTX PRO 6000, so train and evaluation concurrency
were fixed at four.

Evidence roots:

- benchmark: /mnt/vla/validation/TASK2_N1/bench
- train: /mnt/vla/runs/task2_n1
- target eval: /mnt/vla/eval/task2_n1/target
- corrected retention: /mnt/vla/eval/task2_n1/retention_libero90

## Integrity and interpretation

All 12 training cells are complete, have unique final weight hashes, use the
same frozen base SHA, and record clean git commit
cebe04d9ab408eaa37c7ab0249d48fe915ae7b36. All 240 target records and
360 retention records passed exact checks for checkpoint SHA, method,
task/train/eval seeds, first-demo episode, stats suite/digest, and frozen
init-state fingerprint.

Neither LoRA method beats Naive under this frozen protocol. Target-LoRA is
parameter-efficient but loses 8.3 target percentage points and 10.0
retention points. It preserves Bowl target performance and improves one
Wine seed, but is unstable on Drawer. Replay-LoRA loses 35.0 target points
and 19.4 retention points versus Naive; the fixed 25% seen replay does not
preserve the frozen policy.

The Replay-LoRA Drawer cells expose a concrete normalization pathology:
the single target demonstration has gripper action std 1e-6, while
seen replay contains both gripper states. Applying the frozen target
normalizer to replay produces extreme standardized targets; final losses
are 1.44e10 and 2.72e10. This explains the Drawer failure and makes
source-aware/common-scale normalization the first preregistered follow-up.
It does not fully explain Bowl/Wine retention collapse, so the next
diagnostic should also log target/replay loss and gradient norms separately
by source and action dimension.

Recommended research continuation:

1. Run a no-training replay batch audit: source counts, per-source/per-action
   normalized ranges, loss, and gradient norms.
2. Preregister a normalization-safe replay ablation (one common frozen action
   coordinate system or explicitly source-aware transforms), keeping replay
   selection independent of the three probes.
3. Evaluate retention on a broader predeclared set of seen tasks only after
   the method is locked; three probes are useful sentinels but too small for
   method selection.
4. Add at least three more train seeds for the final chosen method because
   the Target-LoRA Drawer result varies from 15/20 to 8/20.
5. Compare replay with a frozen-policy regularizer or distillation objective
   on unselected libero_90 samples, using a preregistered grid and reporting
   the full target-retention Pareto frontier rather than choosing by these
   target successes.
