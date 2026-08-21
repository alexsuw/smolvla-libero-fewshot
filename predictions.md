# Predictions

Committed **before** any final target-task results. Do not edit numerical
claims here after the real grid starts; record actuals in `report/report.md`.

## Falsifiable hypothesis

Ожидается наибольший прирост success между `N=0` и `N=5`, меньший между
`N=5` и `N=10` и насыщение между `N=10` и `N=25`. Parameter-efficient
adaptation с seen replay должна иметь наибольшее преимущество при `N=5`,
где naive target-only fine-tuning наиболее подвержен overfitting и
forgetting; к `N=25` разрыв должен уменьшаться.

## Method ordering at N=5

On the three held-out `libero_goal` tasks, mean success at `N=5` is
predicted to rank:

1. Replay-LoRA (target 75% / `libero_90` 25%)
2. target LoRA
3. naive target-only Action Expert continuation
4. zero-shot from the frozen seen checkpoint (`N=0`)

The strong-result shape from the spec, if it appears at all, should show
Replay-LoRA at `N=5` approaching naive continuation at `N=10`.

## What would falsify this

- Replay-LoRA no better than naive continuation at `N=5` after Wilson
  intervals and the two train seeds are taken into account.
- The largest jump on the cost curve arriving after `N=10` rather than
  between `N=0` and `N=5`.
- Wrong-instruction language control matching correct-instruction success
  on the same initial states.

## What this file is not

These predictions are not hyperparameters. LR, steps, LoRA rank, replay
ratio and seen-checkpoint selection were frozen on `libero_90`
pseudo-target tasks only. Target success must not be used to revise them.
