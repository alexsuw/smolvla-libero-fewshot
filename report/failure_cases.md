# Failure cases

Минимум три содержательно разные ошибки будут размечены после final evaluation.
Pipeline/environment defects не классифицируются как model failures. Слоты
зафиксированы до grid, чтобы отчёт не подбирал кейсы post hoc.

## 1. Language / instruction following

Wrong-instruction rollouts that keep the same initial state as the correct
pair. Discriminating check: action-chunk divergence plus success drop.

Status: pending live `language_control_v1` cells.

## 2. Spatial / object identity

A task that succeeds on one object but fails when the same verb is applied
to a nearby distractor (bowl vs stove vs plate). Discriminating check:
compare `bowl_stove` against the frozen `black_bowl_plate` probe.

Status: pending live `final_v1` cells.

## 3. Forgetting / few-shot overfitting at N=5

Naive target-only continuation that loses seen-domain skills or overfits
the five demonstrations. Discriminating check: Replay-LoRA vs baseline at
the same N, seeds, and eval protocol.

Status: pending live baseline and Replay-LoRA cells.
