# Reproducibility

This appendix is filled with live commands, hashes, and hardware after the
GPU grid. The protocol below is frozen now.

## Pins

- Dataset: `nvidia/LIBERO_LeRobot_v3` @ `e5907374380b8f96511957e6ba5582be52a1e179`
- Model: `lerobot/smolvla_base` @ `c83c3163b8ca9b7e67c509fffd9121e66cb96205`
- LeRobot git: `d451fe4f1f1b00a812f95aa9534389b5e42ab155`
- Predictions: `predictions.md` (committed before any target results)
- Pseudo-target freeze: `configs/splits/pseudo_target_splits.json`
- Hyperparameters: `configs/calibration.yaml`
- Seen checkpoint: `configs/selected_seen_checkpoint.yaml` (hash pending TODO 24)

## Commands

```bash
make check-reporting
python scripts/collect_results.py --runs-root "$VLA_RUNS_DIR" --output-dir report/tables
python scripts/plot_cost_curve.py --long report/tables/results_long.csv
python scripts/make_report_tables.py --long report/tables/results_long.csv --bundle
python scripts/sync_artifacts.py --source <run> --destination "$VLA_OBJECT_URI" --execute
python scripts/verify_backup.py --object-uri "$VLA_OBJECT_URI" --source <run>
```

Cost-curve figures are SVG with x ticks exactly `{0,5,10,25}`. PDF export is
optional once a plotting extra is added; it is not required for the CPU
report contract.

## Known deviations

- M1 hardware acceptance is still `resolved_m1_pending_hardware`.
- Live LIBERO/SmolVLA eval and 100k seen-pretrain are deferred until that gate.
- Wrist image transform remains identity relative to the dataset/policy key.
