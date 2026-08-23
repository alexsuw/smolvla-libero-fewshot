# Reproducibility

The complete reproducibility record is in the
[technical appendix](latex/build/appendix.pdf). Its source is
[appendix.tex](latex/appendix.tex).

## Pins

- Dataset: `nvidia/LIBERO_LeRobot_v3` @ `e5907374380b8f96511957e6ba5582be52a1e179`
- Model: `lerobot/smolvla_base` @ `c83c3163b8ca9b7e67c509fffd9121e66cb96205`
- LeRobot git: `d451fe4f1f1b00a812f95aa9534389b5e42ab155`
- Predictions: `predictions.md` (committed before any target results)
- Seen checkpoint: `step_100000`
- Seen weights SHA-256: `2cd510a594a87580f7368b782ca9b37332c0e5002d807093c759e95fbfb57c88`
- Seen statistics SHA-256: `b159b6fed3e52edf25bd39b377dd64940221b7a030362daf7f726b1c2ecb30cf`
- Target train seeds: `42, 123`
- Target evaluation seeds: `1000..1019`
- Retention probes: three frozen `libero_90` probes, seeds `1000..1009`

## Report build

From `report/latex`:

```bash
make figures
make paper
make appendix
```

The main paper is exactly four pages. The appendix contains all technical
training figures and per-task/per-seed tables. Frozen final evidence is also
summarized in `artifacts/validation/TASK2_N1/results.md`.

## Validation

- Main PDF: 4 pages, no overfull boxes or missing glyphs.
- Appendix PDF: 6 pages.
- Repository suite: 296 passed, 8 skipped.
