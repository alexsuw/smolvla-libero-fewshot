# Reproducibility

The complete reproducibility record is in the
[unified paper and appendix](latex/build/paper.pdf). Its single source is
[paper.tex](latex/paper.tex).

## Pins

- Dataset: nvidia/LIBERO_LeRobot_v3 at e5907374380b8f96511957e6ba5582be52a1e179
- Model: lerobot/smolvla_base at c83c3163b8ca9b7e67c509fffd9121e66cb96205
- LeRobot git: d451fe4f1f1b00a812f95aa9534389b5e42ab155
- Predictions: predictions.md, committed before target results
- Seen checkpoint: step_100000
- Seen weights SHA-256: 2cd510a594a87580f7368b782ca9b37332c0e5002d807093c759e95fbfb57c88
- Seen statistics SHA-256: b159b6fed3e52edf25bd39b377dd64940221b7a030362daf7f726b1c2ecb30cf
- Target train seeds: 42 and 123
- Target evaluation seeds: 1000 through 1019
- Retention probes: three frozen libero_90 probes, seeds 1000 through 1009

## Report build

From report/latex:

    make figures
    make paper

The first four pages are the main article. The remaining pages are the
technical appendix in the same PDF and source file. They contain exact
training figures, percentage-and-count success tables, per-task/per-seed
results, the normalization control, hashes, and artifact paths.

## Validation

- Unified PDF: 11 pages; first four pages are the main report.
- LaTeX log: no overfull boxes, missing glyphs, undefined references, or
  duplicate figure/table anchors.
- New matched grid: 600 rollouts independently recounted and
  integrity_ok=true.
- Repository suite: 304 passed, 8 skipped, 0 failed.
