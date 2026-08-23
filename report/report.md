# Final report

The frozen preregistration contained no final-target numerical claims; all
numerical claims were added only after their evaluation grids completed.

- [Unified four-page paper and technical appendix](latex/build/paper.pdf)
- [LaTeX source and build instructions](latex/README.md)

The main result is a sharp target--retention trade-off. Naive one-shot
adaptation reaches 90.8% target success and 20.6% corrected seen retention.
Target-LoRA and Replay-LoRA do not improve this trade-off. Frozen statistics
alone change retention by only +1.1 points. L2-SP reaches 87.5% target success
and 31.7% retention: +11.1 retention points for -3.3 target points relative
to Naive.

The first four pages follow the result-to-question narrative and contain the
updated efficiency frontier, three real rollout strips, failures, predictions,
and conclusion. The same PDF then gives exact protocols, percentage-and-count
tables, per-task/per-seed results, training curves, hashes, and integrity
evidence.
