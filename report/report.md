# Final report


The frozen preregistration contained no final-target numerical claims; all
numerical claims below were added only after the evaluation grid completed.
The scientific report is complete:

- [Four-page main paper](latex/build/paper.pdf)
- [Technical appendix](latex/build/appendix.pdf)
- [LaTeX source and build instructions](latex/README.md)

The main result is a sharp target--retention trade-off. Naive adaptation with
one target demonstration reaches 109/120 (90.8%) target success, but only
37/180 (20.6%) corrected seen retention. Target-LoRA uses 23.7 times fewer
trainable parameters and reaches 99/120 target success; Replay-LoRA reaches
67/120. Neither low-rank method beats the matched naive baseline.

The four-page paper contains the required cost curve, preregistered predictions,
three real failure rollouts, idea graveyard, limitations, and bonus status.
