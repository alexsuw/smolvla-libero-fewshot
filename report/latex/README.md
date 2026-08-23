# LaTeX report

This folder contains the complete T-Lab 2026 SmolVLA few-shot adaptation
report. The four-page main article and the technical appendix are compiled
from one source into one PDF.

## Deliverables

- Unified paper and appendix: [build/paper.pdf](build/paper.pdf)
- Unified LaTeX source: [paper.tex](paper.tex)
- References: [references.bib](references.bib)
- Plot source: [scripts/make_figures.py](scripts/make_figures.py)
- Figures and real rollout frames: [imgs/](imgs/)

## Build

From this directory, with Tectonic available:

    make paper

To rebuild plots from the frozen experiment logs mounted under /mnt/vla:

    make figures

A different executable can be selected with TECTONIC=/path/to/tectonic.
The compiled figures and PDF are committed, so reading the report does not
require the private training mount.
