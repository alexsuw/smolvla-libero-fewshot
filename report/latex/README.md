# LaTeX report

This folder contains the complete source and compiled PDFs for the T-Lab 2026
SmolVLA few-shot adaptation report.

## Deliverables

- Main four-page paper: [build/paper.pdf](build/paper.pdf)
- Technical appendix: [build/appendix.pdf](build/appendix.pdf)
- Main source: [paper.tex](paper.tex)
- Appendix source: [appendix.tex](appendix.tex)
- References: [references.bib](references.bib)
- Plot source: [scripts/make_figures.py](scripts/make_figures.py)
- Figures and real rollout frames: [imgs/](imgs/)

## Build

From this directory, with Tectonic available:

    make paper
    make appendix

To rebuild the plots from the frozen experiment logs mounted under /mnt/vla:

    make figures

A different executable can be selected with TECTONIC=/path/to/tectonic.
The compiled figures and PDFs are committed, so reading the report does not
require the private training mount.
