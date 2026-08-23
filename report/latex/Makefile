.RECIPEPREFIX := >
TECTONIC ?= tectonic
PYTHON ?= ../../.venv/bin/python

.PHONY: all figures paper

all: paper

figures:
>$(PYTHON) scripts/make_figures.py

paper:
>mkdir -p build
>$(TECTONIC) paper.tex --outdir build --keep-logs
