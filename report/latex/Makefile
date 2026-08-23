.RECIPEPREFIX := >
TECTONIC ?= tectonic
PYTHON ?= ../../.venv/bin/python

.PHONY: all figures paper appendix

all: paper appendix

figures:
>$(PYTHON) scripts/make_figures.py

paper:
>mkdir -p build
>$(TECTONIC) paper.tex --outdir build --keep-logs

appendix:
>mkdir -p build
>$(TECTONIC) appendix.tex --outdir build --keep-logs
