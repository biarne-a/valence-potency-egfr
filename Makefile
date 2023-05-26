# -*- mode: makefile -*-
install:
	mamba env create -f env.yaml


run:
	python3 src/main.py


clean:
	./scripts/clean.sh
