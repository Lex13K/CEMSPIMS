.PHONY: help install test run-default

help:
	@echo "CEMSPIMS — see README.md"
	@echo ""
	@echo "  make install      pip install -e .[dev,train]"
	@echo "  make test         pytest (core tests)"
	@echo "  make run-default  full default pipeline"

install:
	pip install -e ".[dev,train]"

test:
	python -m pytest tests/

run-default:
	python scripts/run.py run --run default
