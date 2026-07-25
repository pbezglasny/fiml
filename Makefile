PYTHON_VENV := .venv
PYTHON := $(PYTHON_VENV)/bin/python
MATURIN := $(PYTHON_VENV)/bin/maturin

.PHONY: test test-rust test-python build-python python-venv jupyter-lab

test: test-rust test-python

test-rust:
	cargo test --all-features

test-python: python-venv
	uv pip install --python $(PYTHON) --reinstall-package fiml "./crates/fiml-python[test]"
	$(PYTHON) -m pytest crates/fiml-python/tests

build-python: python-venv
	cd crates/fiml-python && ../../$(MATURIN) build --release --out dist

jupyter-lab:
	cd notebooks && uv run jupyter lab --ip=0.0.0.0

python-venv: $(MATURIN)

$(MATURIN):
	uv venv $(PYTHON_VENV)
	uv pip install --python $(PYTHON_VENV)/bin/python "maturin>=1.5,<2.0"
