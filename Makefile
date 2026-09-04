PYTHON_VENV := .venv
PYTHON := $(PYTHON_VENV)/bin/python
MATURIN := $(PYTHON_VENV)/bin/maturin

.PHONY: build build-rust build-python test test-rust test-python test-notebook python-venv marimo

build: build-rust build-python

build-rust:
	cargo build -p fiml --all-features --release

test: test-rust test-python test-notebook

test-rust:
	cargo test --all-features

test-python: python-venv
	uv pip install --python $(PYTHON) --reinstall-package fiml "./crates/fiml-python[test]"
	$(PYTHON) -m pytest crates/fiml-python/tests

test-notebook:
	cd notebooks && uv sync
	uv pip install --python notebooks/.venv/bin/python --reinstall-package fiml ./crates/fiml-python
	cd notebooks && uv run --no-sync marimo export html test.py --output /tmp/fiml-test.html --force

build-python: python-venv
	cd crates/fiml-python && ../../$(MATURIN) build --release --out dist

marimo:
	cd notebooks && uv run marimo edit --host=0.0.0.0 --headless

python-venv: $(MATURIN)

$(MATURIN):
	uv venv $(PYTHON_VENV)
	uv pip install --python $(PYTHON_VENV)/bin/python "maturin>=1.5,<2.0"
