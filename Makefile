PYTHON_VENV := .venv
PYTHON := $(PYTHON_VENV)/bin/python
MATURIN := $(PYTHON_VENV)/bin/maturin

.PHONY: build build-rust build-python test test-rust test-python test-notebook python-venv jupyter-lab

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
	cd notebooks && uv run --no-sync jupyter execute test.ipynb --timeout=120

build-python: python-venv
	cd crates/fiml-python && ../../$(MATURIN) build --release --out dist

jupyter-lab:
	cd notebooks && uv run jupyter lab --ip=0.0.0.0

python-venv: $(MATURIN)

$(MATURIN):
	uv venv $(PYTHON_VENV)
	uv pip install --python $(PYTHON_VENV)/bin/python "maturin>=1.5,<2.0"
