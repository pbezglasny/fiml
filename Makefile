PYTHON_VENV := .venv
MATURIN := $(PYTHON_VENV)/bin/maturin

.PHONY: build-python python-venv jupyter-lab

build-python: python-venv
	cd crates/fiml-python && ../../$(MATURIN) build --release --out dist

jupyter-lab:
	cd notebooks && uv run jupyter lab --ip=0.0.0.0

python-venv: $(MATURIN)

$(MATURIN):
	uv venv $(PYTHON_VENV)
	uv pip install --python $(PYTHON_VENV)/bin/python "maturin>=1.5,<2.0"
