python_venv := ".venv"
python := python_venv / "bin/python"
maturin := python_venv / "bin/maturin"

build: build-rust build-python

build-rust:
    cargo build -p fiml --all-features --release

test: test-rust test-python test-notebook

test-rust:
    cargo test --all-features

test-python: python-venv
    uv pip install --python "{{ python }}" --reinstall-package fiml "./crates/fiml-python[test]"
    "{{ python }}" -m pytest crates/fiml-python/tests

test-notebook:
    cd notebooks && uv sync
    uv pip install --python notebooks/.venv/bin/python --reinstall-package fiml ./crates/fiml-python
    cd notebooks && uv run --no-sync marimo export html test.py --output /tmp/fiml-test.html --force

build-python: python-venv
    cd crates/fiml-python && "../../{{ maturin }}" build --release --out dist

marimo:
    cd notebooks && uv run marimo edit --host=0.0.0.0 --headless

python-venv:
    if [ ! -f "{{ maturin }}" ]; then \
        uv venv "{{ python_venv }}" && \
        uv pip install --python "{{ python }}" "maturin>=1.5,<2.0"; \
    fi
