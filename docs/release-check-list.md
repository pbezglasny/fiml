# Release 0.2.0 checklist

Target tag: `v0.2.0`. Rust and Python packages ship together.
The previous release, [v0.1.0](https://github.com/pbezglasny/fiml/releases/tag/v0.1.0),
was published on 2026-09-15 from `57619d5`.

## Release records

- [x] Set both Cargo package versions to `0.2.0`; Python derives its version
  from the bindings manifest.
- [x] Document new features and breaking changes in [CHANGELOG.md](../CHANGELOG.md).
- [x] Prepare [GitHub release notes](releases/v0.2.0.md), read by `release.yml`.
- [x] Update installation examples and link the [migration guide](sample-average-migration.md).
- [x] Keep Rust 1.89, CPython 3.12+, and the existing five wheel targets.
- [ ] Replace `Unreleased` for 0.2.0 in the changelog with the publication date.

## Local validation

Checked on 2026-10-02 in the release worktree. Repeat from the final clean
release commit before tagging. Local packaging uses `--allow-dirty` only to
validate the prepared, uncommitted changes; CI packages a clean checkout.

- [x] `cargo fmt --all -- --check`
- [x] `cargo +1.89.0 test --workspace --all-features`: 278 tests and 2 doctests.
- [x] `cargo +1.89.0 clippy --workspace --all-targets --all-features -- -D warnings`
- [x] `RUSTDOCFLAGS="-D warnings -D missing_docs" cargo +1.89.0 doc -p fiml --all-features --no-deps`
- [x] `make test-python`: 489 tests on CPython 3.13.
- [x] `make test-notebook`: marimo execution and export.
- [x] `cargo +1.89.0 package -p fiml --all-features --allow-dirty`: 88 files;
  the final local run used `--offline` after fetching dependencies.
- [x] Build a local release wheel and source distribution with Maturin 1.14.1.
- [x] Inspect crate, wheel, and source archive for version, source files, README,
  and license text.
- [x] Install the wheel in a fresh CPython 3.13 environment; all 489 Python
  tests and the quickstart pass.
- [x] Install the source distribution in a separate fresh environment and run
  the quickstart.
- [x] Build and run the README Rust example against the packaged crate and
  the pipeline exported by the README Python example; both return `1.224745`.

The local Linux x86-64 wheel validates this host only. The release workflow
builds manylinux 2.17 x86-64/AArch64, macOS x86-64/AArch64, and Windows x86-64
wheels; those platform builds remain a CI gate.

## Final CI and publishing

Pushing `v0.2.0` starts `.github/workflows/release.yml`, which publishes to
crates.io and PyPI, then creates the GitHub release. A manual workflow dispatch
builds artifacts without publishing. Do not upload packages manually first.

- [ ] Commit the prepared changes and obtain green CI for that exact commit.
- [ ] Run the release workflow manually on the release commit; verify the crate,
  all five wheels, and source distribution, including the Linux wheel API tests.
- [ ] Confirm the existing `CARGO_REGISTRY_TOKEN` and PyPI Trusted Publishing
  configuration are ready for the release.
- [ ] Tag the validated commit as `v0.2.0` and push the tag.
- [ ] Verify crates.io and PyPI both contain `fiml` 0.2.0 and the GitHub release
  contains seven assets and the intended release notes.
- [ ] Verify fresh `cargo add fiml@0.2.0` and `pip install fiml==0.2.0`
  installations, documentation, and registry metadata.
