# Release 0.3.0 checklist

Target tag: `v0.3.0`. Rust and Python packages ship together.
The previous release, [v0.2.0](https://github.com/pbezglasny/fiml/releases/tag/v0.2.0),
was published on 2026-10-02 from `e685063845ba4c0f52720a5a18eb31e0ff8443ca`.

## Release records

- [x] Set both Cargo package versions to `0.3.0`; Python derives its version
  from the bindings manifest. Preserve the ignored Cargo lockfile policy;
  the notebook lockfile uses the editable package without a pinned version.
- [x] Document changes since 0.2.0 in [CHANGELOG.md](../CHANGELOG.md).
- [x] Prepare [release notes and migration examples](releases/v0.3.0.md),
  read by `release.yml`; update maintained installation and Python API guides.
- [x] Mirror documentation in Russian and update its index.
- [x] Retain Rust 1.89, CPython 3.12+, and the five wheel targets.
- [ ] Replace `Unreleased` in the changelog and the pending publication date
  in the release notes (both languages) when the date is known; validate again.

## Local validation

Preparation checks on 2026-10-24 used Rust 1.89.0 and CPython 3.13.5.
These results describe the preparation worktree, not final-commit approval.
Local packaging may use `--allow-dirty`; the release workflow must package a
clean commit. No old 0.2.0 test counts or packaging results carry forward.

- [x] `cargo fmt --all -- --check`
- [x] `cargo +1.89.0 test --all-features`: 352 tests and 2 doctests.
- [x] `cargo +1.89.0 clippy --workspace --all-targets --all-features -- -D warnings`
- [x] `RUSTDOCFLAGS="-D warnings -D missing_docs" cargo +1.89.0 doc -p fiml --all-features --no-deps`
- [x] `RUSTUP_TOOLCHAIN=1.89.0 just test-python`: 932 tests on CPython 3.13.5.
- [x] `RUSTUP_TOOLCHAIN=1.89.0 just test-notebook`: execution and HTML export.
- [x] Package and verify the Rust crate; build a release wheel and sdist.
- [x] Inspect versions, metadata, README/license content, and required source files.
- [x] Install wheel and sdist into separate fresh CPython 3.12 environments;
  run maintained quickstarts, named-output examples, and the migration examples.
- [x] Build and run the README Rust example against the extracted crate and
  a Python-exported pipeline; compare the named output (`1.224745`).

The local crate contained 114 files. Maturin 1.15.0 built the local wheel and
sdist; the workflow retains Maturin 1.14.1. Fresh CPython 3.12.15 installations
passed the quickstarts, migration snippets, and Python-to-Rust README replay;
the wheel also passed all 932 Python tests. The local Linux wheel targets
manylinux 2.34; only the workflow establishes the required manylinux 2.17 builds.

## Final validation record

Record the final candidate SHA, CI and Release workflow URLs, artifact inventory,
checksums, supported versions, and fresh-install results in the release PR body.
Link that record here before freezing the candidate. Results for that exact SHA
are authoritative; the checklist below is the procedure, not a claim of success.
Keeping final run evidence in the PR avoids changing the tested commit merely
to record its own SHA. Any subsequent commit requires a new validation record.

- [ ] Push the candidate and open the release-preparation PR referencing #202.
- [ ] Dispatch `ci.yml` on the candidate branch; verify the run's `head_sha`
  equals the intended commit and every existing check passes. Ordinary PR CI
  also remains required; its merge checkout is not exact-head evidence.
- [ ] Dispatch `release.yml` on the same branch/commit. Confirm no publishing
  jobs run and all packaging jobs, including Linux wheel API tests, pass.
- [ ] Download and inspect exactly seven artifacts: one crate, one sdist, and
  Linux x86-64/AArch64, macOS x86-64/AArch64, Windows x86-64 wheels.
- [ ] Check 0.3.0 versions, CPython 3.12 ABI tags, metadata, README/license content,
  and required sources across the downloaded packages.
- [ ] Install the downloaded Linux x86-64 wheel and sdist in separate fresh
  environments; run quickstarts and named-output Python-to-Rust replay against
  the downloaded crate. Record results in the final validation record.

## Tagging and publishing (separate step)

Manual workflow dispatch builds artifacts without publishing. Pushing `v0.3.0`
starts `.github/workflows/release.yml`, which publishes to crates.io and PyPI,
then creates the GitHub release. Do not upload packages manually first.
Preparation under #202 stops before merging, tagging, or publishing.

- [ ] Confirm the publication date and exact commit have final green validation.
- [ ] Confirm `CARGO_REGISTRY_TOKEN` and PyPI Trusted Publishing are ready.
- [ ] Tag the validated commit as `v0.3.0` and push the tag.
- [ ] Verify both registries contain `fiml` 0.3.0 and the GitHub release contains
  seven assets and the intended release notes.
- [ ] Verify fresh `cargo add fiml@0.3.0` and `pip install fiml==0.3.0`
  installations, documentation, and registry metadata.
