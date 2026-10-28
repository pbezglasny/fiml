# Release 0.4.0 checklist

Target tag: `v0.4.0`. Rust and Python packages ship together on 2026-10-28.
The previous release, [v0.3.0](https://github.com/pbezglasny/fiml/releases/tag/v0.3.0),
was published on 2026-10-24 from `a1fcacc4c3954793000572c827ccfb01852aa1d1`.

## Release records

- [x] Set both Cargo package versions to `0.4.0`; Python derives its version
  from the bindings manifest. Preserve the ignored Cargo lockfile policy.
- [x] Document the order-book range change and Rust migration in
  [CHANGELOG.md](../CHANGELOG.md) and [release notes](releases/v0.4.0.md).
- [x] Update maintained installation guides and Russian translations, including
  the translation index. Set the release date in both changelogs and notes.
- [x] Retain Rust 1.89, CPython 3.12+, the five wheel targets, extractor JSON
  `2.0`, and pipeline JSON `4.0`.

## Candidate validation

Record the candidate SHA, commands, workflow URLs, artifact inventory and
fresh-install results in the release PR. These checkboxes describe the required
procedure; the PR and workflow results provide evidence for the exact commit.
Do not reuse validation results from 0.3.0. Any later commit needs revalidation.

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo +1.89.0 test --all-features`
- [ ] `cargo +1.89.0 clippy --workspace --all-targets --all-features -- -D warnings`
- [ ] `RUSTDOCFLAGS="-D warnings -D missing_docs" cargo +1.89.0 doc -p fiml --all-features --no-deps`
- [ ] `RUSTUP_TOOLCHAIN=1.89.0 just test-python`
- [ ] `RUSTUP_TOOLCHAIN=1.89.0 just test-notebook`
- [ ] Run ordinary PR CI and dispatch `ci.yml` on the candidate branch; require
  every check to pass and the dispatch run's `head_sha` to match the candidate.
- [ ] Dispatch `release.yml` on that commit without publishing; require all
  packaging jobs and Linux wheel API tests to pass.
- [ ] Inspect seven artifacts: one crate, one sdist, and Linux x86-64/AArch64,
  macOS x86-64/AArch64, Windows x86-64 wheels. Check versions, ABI tags,
  metadata, README/license content, and required source files.
- [ ] Install the downloaded Linux x86-64 wheel and sdist in separate fresh
  CPython 3.12 environments. Run quickstarts, the order-book migration examples,
  and Python-to-Rust named-output replay against the downloaded crate.

## Tagging and publishing

Pushing `v0.4.0` starts `.github/workflows/release.yml`, which publishes to
crates.io and PyPI, then creates the GitHub release with seven assets and
`docs/releases/v0.4.0.md` as its body. Do not upload packages manually first.

- [ ] Merge the release PR after checks pass. Verify the merged tree matches
  the validated candidate and the final commit has passing CI.
- [ ] Confirm registry publishing credentials are available, tag the validated
  release commit as `v0.4.0`, and push the tag.
- [ ] Verify the publishing workflow, both registries, and all seven GitHub
  release assets report `0.4.0`.
- [ ] Verify fresh `cargo add fiml@0.4.0` and `pip install fiml==0.4.0`
  installations and the order-book range API.
