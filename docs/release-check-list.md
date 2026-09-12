# First release checklist

Target release: `v0.1.0`

The implementation is close to release-ready, but the package metadata,
documentation, and Python distribution still need work.

## Current status

- [x] `cargo fmt --all -- --check`
- [x] `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- [x] Rust workspace tests: 220 passed
- [x] Python tests: 380 passed
- [x] Notebook execution
- [x] `cargo package -p fiml`: 70 files, 118.9 KiB compressed
- [x] Current GitHub `main` build
- [x] No existing `fiml` package was found on crates.io or PyPI on 2026-09-12
- [ ] Strict Rust documentation check; `missing_docs` currently reports many
  undocumented public items

## Release blockers

### 1. Define the release scope

- [ ] Decide whether `v0.1.0` includes both the Rust crate and Python package.
- [ ] Decide which Python operating systems and architectures are supported.
  The current local build produces only a Linux x86-64 wheel.
- [ ] Define the minimum supported Rust version, or explicitly state that only
  the current stable Rust toolchain is supported.

### 2. Complete the first-release indicator set

Do not delay `v0.1.0` for a large indicator catalogue. Add only the missing
ML fundamentals, then stop:

- [ ] Simple and log returns over configurable sample lags.
- [ ] Rolling volatility as the standard deviation of returns over
  configurable windows.
- [ ] Rolling trade volume over timed windows.
- [ ] VWAP from trade price and volume over timed windows.

For every added indicator:

- [ ] Update the feature-vector builder, feature key, compiler, and runtime
  derivation.
- [ ] Update serialization, JSON schema, and canonical feature IDs.
- [ ] Update the Python builder and documentation.
- [ ] Add Rust/Python parity, warm-up, input-validation, and steady-state
  allocation checks.

Defer RSI, MACD, Bollinger Bands, ATR, ADX, and stochastic oscillators. Add
them after user demand; ATR, ADX, and stochastic oscillators should wait for a
proper OHLC/bar event model.

### 3. Complete Rust package metadata

- [ ] Add a root `README.md` containing:
  - purpose and development status;
  - installation and a minimal example;
  - supported indicators and transformations;
  - important limits and warm-up behavior;
  - Rust and Python support policy;
  - Apache-2.0 license notice.
- [ ] Add `description`, `license`, `repository`, `readme`, and `rust-version`
  to `crates/fiml/Cargo.toml`.
- [ ] Add useful crates.io `keywords` and `categories`.
- [ ] Make `cargo package -p fiml` complete without metadata warnings.

Cargo's official publishing guide recommends this metadata and a successful
package dry run:
<https://doc.rust-lang.org/cargo/reference/publishing.html>.

### 4. Complete Python package metadata and documentation

- [ ] Add the license, project URLs, maintainers, and supported Python-version
  classifiers to `crates/fiml-python/pyproject.toml`.
- [ ] Replace the `<repo-url>` placeholder in
  `crates/fiml-python/README.md`.
- [ ] Replace the statement that PyPI publishing is only planned.
- [ ] Build the supported wheels and a source distribution.
- [ ] Install and test the generated artifacts in fresh virtual environments.

### 5. Audit and document the public Rust API

- [ ] Decide which modules and types are intentionally public; make
  implementation details private before users depend on them.
- [ ] Add crate-level documentation with a minimal end-to-end example.
- [ ] Document the supported public types, traits, methods, errors, and limits.
- [ ] Add a CI documentation check once the public surface is documented:

  ```bash
  RUSTDOCFLAGS="-D warnings -D missing_docs" \
    cargo doc -p fiml --all-features --no-deps
  ```

### 6. Prepare release records

- [ ] Add `CHANGELOG.md` with the `0.1.0` features, limitations, and breaking
  change policy.
- [ ] Ensure the Rust and Python package versions are both `0.1.0`.
- [ ] Prepare concise GitHub release notes.
- [ ] Update the GitHub repository description and topics.

### 7. Prepare publishing

- [ ] Create and verify the crates.io account and publishing token.
- [ ] Configure PyPI Trusted Publishing with GitHub Actions instead of storing
  a long-lived PyPI token:
  <https://docs.pypi.org/trusted-publishers/>.
- [ ] Add a Python release workflow that builds wheels for every supported
  platform.
- [ ] Verify the exact files included in both Rust and Python artifacts.
- [ ] Confirm that the `fiml` names are still available immediately before
  publishing; registry names cannot be reserved by this checklist.

## Final validation

Run from a clean release commit:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo package -p fiml
make test-python
make test-notebook
```

Then test the packaged artifacts rather than importing or compiling directly
from the repository:

- [ ] Create a temporary Rust project and build it using packaged `fiml`.
- [ ] Create a fresh Python environment, install a built wheel, and run the
  quick-start example.
- [ ] Check that the README and license render correctly in both registries.

## Publish and verify

- [ ] Publish `fiml` to crates.io.
- [ ] Publish the Python wheels and source distribution to PyPI.
- [ ] Tag the published commit as `v0.1.0`.
- [ ] Create the GitHub release from that tag.
- [ ] Verify fresh `cargo add fiml` and `pip install fiml` installations.
- [ ] Verify the published documentation and package metadata.

## Deferred until needed

- Project website and logo.
- Complex release-management tooling.
- Large benchmark infrastructure beyond the existing performance checks.
- Compatibility automation for versions and platforms that are not part of
  the declared support policy.
