## Base information

This is a financial library for machine learning prediction. It is intended for use with ML libraries like lightgbm, xgboost, catboost, sklearn, etc. Its main purpose is to provide multiple trading indicators and store them inside a feature vector passed to an ML model without additional memory allocation.

## Requirements of code

- minimal memory allocation
- great performance
- human readable code

## Coding Style & Naming Conventions

- Follow standard Rust style: 4-space indentation, `snake_case` for functions and modules, `UpperCamelCase` for types.
- Run `cargo fmt` at the end of edits to ensure consistent formatting.
- Do not produce duplicate code; refactor common logic into functions or modules as needed.
- In case of refactoring, ask details first to avoid unnecessary work and ensure alignment with project goals.

## Agent Behavior

- implement code with small changes, focusing on one aspect at a time.
- ask for clarifications if requirements are ambiguous.
- Apply clippy suggestions.
- Do not suggest skipping or disabling CI checks
