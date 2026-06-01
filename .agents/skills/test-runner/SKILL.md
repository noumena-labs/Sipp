---
name: test-runner
description: Runs the narrowest relevant tests to validate changes. Use this skill when the user asks to run tests, verify functionality, run checks, or before concluding any change in the repository.
compatibility: Requires cargo, bun/pnpm, and python testing suites.
allowed-tools: Bash(cargo:*) Bash(bun:*) Bash(npm:*) Bash(pnpm:*) Bash(pytest:*) Read Edit
---

# Test Runner Skill

You are responsible for validating changes in the repository using the appropriate testing framework.

## Core Rule

Always run the **narrowest relevant test suite** based on the files you modified. Avoid running full repository checkouts or testing suites that check unchanged packages, as this wastes resources and increases execution time.

---

## Test Suites by Target

Identify the files modified and run the corresponding command:

### 1. Rust Native Core (`crates/`)
- Run unit/integration tests for the affected crate:
  ```bash
  cargo test -p <crate_name>
  ```
- Example: `cargo test -p cogent-engine`

### 2. Node.js Bindings (`bindings/node/`)
- Run the smoke tests:
  ```bash
  node bindings/node/examples/node_smoke.mjs
  ```
- Run unit tests:
  ```bash
  bun test bindings/node/tests/
  ```

### 3. TypeScript NPM Packages (`packages/npm/`)
- Run tests in the specific package:
  ```bash
  pnpm --filter <package_name> test
  # Or with bun:
  bun run --cwd packages/npm/<pkg> test
  ```
- Check types:
  ```bash
  pnpm typecheck
  ```

### 4. Python Bindings (`bindings/python/`)
- Run pytest suite:
  ```bash
  python -m pytest bindings/python/tests/
  ```

---

## Pre-Test Check
Ensure that you build the necessary components first using the **`build-orchestrator`** skill before running their tests (e.g. Node bindings must be built before `node_smoke.mjs` will work).
