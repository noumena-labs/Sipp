---
name: test-runner
description: Runs the narrowest relevant tests to validate changes. Use this skill when the user asks to run tests, verify functionality, run checks, or before concluding any change in the repository.
compatibility: Requires cargo, bun/pnpm, and python testing suites.
allowed-tools: Bash(cargo:*) Bash(bun:*) Bash(npm:*) Bash(pnpm:*) Bash(pytest:*) Read Edit
---

# Test Runner Skill

You are responsible for validating changes in the repository using the
appropriate testing framework.

## Core Rule

Always run the **narrowest relevant test target** based on the files you
modified. Avoid full-repo checks when a target-specific command covers the
change.

---

## Test Targets by Area

### 1. Broad and automation checks
- Run every deterministic unit suite:
  ```bash
  cargo xtask test unit group full
  ```
- Run all white-box unit suites:
  ```bash
  cargo xtask test unit group whitebox
  ```
- Run xtask-only checks when the change is limited to developer automation:
  ```bash
  cargo xtask test unit suite xtask
  ```

### 2. Rust Native Core (`crates/`)
- Run cataloged Rust unit tests for the affected crate:
  ```bash
  cargo xtask test unit suite rust-crates --package <crate_name>
  ```
- Example: `cargo xtask test unit suite rust-crates --package cogentlm`

### 3. Node.js Bindings And Package (`bindings/node/`, `lib/node/`)
- Run deterministic Node package API tests:
  ```bash
  cargo xtask test unit suite node-package --backend cpu
  ```
- Run model-backed Node smoke when local inference behavior changed:
  ```bash
  cargo xtask test smoke suite example-node --backend cpu
  ```

### 4. Browser Package And Demos (`lib/web/`, `demos/`)
- Run browser package TypeScript tests:
  ```bash
  cargo xtask test unit suite browser-package
  ```
- Demo tests are cataloged separately:
  ```bash
  cargo xtask test unit suite demos
  ```

### 5. Python Bindings And Package (`bindings/python/`, `lib/python/`)
- Run deterministic Python package API tests:
  ```bash
  cargo xtask test unit suite python-package --backend cpu
  ```
- Run model-backed Python smoke when local inference behavior changed:
  ```bash
  cargo xtask test smoke suite example-python --backend cpu
  ```

### 6. Browser and holistic smoke checks
- Run browser example smoke:
  ```bash
  cargo xtask test smoke suite example-browser
  ```
- Run browser playground runtime smoke:
  ```bash
  cargo xtask test smoke suite playground-browser
  ```
- Run CLI, Rust, Node, and Python model-backed smoke:
  ```bash
  cargo xtask test smoke group local-model --backend cpu
  ```
- Run llama.cpp backend correctness smoke:
  ```bash
  cargo xtask test smoke suite llama-backend-ops --backend cpu
  ```

### 7. Coverage and verification
- List the catalog before choosing a target:
  ```bash
  cargo xtask test list --cases
  ```
- Verify existing coverage artifacts and test structure:
  ```bash
  cargo xtask test verify --target whitebox
  ```
- Validate changed source files have matching catalog-owned tests:
  ```bash
  cargo xtask test verify --changed
  ```

---

## Pre-Test Check

The xtask test catalog builds required artifacts before suites that need them.
Use the **`build-orchestrator`** skill first only when you are explicitly
compiling or packaging a target outside the test catalog.

Use `cargo xtask test list --cases` to inspect available suites and discoverable
cases before choosing a narrow command.
