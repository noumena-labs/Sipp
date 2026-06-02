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

### 1. Broad and automation checks
- Run every cataloged suite:
  ```bash
  cargo xtask test run
  ```
- Run all white-box suites:
  ```bash
  cargo xtask test run --category whitebox
  ```
- Run xtask-only checks when the change is limited to developer automation:
  ```bash
  cargo xtask test run --suite xtask
  ```

### 2. Rust Native Core (`crates/`)
- Run cataloged unit tests for the affected crate:
  ```bash
  cargo xtask test run --suite rust-crates --package <crate_name>
  ```
- Example: `cargo xtask test run --suite rust-crates --package cogentlm-engine`

### 3. Node.js Bindings (`bindings/node/`)
- Run the cataloged Node interface tests:
  ```bash
  cargo xtask test run --suite node-package --backend cpu
  ```

### 4. TypeScript NPM Packages (`packages/npm/`)
- Run the cataloged browser package TypeScript tests:
  ```bash
  cargo xtask test run --suite package-ts
  ```
- App tests are cataloged separately:
  ```bash
  cargo xtask test run --suite app-ts
  ```

### 5. Python Bindings (`bindings/python/`)
- Run the cataloged Python interface tests:
  ```bash
  cargo xtask test run --suite python-package --backend cpu
  ```

### 6. Coverage and verification
- List the catalog before choosing a suite:
  ```bash
  cargo xtask test list --cases
  ```
- Run tests and produce run/coverage artifacts:
  ```bash
  cargo xtask test run --suite xtask
  ```
- Verify existing coverage artifacts and test structure:
  ```bash
  cargo xtask test verify --category whitebox
  ```
- Validate changed source files have matching catalog-owned tests:
  ```bash
  cargo xtask test verify --changed
  ```

---

## Pre-Test Check
The xtask test catalog builds required artifacts before suites that need them. Use the **`build-orchestrator`** skill first only when you are explicitly compiling or packaging a target outside the test catalog.

Use `cargo xtask test list --cases` to inspect available suites and discoverable cases before choosing a narrow command.
