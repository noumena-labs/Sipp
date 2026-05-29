# CogentLM Style Guidance

This document defines the coding conventions, architectural boundaries, and best practices for the CogentLM monorepo. Agents and developers must adhere to these rules when submitting pull requests or generating code.

## 🦀 Rust Guidelines

### 1. Error Handling
- **Libraries (`crates/core`, `crates/engine`, `crates/shard`, etc.)**:
  - NEVER use `anyhow::Result`.
  - ALWAYS define a custom error enum using `thiserror::Error`.
  - Export a local `Result<T, Error>` alias.
  - Make error variants descriptive and enclose inner errors when necessary (e.g., `#[error("I/O error: {0}")] Io(#[from] std::io::Error)`).
- **Binaries & Scripts (`crates/cli`, `crates/xtask`)**:
  - `anyhow::Result` is allowed and encouraged for rapid bubbling of errors to the top-level binary.

### 2. Documentation
- Use `//!` at the top of every `lib.rs` to explain the crate's purpose and role in the workspace.
- Use `///` for public structs, enums, traits, and functions. 
- Include markdown examples in rustdoc where appropriate.

### 3. Module Visibility & Architecture
- Keep visibility as restrictive as possible (`pub(crate)` over `pub`).
- The `sys` crate is strictly for unsafe FFI bindings to C/C++. Higher-level crates should wrap these in safe abstractions.
- Avoid passing raw strings for errors; use typed parameters.
- Prefer explicit memory layouts (e.g., `#[repr(C)]`, `#[repr(u32)]`) when interfacing with WASM or native bindings.

---

## 🟦 TypeScript / JavaScript Guidelines

### 1. Strict Typing & NodeNext Modules
- The project targets `NodeNext` and `ES2022`.
- **Imports**: You MUST use explicit `.js` extensions for local file imports (e.g., `import { X } from './module.js';`). Do NOT omit the extension or use `.ts`.
- **Strict Mode**: `strict: true` is enabled. You must explicitly define types; no implicit `any`.

### 2. Immutability
- Prefer `readonly` for interfaces and arrays wherever possible.
  - Use `readonly string[]` instead of `string[]`.
  - Interface properties should be `readonly` if they are not meant to be mutated.

### 3. JSDoc and Comments
- Use JSDoc block comments `/** ... */` for interfaces, exported classes, and complex logic.
- Do not leave dead code or commented-out blocks.

### 4. Code Organization
- Tests must be placed alongside the implementation in `*.test.ts` files (not nested in a generic `__tests__` folder).
- Keep modules focused and export clear interfaces. Prefer exporting pure functions over complex class hierarchies unless managing state (e.g., `AgentRuntime`).

---

## 📂 Monorepo Organization

- **`crates/`**: Core Rust implementation.
- **`bindings/`**: FFI and language bindings (`node`, `python`, `wasm`).
- **`packages/`**: NPM packages intended for distribution (e.g., `@noumena-labs/cogentlm-browser`).
- **`apps/`**: Applications, examples, and benchmarks utilizing the engine.
