# CogentLM Style Guidance

This document defines coding conventions, architectural boundaries, and contribution standards for the CogentLM polyglot monorepo.

Agents and developers must follow these rules when generating code, reviewing changes, or submitting pull requests. The goal is to keep the codebase safe, maintainable, idiomatic in each language, and friendly to open-source contributors.

## General Principles

* Prefer clear, boring, maintainable code over clever abstractions.
* Keep changes small, focused, and reviewable.
* Match existing local conventions before introducing new patterns.
* Use the language's established community style as the default when this guide
  does not state a stricter CogentLM rule.
* Treat public APIs, exported types, CLI flags, config formats, and serialized data as compatibility surfaces.
* Keep configuration and build/deployment files explicit and human-managed.
  Do not generate or synthesize Docker Compose YAML, TOML, env files, CI YAML,
  or similar configuration from code or `xtask`; tooling may only validate or
  execute explicitly selected configuration files.
* Preserve the gateway configuration boundary: `.env` files contain secrets
  only, TOML files contain gateway application/runtime configuration, and
  Docker Compose YAML owns Docker build, image, port, mount, healthcheck, and
  orchestration settings.
* Add tests for behavior changes and bug fixes.
* Avoid unrelated formatting, refactors, or dependency changes.
* Document non-obvious decisions in code, docs, or pull request notes.
* Do not weaken type safety, lint rules, test coverage, or CI checks to make a change pass.

---

## Rust Guidelines

Rust code should follow the default Rust style described by the official
[Rust Style Guide](https://doc.rust-lang.org/style-guide/index.html), with
`rustfmt` defaults treated as the mechanical formatter of record. Public Rust
APIs should also follow the documentation expectations from the
[Rust API Guidelines](https://rust-lang.github.io/api-guidelines/documentation.html)
unless a local CogentLM rule below is stricter.

### 1. Error Handling

#### Libraries

For library crates such as `crates/core`, `crates/engine`, `crates/shard`, and similar reusable crates:

* Do not use `anyhow::Result` in library APIs.
* Define a crate-local custom error enum using `thiserror::Error`.
* Export a local result alias:

```rust
pub type Result<T> = std::result::Result<T, Error>;
```

* Make error variants descriptive and actionable.
* Preserve source errors with `#[from]` or `#[source]` where appropriate.

Example:

```rust
use thiserror::Error;

/// Errors returned by the engine crate.
#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to read model metadata from {path}")]
    ReadMetadata {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid shard index {index}; expected less than {shard_count}")]
    InvalidShardIndex {
        index: usize,
        shard_count: usize,
    },
}

pub type Result<T> = std::result::Result<T, Error>;
```

#### Binaries, examples, and scripts

For binaries and developer tooling such as `apps/cli`, `xtask`, examples, and one-off migration scripts:

* `anyhow::Result` is allowed.
* Prefer `anyhow::Context` for user-facing failure messages.
* Convert library errors into helpful top-level diagnostics.

### 2. Panics and Assertions

* Do not use naked `unwrap()` or `expect()` in production paths.
* Use explicit error handling, propagation, or typed validation instead.
* `unwrap()` and `expect()` are acceptable in tests when the failure would make the test invalid.
* Use `debug_assert!` for internal invariants that should not affect release behavior.
* Use `assert!` only when violating the condition indicates a programmer error.

### 3. Documentation

* Use `//!` at the top of every `lib.rs` to describe the crate's purpose and role in the workspace.
* Use `///` for public structs, enums, traits, functions, modules, and important associated items.
* Include rustdoc examples for public APIs where usage is not obvious.
* Keep examples compilable when practical.
* Document safety invariants on every `unsafe` function or block.
* Use `//!` only for crate-level and module-level documentation; use outer
  `///` rustdoc comments for public items.
* Put rustdoc comments before attributes such as `#[derive(...)]`, `#[napi]`,
  and `#[pyclass]`.
* Prefer line doc comments (`///`) over block doc comments (`/** ... */`).
* Write rustdoc summaries as complete sentence-style prose: start with a
  capital letter and end with punctuation.
* Keep standalone comment and doc-comment lines readable; prefer wrapping near
  the Rust style guide's 80-character comment recommendation and never exceed
  the normal formatted line width unless a URL or generated value requires it.
* Use `?` in fallible rustdoc examples instead of `unwrap()` or `expect()`.
* Document `# Errors`, `# Panics`, and `# Safety` sections when the API can
  return errors, panic for caller-controlled reasons, or requires unsafe
  invariants.

Example:

```rust
//! Core inference primitives for CogentLM.
//!
//! This crate contains model-agnostic types shared by the runtime,
//! engine, and language bindings.

/// Describes the layout of a loaded model shard.
pub struct ShardLayout {
    /// Number of tensors stored in the shard.
    pub tensor_count: usize,
}
```

### 4. Module Visibility and Architecture

* Keep visibility as restrictive as possible.
* Prefer `pub(crate)` over `pub` unless an item is part of a real public API.
* Avoid exposing implementation details through public modules.
* Keep `sys` crates limited to low-level unsafe FFI bindings.
* Wrap `sys` APIs in safe abstractions before exposing them to higher-level crates.
* Avoid passing raw strings for structured data or errors; use typed parameters.
* Prefer explicit memory layouts such as `#[repr(C)]` or `#[repr(u32)]` when interfacing with FFI, WASM, or native bindings.
* Keep unsafe code isolated, documented, and reviewed carefully.
* Keep browser WASM ABI structs in dedicated modules with compile-time layout checks when TypeScript reads their memory directly.

### 5. Async, Concurrency, and Performance

* Use async only where it provides clear value.
* Avoid blocking operations inside async tasks.
* Prefer explicit ownership and borrowing over unnecessary cloning.
* Do not introduce global mutable state unless there is no safer alternative.
* Use `Arc` intentionally; do not use it as a default escape hatch.
* For performance-sensitive changes, include benchmarks, measurements, or a clear rationale.

### 6. Testing

* Unit test bodies should live under crate-local `src/tests/` trees, not inline in runtime files or under `src/<subject>/tests/`.
* Runtime modules may keep minimal `#[cfg(test)] #[path = "..."] mod *_tests;` declarations and test-only helper hooks when private access is required.
* Rust test files must start with `//!` module-level rustdoc that states the source module under test, the behavior or edge cases covered, and any deterministic boundary such as fake handles, model-free paths, or intentionally skipped native/model execution.
* Do not use C-style block comments for Rust test-file purpose headers. Prefer concise `//!` paragraphs that rustdoc can render.
* When a source file needs private test access, put the `#[cfg(test)]` declarations directly below the `use` statements in a clearly marked `TESTS` section, followed by a `SRC` section before production code.
* Integration tests should live under crate-level `tests/` directories.
* Test public behavior, edge cases, and error paths.
* Avoid tests that depend on local machine state, network access, timing, or test order.
* Use deterministic fixtures where possible.

example:
```
/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
#[path = "tests/error_tests.rs"]
mod error_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////
```

---

## TypeScript and JavaScript Guidelines

TypeScript source should follow the project `tsconfig` settings and the
[Google TypeScript Style Guide](https://google.github.io/styleguide/tsguide.html)
where this guide is silent. JavaScript runtime shims should follow the
[Google JavaScript Style Guide](https://google.github.io/styleguide/jsguide.html)
where it does not conflict with the existing package style.

### 1. Strict Typing and NodeNext Modules

The project targets `NodeNext` and `ES2022`.

* Use explicit `.js` extensions for local file imports.

Correct:

```ts
import { createRuntime } from './runtime.js';
```

Incorrect:

```ts
import { createRuntime } from './runtime';
import { createRuntime } from './runtime.ts';
```

* `strict: true` is required.
* Do not introduce implicit `any`.
* Avoid explicit `any`; use `unknown`, generics, discriminated unions, or narrower types instead.
* Export explicit types at package boundaries.
* Do not weaken `tsconfig`, lint, or package-level type settings.

### 2. Immutability

* Prefer `readonly` for arrays, tuples, and object properties that should not be mutated.
* Use `readonly string[]` instead of `string[]` for immutable arrays.
* Use `ReadonlyArray<T>` when it improves readability.
* Mark interface properties `readonly` unless mutation is part of the contract.

Example:

```ts
export interface RuntimeOptions {
  readonly modelPath: string;
  readonly flags: readonly string[];
}
```

### 3. API Design

* Prefer small pure functions over complex class hierarchies.
* Use classes when managing state, lifecycle, resources, or identity.
* Keep package exports intentional and stable.
* Avoid default exports for shared library code unless the package already uses them consistently.
* Prefer discriminated unions for structured variants.
* Validate untrusted inputs at boundaries.

### 4. JSDoc, TSDoc, and Comments

* Use `/** ... */` JSDoc/TSDoc comments for public exports and other
  documentation intended for package users or editor/documentation tooling.
* Use `//` line comments for implementation notes intended only for maintainers.
* Avoid ordinary `/* ... */` block comments for implementation notes; prefer
  multiple `//` lines for multi-line implementation comments.
* Do not use decorative boxed comments for new TypeScript or JavaScript code.
* Explain why code exists, what inputs are valid, what the output means, and
  important side effects or failure modes. Do not restate types that TypeScript
  already expresses.
* Use Markdown inside JSDoc/TSDoc when structure is needed.
* Keep declaration comments aligned between source TypeScript, generated
  declaration files, and hand-maintained Node `.d.ts` files.
* Do not leave dead code or commented-out blocks.
* Keep TODO comments actionable and attributable when possible.

Example:

```ts
/**
 * Creates an AgentRuntime backed by a local CogentLM engine instance.
 */
export function createRuntime(options: RuntimeOptions): AgentRuntime {
  // ...
}
```

### 5. Tests

* Place package tests under dedicated package-level `tests/` folders, mirroring the `src/` tree when useful.
* Do not use a generic `__tests__` folder for new tests.
* Test public behavior rather than implementation details.
* Include tests for error cases and boundary conditions.
* Avoid snapshot tests for large unstable output unless they are clearly useful.

---

## Python Guidelines

Python may be used for bindings, tooling, tests, examples, packaging, or data-processing utilities.

Python code should follow [PEP 8](https://peps.python.org/pep-0008/) where
this guide is silent. Public docstrings should follow
[PEP 257](https://peps.python.org/pep-0257/).

### 1. Typing

* Use modern type hints for all public functions.
* Prefer `list[str]`, `dict[str, T]`, and `Path` over legacy or stringly typed APIs.
* Avoid untyped public APIs.
* Use `typing.Protocol`, `TypedDict`, or dataclasses where they clarify structure.
* Avoid `Any` unless there is no practical alternative.

### 2. Errors

* Raise specific exception types.
* Do not swallow exceptions silently.
* Preserve context when wrapping exceptions.
* Use `ValueError`, `TypeError`, `RuntimeError`, or custom exceptions intentionally.

### 3. Style

* Prefer `pathlib.Path` over raw string paths.
* Keep scripts importable and testable.
* Put command-line behavior behind a `main()` function.
* Avoid side effects at import time.
* Use deterministic behavior in tests and scripts.

Example:

```python
from pathlib import Path


def read_config(path: Path) -> str:
    """Read a UTF-8 configuration file."""
    return path.read_text(encoding="utf-8")
```

### 4. Docstrings

* Use triple double-quoted docstrings for public modules, classes, functions,
  constructors, and public methods.
* Use one-line docstrings only for obvious APIs that fit naturally on one line;
  otherwise use a summary line, a blank line, and concise detail.
* Phrase one-line docstrings as an imperative or effect summary such as
  "Return the active backend name."
* Do not repeat the signature in the docstring when type annotations or native
  binding signatures already expose it.
* Document arguments, return values, raised exceptions, side effects,
  environment variables, and call-order restrictions when they matter to a
  package user.

---

## C, C++, FFI, and Native Bindings

### 1. FFI Boundaries

* Keep unsafe or low-level native bindings isolated in dedicated `sys`, `native`, or binding-specific modules.
* Expose safe wrappers to higher-level Rust, TypeScript, Python, or WASM layers.
* Document ownership, lifetime, allocation, and deallocation rules.
* Do not pass ownership across FFI boundaries without an explicit contract.
* Prefer fixed-width integer types at ABI boundaries.
* Keep ABI-facing structs explicitly laid out.
* Keep the `crates/sys` CXX bridge focused on llama.cpp, ggml, mtmd, and native backend integration. Do not use it as a generic host-language bridge.
* Add CXX bridge methods only when Rust needs access to native backend behavior that cannot reasonably live in Rust.
* Keep C++ browser support limited to Emscripten, llama.cpp/sys backend objects, and narrowly scoped host shims. Do not add custom C++ layers between TypeScript and Rust browser APIs.
* Implement browser `CE_*` exports in Rust under `bindings/wasm/src/exports.rs`; keep shared browser ABI layouts in `bindings/wasm/src/abi.rs`.
* When adding or changing a browser `CE_*` export, update the Rust export, `bindings/wasm/CMakeLists.txt` export roots, and the TypeScript `WasmBridge` call site together.
* Keep Node bindings on napi-rs and Python bindings on PyO3/Maturin unless there is a deliberate architecture change.

### 2. Safety

* Validate all pointers, lengths, and enum values crossing language boundaries.
* Avoid undefined behavior even in error paths.
* Make cleanup idempotent where possible.
* Include tests or examples that exercise binding lifecycle behavior.
* Keep raw pointer and callback usage in the smallest practical unsafe scope.
* Validate raw WASM callback function pointers before adapting them into Rust traits or writers.
* Preserve explicit string-freeing contracts. Rust-owned browser strings are freed through `CE_FreeString`; borrowed pointers must have a documented lifetime and must not be freed by TypeScript.

---

## WASM Guidelines

* Keep WASM-facing APIs small, typed, and stable.
* Avoid leaking native implementation details into browser-facing packages.
* Prefer explicit serialization formats for data crossing the WASM boundary.
* Document memory ownership and performance-sensitive copies.
* Keep browser APIs compatible with modern evergreen browsers unless package documentation says otherwise.
* Preserve existing `CE_*` export names, status codes, scalar ABI choices, and memory layouts unless the change is intentionally breaking and all TypeScript readers are updated.
* Keep `CE_RequestId` and other ABI identifiers fixed-width. Use `#[repr(C)]` and size assertions for structs read from WASM memory.
* Prefer Rust-owned `#[no_mangle] extern "C"` exports for browser APIs on `wasm32-unknown-emscripten`.
* Do not introduce `wasm-bindgen` into the browser runtime while it depends on Emscripten for filesystem, WebGPU, and llama.cpp integration.
* Route Emscripten host callbacks through small JS library shims under `bindings/wasm/native/emscripten/`, not through broad C++ bridge layers.
* For callbacks passed from TypeScript with `Module.addFunction`, keep explicit `user_data`, validate callback presence at the exported boundary, and remove callbacks on the TypeScript side when finished.

---

## Documentation Guidelines

### 1. Repository Documentation

* Update documentation when behavior, configuration, public APIs, or workflows change.
* Keep `README.md` files useful for new contributors.
* Prefer concise examples that can be copied and run.
* Avoid documentation that depends on private services or local-only setup unless clearly marked.

### 2. API Documentation

* Document public APIs in the language-native style:

  * Rust: rustdoc.
  * TypeScript: JSDoc.
  * Python: docstrings.
* Include examples for APIs that are non-obvious or commonly misused.
* Mention errors, panics, side effects, and performance costs where relevant.

### 3. Open-Source Contributor Experience

* Keep setup instructions current.
* Prefer commands that work from a clean checkout.
* Document required tool versions.
* Avoid assuming access to private infrastructure for basic build, test, or lint workflows.

---

## Testing and Validation

### 1. Required Test Mindset

For every change, consider whether it needs:

* Unit tests.
* Integration tests.
* Binding tests.
* CLI tests.
* Browser or WASM tests.
* Documentation examples.
* Regression tests for fixed bugs.

### 2. Validation Commands

Use the narrowest relevant validation command available. Prefer package-specific or crate-specific checks over full-repo checks when possible.

Common commands may include:

```bash
cargo fmt
cargo clippy
cargo test
npm run typecheck
npm run lint
npm test
pnpm typecheck
pnpm lint
pnpm test
python -m pytest
python -m ruff check
python -m mypy
```

Use the package manager and task runner already used by the affected workspace.

### 3. CI Compatibility

* Do not bypass CI with local-only assumptions.
* Do not skip tests without a documented reason.
* Keep generated files deterministic.
* Avoid changes that require network access during normal tests unless explicitly designed that way.

---

## Dependency Guidelines

* Avoid adding dependencies unless they provide clear value.
* Prefer existing workspace dependencies where suitable.
* Check license compatibility before adding dependencies.
* Avoid large, unmaintained, deprecated, or security-sensitive dependencies.
* Keep dependency changes separate from unrelated feature work when practical.
* Do not introduce runtime dependencies for code that can reasonably stay dev-only.

---

## Security and Supply Chain

* Treat all external input as untrusted.
* Validate paths, URLs, serialized data, model metadata, and FFI inputs.
* Avoid command injection risks in scripts and tooling.
* Do not log secrets, tokens, credentials, or private paths.
* Do not commit generated secrets, credentials, private keys, or local environment files.
* Prefer safe defaults for file permissions, network behavior, and execution.
* Use constant-time comparisons where security-sensitive tokens or signatures are involved.

---

## Monorepo Organization

The repository is organized by responsibility:

* `crates/`: Core Rust implementation.
* `bindings/`: FFI and language bindings such as Node, Python, and WASM.
* `apps/`: First-party applications such as the Rust CLI.
* `demos/`: Browser demos using the public browser package.
* `tools/`: Developer tools such as the browser playground.
* `examples/`: Runnable onboarding examples for public package surfaces.
* `lib/`: Public facade and language/runtime package source, including Rust, Python, Node, and browser packages.
* `docs/`: User-facing and contributor-facing documentation.
* `xtask/`: Developer automation and repository maintenance tooling.
* `.agents/skills/`: Agent skills and repository-specific agent guidance.

### Boundaries

* Core engine behavior belongs in `crates/`, not duplicated in bindings, demos, or apps.
* Bindings should be thin layers over stable core APIs.
* Public library packages should expose polished public APIs, not internal engine details.
* Demos and apps may compose packages and crates but should not become hidden libraries.
* Shared test fixtures should be placed where all relevant languages can consume them without circular dependencies.

---

## Pull Request Expectations

A good pull request should:

* Have a focused purpose.
* Include tests or explain why tests are not needed.
* Update docs when user-facing behavior changes.
* Avoid unrelated refactors.
* Preserve public compatibility unless the change is explicitly breaking.
* Pass relevant formatting, linting, typing, and tests.
* Include clear notes for migrations, breaking changes, or operational impacts.

---

## Agent-Specific Instructions

When an agent modifies code in this repository:

1. Inspect the diff before finishing.
2. Read this file before enforcing style.
3. Fix style violations directly when safe.
4. Prefer minimal local edits over broad rewrites.
5. Run the narrowest relevant validation command available.
6. Report what changed and what validation was run.
7. If validation could not be run, say why.

Agents must not invent new conventions when this document or nearby code already establishes one.
