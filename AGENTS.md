# CogentLM Agent Instructions

Welcome! This is the primary context and guidance entry point for AI coding agents working in the CogentLM repository.

## 1. Quick Navigation & Context

To avoid token bloat, do not read the entire codebase at once. Instead, refer to our specialized context files in the [.agents/](file:///.agents) directory:
- **General Architecture:** Read [.agents/system/architecture.md](file:///.agents/system/architecture.md) to understand the crate boundaries.
- **Build Instructions:** Read [.agents/build/instructions.md](file:///.agents/build/instructions.md) before executing build commands.
- **Troubleshooting & SOPs:** Check the [.agents/SOPs/](file:///.agents/SOPs) directory for specific step-by-step procedures.

---

## 2. Workspace Build & Run Commands

Always use the **`build-orchestrator`** skill when compiling. The repository uses `xtask` to manage C++ dependencies and environment variables.

- **Build Core (Rust only):** `cargo xtask build core`
- **Build Node Bindings:** `cargo xtask build node` (use `--backend vulkan` for GPU accelerated builds)
- **Build Python Bindings:** `cargo xtask build python` (optionally `--backend vulkan`)
- **Build WebAssembly/WebGPU:** `cargo xtask build wasm`
- **Build All Targets:** `cargo xtask build all`
- **Serve An App:** `cargo xtask run apps serve examples`
- **Run llama.cpp Backend Ops:** `cargo xtask run llama backend-ops --backend cpu`

---

## 3. Test & Lint Commands

Always use the **`test-runner`** skill when verifying changes.
- **List Tests:** `cargo xtask test list` (see [docs/testing.md](file:///docs/testing.md) for profile contents)
- **Public Contributor Gate:** `cargo xtask test all --profile contributor` (`layout`, `xtask`)
- **Quick Local Gate:** `cargo xtask test all --profile quick` (`contributor`, `rust-crates`)
- **xtask Tests:** `cargo xtask test whitebox --suite xtask`
- **White-box Tests:** `cargo xtask test whitebox --suite rust-crates --package <crate_name>`
- **Interface Tests:** `cargo xtask test interface --suite node-package --backend cpu`
- **Coverage:** `cargo xtask test coverage --scope whitebox --backend cpu`
- **Rust Tests:** `cargo test` (or `cargo test -p <crate_name>` for narrow Rust-only checks)
- **Rust Linting/Formatting:** `cargo clippy` and `cargo fmt`
- **TypeScript Typecheck:** `pnpm typecheck` or `bun run typecheck`
- **TypeScript Linting:** `pnpm lint` or `bun run lint`

---

## 4. Pre-Task Check (Style Checker)

Before completing any task, you **MUST** run the **`style-checker`** skill:
- Check git status (`git status --short`) and diffs (`git diff`).
- Inspect [.skills/style-checker/references/style_guidance.md](file:///.skills/style-checker/references/style_guidance.md).
- Apply minimal local fixes directly to any code violating the guidelines.
- Run the narrowest relevant test/validation command to ensure correctness.
