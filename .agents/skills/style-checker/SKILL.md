---

name: style-checker
description: Enforces this monorepo's coding style rules by inspecting git diffs, reading .agents/skills/style-checker/references/style_guidance.md, fixing style violations, and reporting the result. Use when the user asks to check style, review codebase style, clean up a diff, verify coding conventions, or before completing any coding task in this repository.
compatibility: Designed for coding agents working inside a git monorepo with shell access, file read access, and file editing tools.
allowed-tools: Bash(git:*) Bash(cargo:*) Bash(npm:*) Bash(pnpm:*) Bash(yarn:*) Read Edit MultiEdit
---

# Style Checker

You are an expert code reviewer enforcing this monorepo's local style rules.

Run this skill when asked to check style, review coding conventions, clean up a diff, or before finishing any coding task in this repository.

## Core Rule

Always review the changed code against the repository's canonical style guide:

```text
.agents/skills/style-checker/references/style_guidance.md
```

That file is the source of truth. If these instructions conflict with `.agents/skills/style-checker/references/style_guidance.md`, follow `.agents/skills/style-checker/references/style_guidance.md`.

## Workflow

### 1. Inspect the current changes

Determine what changed before reviewing style.

Run the narrowest useful git commands, typically:

```bash
git status --short
git diff
git diff --cached
```

Review only changed files and changed lines unless the style guide requires broader context.

Do not perform unrelated cleanup.

### 2. Read the style guide

Read:

```text
.agents/skills/style-checker/references/style_guidance.md
```

Use it to evaluate the diff. Do not rely only on general language or framework conventions.

### 3. Analyze the diff

Compare the changed code against `.agents/skills/style-checker/references/style_guidance.md`.

Pay special attention to these monorepo expectations.

#### Rust

Check changed Rust code for:

* Custom library error types, preferably using `thiserror`, where typed errors are appropriate.
* No naked `unwrap()` or `expect()` in production paths.
* Intentional error handling or propagation for fallible operations.
* Appropriate `//!` or `///` documentation for public modules, public types, and important public functions.
* Consistency with nearby crate conventions for errors, logging, async behavior, and public APIs.

#### TypeScript

Check changed TypeScript code for:

* NodeNext-compatible relative imports with explicit `.js` extensions where required.
* `readonly` arrays, tuples, and object shapes where immutability is intended or required by project convention.
* No newly introduced implicit `any`.
* Explicit types at module boundaries and public APIs.
* No weakening of strictness, type safety, or existing lint expectations.

### 4. Fix violations directly

If you find a style violation, edit the affected file immediately.

Use the smallest safe fix. Prefer targeted edits over broad rewrites.

Good fixes include:

* Replacing `unwrap()` with proper error propagation.
* Adding or refining a typed error enum.
* Adding missing `.js` extensions to TypeScript relative imports.
* Adding `readonly` where required.
* Adding public documentation required by the style guide.
* Tightening an implicit or overly broad type.

Avoid:

* Reformatting unrelated code.
* Renaming symbols for preference only.
* Refactoring outside the changed area.
* Applying generic best practices that are not supported by the repo guidance.

### 5. Re-check the diff

After editing, inspect the diff again.

Confirm that:

* The style issue is fixed.
* The edit is minimal.
* No unrelated changes were introduced.
* The code still follows nearby patterns.

### 6. Run relevant validation

When practical, run the narrowest relevant validation command for the changed files.

Prefer project-specific commands from package scripts, justfiles, makefiles, cargo configs, or repo documentation.

Possible commands include:

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
yarn typecheck
yarn lint
yarn test
```

Do not run expensive full-repo checks when a narrower check is available.

If validation cannot be run, explain why.

## Reporting

At the end, report clearly and briefly.

Include:

* Whether the style check passed.
* What style issues were found.
* What files were changed.
* What validation commands were run.
* Any validation that could not be run.

If no issues were found, say exactly:

```text
Style check passed; no changes were needed.
```

If fixes were made, summarize them in plain language.

Example:

```text
Style check completed. I fixed two TypeScript style issues in packages/api/src/client.ts: added NodeNext .js import extensions and made the exported config map readonly. I ran pnpm typecheck and it passed.
```

## Constraints

* Always read `.agents/skills/style-checker/references/style_guidance.md` before enforcing style.
* Review the diff, not the whole repository.
* Fix style violations directly when safe.
* Keep changes minimal and local to the relevant diff.
* Do not invent new conventions.
* Do not ignore the style check before completing coding work.
* If `.agents/skills/style-checker/references/style_guidance.md` is missing, report that the canonical style guide could not be found and apply only conservative checks based on nearby code.
