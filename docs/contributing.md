# Contributing

CogentLM is a polyglot monorepo. Keep contributions focused, documented, and
validated with the narrowest useful commands.

## Before Editing

- Read the root README and the relevant package or app README.
- Use `cargo xtask test list` to inspect available validation targets.
- Use `cargo xtask` commands for builds and long-running workflows.
- Avoid changing vendored files under `third_party/` unless the task is
  explicitly about the vendor source.

## Documentation Changes

- Keep README files short and task-oriented.
- Put detailed guides and references in this mdBook.
- Prefer examples that can be copied and run from a clean checkout.
- Update docs when public APIs, package behavior, commands, or configuration
  change.

## Validation

For documentation-only changes:

```bash
mdbook build
cargo xtask test list
```

For code changes, use the narrowest relevant test target from
[Testing](testing.md). Run broader suites only when the change crosses package
or runtime boundaries.
