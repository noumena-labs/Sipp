# Contributing

CogentLM is a polyglot monorepo. Keep contributions focused, documented, and
validated with the narrowest useful commands.

Before submitting issues or PRs, be ready to explain why the change matters and
how it works. AI-assisted coding is fine, including agent-generated drafts, but
the author is responsible for reviewing, understanding, and maintaining the
final change.

## Identify **The Why**

For issues and feature requests, explain the problem, who it affects, and how
it could affect the system. This helps maintainers evaluate the priority and
choose the right implementation path.

## Explain **The How**

For PRs, describe what changed and how the implementation works. If you cannot
explain the behavior, risks, and validation, revisit the change before asking
for review.

## Communication

Use your own words in issues and PRs. Keep the main message concise, then add
supporting detail only when it helps reviewers understand the change.

For each issue or PR:

- Explain why it matters.
- Describe what changed.
- Keep the scope atomic.
- Avoid unrelated cleanup.

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
cargo xtask test verify --target public-docs
```

For code changes, use the narrowest relevant test target from
[Testing](testing.md). Run broader suites only when the change crosses package
or runtime boundaries.
