# Contributing

CogentLM is a polyglot monorepo. Keep contributions focused, documented, and
validated with the narrowest useful commands.

Before submitting issues and PRs, you must understand **the WHY** and **the HOW**. AI assisted coding is fine, even when your code is mostly written by coding agents. However, manual review is required to avoid AI-generated slop over time.

## Identify **the WHY**

Whether it is an issue or a feature, you should first think why it should be handled and how it could impact the system in the first place. It helps you to think deeper into the problem, make the right decision, and write a better prompt to describe the issue and expected system behavior.

## Explain **the HOW**

You must understand your code and should be able to explain what you code (or ask the agent to code). If you can't, please revisit **the WHY** or your issues and PRs will be closed. 

## Communication

Please avoid using agents in the issues and PRs for communication. Use your own words and tones for human interactions. 

For each issue and PR, 
- keep it short and concise for the main message. 
- Add additional information if necessary to help reviewers to understand the changes.
- Explain why it matters and what you change.
- Be atomic, and only change what should be changed. 

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
