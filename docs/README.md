# CogentLM Documentation

CogentLM packages local and gateway-backed inference runtimes for browser,
Node.js, Python, and Rust applications. The project is organized around one
client model: register local and remote endpoints with `CogentClient.add`, keep
the returned endpoint reference, and choose that reference for `query`, `chat`,
or `embed`.

This book starts with the published packages that application developers use.
Source checkout, build orchestration, repository architecture, and contribution
workflow live in the maintainer section.

> [!WARNING]
> CogentLM is under active development. Changes will be made frequently.
> If you find any issues, bugs, or need any features, please raise them in the github or Discord server ([Discord](https://discord.gg/abzgfghhrq)).

## Start Here

- [Installation](getting-started/installation.md) lists the published package
  install commands.
- [Quickstarts](getting-started/quickstarts.md) shows short Browser, Node.js,
  Python, Rust, and gateway paths.
- [Using the Core Library](packages/) describes the public package
  surfaces in depth.
- [Gateway](gateway/) explains the first-party server, Docker workflows,
  configuration, testing, operations, toolkit, and architecture.
- [Frameworks](packages/frameworks/) covers Next.js, TanStack, and
  React/Vite integration patterns.
- [Gateway And Hybrid Inference](guides/gateway-hybrid.md) explains when to use
  local endpoints, gateway endpoints, and provider endpoints.
- [Maintainers](maintainers/) covers source builds, tests, repo
  structure, and contribution workflow.

## Build The Book Locally

Use `clm docs` from a source checkout:

```bash
clm docs build
clm docs serve
```

`clm docs build` installs `mdbook` and `mdbook-mermaid` when missing, extracts
the bundled Mermaid JavaScript assets, and writes the generated book to
`book/`; If the `clm` launcher is not active, use `cargo xtask docs ...`
with the same arguments.
