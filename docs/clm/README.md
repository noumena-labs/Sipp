# clm CLI

`clm` is the repo-local launcher for CogentLM source checkout workflows. It
forwards to `cargo xtask` after setup has installed wrapper scripts under
`.build/bin`.

Use `clm` when you are working from the repository and need to build native
artifacts, run demos, start the gateway server, manage xtask toolchains, or run
cataloged tests. Published packages such as `cogentlm`, `cogentlm-server`, and
the Python wheel do not require `clm`.

## Command Shape

Every `clm` command has the same arguments as `cargo xtask`:

```bash
clm doctor
clm build node --backend cpu
clm run examples serve browser
clm test list
```

If the launcher is not active in the current shell, use the same command after
`cargo xtask`:

```bash
cargo xtask doctor
cargo xtask build node --backend cpu
```

## Pages

- [Setup](setup.md)
- [Commands](commands.md)
- [Troubleshooting](troubleshooting.md)

