# sipp CLI

`sipp` is the repo-local launcher for Sipp source checkout workflows. It
forwards to `cargo xtask` after setup has installed wrapper scripts under
`.build/bin`.

Use `sipp` when you are working from the repository and need to build native
artifacts, run demos, start the gateway server, manage xtask toolchains, or run
cataloged tests, or build the documentation book. Published packages such as
`sipp`, `sipp-server`, and the Python wheel do not require `sipp`.

## Command Shape

Every `sipp` command has the same arguments as `cargo xtask`:

```bash
sipp doctor
sipp build node --backend cpu
sipp run examples serve browser
sipp test list
sipp docs build
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
