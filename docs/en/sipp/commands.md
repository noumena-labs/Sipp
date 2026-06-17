# Commands

`sipp` groups source checkout automation into focused command families. Use
`sipp <group> --help` for generated help and the current option list.

## Health Checks

```bash
sipp doctor
sipp doctor --target wasm
sipp doctor --target node --backend vulkan
sipp toolchain status
```

`doctor` checks local readiness without installing or deleting anything.
`toolchain status` reports xtask-managed tools such as Bun, Python, uv,
Emscripten, and Ninja. CUDA is externally installed; xtask reports it but does
not install or delete it.

## Build

```bash
sipp build core
sipp build wasm
sipp build node --backend cpu
sipp build python --backend vulkan
sipp build cli --backend all
sipp build gateway-server --backend cpu
sipp build all
```

`build all` builds the main target families with default CPU native outputs. It
does not build every backend variant for every package.

Backend values:

- `cpu`: portable default.
- `cuda`: NVIDIA CUDA backend; requires a local CUDA Toolkit.
- `metal`: Apple Metal backend on macOS.
- `vulkan`: Vulkan backend; xtask can bootstrap the Vulkan SDK when needed.
- `all`: host-supported backend set for the selected target.

## Run

```bash
sipp run examples serve browser --port 5173
sipp run examples serve gateway-local --model .build/models/model.gguf --bind 127.0.0.1:8787
sipp run examples gateway rust --case query
sipp run demos serve chat
sipp run tools serve playground
sipp run gateway-server check --config apps/gateway-server/config/local.toml
sipp run gateway-server serve --config apps/gateway-server/config/local.toml --backend cpu
```

`run` commands are for long-lived demos, gateway processes, example servers,
and non-test diagnostics. Test execution lives under `sipp test`.

## Docs

```bash
sipp docs build
sipp docs serve
sipp docs build --lang zh
```

`docs build` installs `mdbook` and `mdbook-mermaid` when missing, extracts the
bundled Mermaid JavaScript assets into `theme/`, and writes the generated book
to `book/`.

## Test

```bash
sipp test list
sipp test list --group unit --layer interface --cases --search router --format json
sipp test unit group full
sipp test unit suite rust-crates --package sipp-rs
sipp test unit suite node-package --backend cpu
sipp test unit suite browser --wasm-threading single-thread
sipp test smoke suite example-node --backend cpu
sipp test smoke group local-model --backend cpu
sipp test verify --changed
sipp test verify --target public-docs
```

Model-backed smoke tests use the setup sample model cache under `.build/models`
when `--model` is omitted. See [Testing](../testing.md) for the full suite
catalog.

## Clean

```bash
sipp clean --dry-run
sipp clean
sipp clean --purge
sipp clean --toolchains
```

`clean` removes generated build outputs while preserving downloaded toolchains
and dependency installs. `--purge` also removes workspace `node_modules`
directories. `--toolchains` removes xtask-managed toolchains under
`.build/toolchain`.

## Output Flags

Most command groups accept the shared output flags:

- `--verbose`: stream subprocess output directly.
- `--no-banner`: disable decorative banners.
- `--plain`: disable bounded inline rendering.
