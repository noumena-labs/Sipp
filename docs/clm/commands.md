# Commands

`clm` groups source checkout automation into focused command families. Use
`clm <group> --help` for generated help and the current option list.

## Health Checks

```bash
clm doctor
clm doctor --target wasm
clm doctor --target node --backend vulkan
clm toolchain status
```

`doctor` checks local readiness without installing or deleting anything.
`toolchain status` reports xtask-managed tools such as Bun, Python, uv,
Emscripten, and Ninja. CUDA is externally installed; xtask reports it but does
not install or delete it.

## Build

```bash
clm build core
clm build wasm
clm build node --backend cpu
clm build python --backend vulkan
clm build cli --backend all
clm build gateway-server --backend cpu
clm build all
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
clm run examples serve browser --port 5173
clm run examples serve gateway-local --model .build/models/model.gguf --bind 127.0.0.1:8787
clm run examples gateway rust --case query
clm run demos serve chat
clm run tools serve playground
clm run gateway-server check --config apps/gateway-server/config/local.toml
clm run gateway-server serve --config apps/gateway-server/config/local.toml --backend cpu
```

`run` commands are for long-lived demos, gateway processes, example servers,
and non-test diagnostics. Test execution lives under `clm test`.

## Docs

```bash
clm docs build
clm docs serve
clm docs build --lang zh
```

`docs build` installs `mdbook` and `mdbook-mermaid` when missing, extracts the
bundled Mermaid JavaScript assets into `theme/`, and writes the generated book
to `book/`.

## Test

```bash
clm test list
clm test list --group unit --layer interface --cases --search router --format json
clm test unit group full
clm test unit suite rust-crates --package cogentlm-engine
clm test unit suite node-package --backend cpu
clm test unit suite browser-package
clm test smoke suite example-node --backend cpu
clm test smoke group local-model --backend cpu
clm test verify --changed
clm test verify --target public-docs
```

Model-backed smoke tests use the setup sample model cache under `.build/models`
when `--model` is omitted. See [Testing](../testing.md) for the full suite
catalog.

## Clean

```bash
clm clean --dry-run
clm clean
clm clean --purge
clm clean --toolchains
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
