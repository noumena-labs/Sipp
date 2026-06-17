# Backend Matrix

Sipp local inference is built on llama.cpp and ggml. Sipp owns the
client APIs, endpoint model, scheduling, package bindings, browser lifecycle,
and gateway integration; llama.cpp and ggml provide the GGUF runtime and
backend kernels.

Backend support therefore has two layers:

- Sipp support: which backend names each package can select and how the
  backend is built or chosen.
- ggml support: which tensor operations each ggml backend implements.

For the ggml operation-level matrix, use the upstream
[llama.cpp GGML operations table](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md).
That table is generated from llama.cpp backend probes and is the source of
truth for per-operation support.

## Sipp Backend Names

| Backend | Device class | Where Sipp exposes it | Notes |
| --- | --- | --- | --- |
| `cpu` | Host CPU | Browser, Node.js, Python, Rust/source, CLI, gateway server | Portable default. Native builds use ggml CPU; browser builds use WASM CPU with the browser runtime. |
| `webgpu` | Browser GPU through WebGPU | Browser package | Browser-only. Selected with browser local endpoint `options.backend`; requires a WebGPU-capable browser and adapter. |
| `cuda` | NVIDIA GPU | Native source builds, Node.js, Python, CLI, gateway server | Requires a local CUDA Toolkit and compatible NVIDIA driver. xtask reports CUDA readiness but does not install CUDA. |
| `metal` | Apple GPU through Metal | Native source builds, Node.js, Python, CLI, gateway server on macOS | macOS-only native backend. Best for Apple Silicon and validated AMD Macs; use CPU on Intel integrated GPUs. |
| `vulkan` | GPU through Vulkan | Native source builds, Node.js, Python, CLI, gateway server | Requires a Vulkan-capable system and driver. xtask can bootstrap the Vulkan SDK for builds. |

Upstream llama.cpp/ggml supports more backend families than Sipp currently
exposes as package/runtime selectors, including BLAS, CANN, OpenCL, SYCL,
ZenDNN, and zDNN. Those appear in the upstream operation matrix but are not
first-party Sipp backend names at this time.

## Package And Runtime Selection

| Surface | Supported backend selectors | How to select |
| --- | --- | --- |
| Browser local | `auto`, `cpu`, `webgpu` | `client.add(..., { kind: 'local', options: { backend: 'webgpu' } })` |
| Node.js local | `cpu`, `vulkan`, `cuda`, `metal` | `SIPP_NODE_BACKEND=cpu|vulkan|cuda|metal` |
| Python local | `cpu`, `vulkan`, `cuda`, `metal` | `SIPP_PYTHON_BACKEND=cpu|vulkan|cuda|metal` |
| CLI | `auto`, `cpu`, `cuda`, `metal`, `vulkan` | `sipp ... --backend <backend>` |
| Gateway server | `auto`, `cpu`, `cuda`, `metal`, `vulkan` | Build or run with `sipp ... --backend <backend>`; target TOML can set `backend = "auto"` or a concrete backend. |
| Rust source/client workflows | Compiled native backend set | Build through `sipp` or `cargo xtask`; runtime availability follows the linked native artifacts. |

`auto` is a runtime selection policy. `all` is a build/test selector used by
`sipp` and `cargo xtask`; it builds or checks the host-supported backend set for
that target and is not a runtime backend name.

## Mixing Backends

Keep build artifact selection separate from engine backend selection.

- A build artifact decides which ggml GPU backends are compiled and loadable in
  the current process. A CUDA-only artifact does not make Vulkan available, and
  a Metal-only artifact does not make CUDA or Vulkan available.
- `cpu` is the exception in the engine policy. When an engine is explicitly
  planned for `cpu`, Sipp disables GPU layers, device placement, GPU K/V
  offload, op offload, flash attention, and GPU residency leasing for that
  load.
- Explicit GPU selections such as `cuda`, `metal`, `vulkan`, and `webgpu` must
  be both compiled into the active artifact and available on the host.
- Node.js and Python choose the native binding at process load with
  `SIPP_NODE_BACKEND` or `SIPP_PYTHON_BACKEND`. Their local model
  descriptors do not carry a separate per-engine backend field, so use a
  different process or artifact when you need a different GPU backend.
- Gateway, CLI, browser, and lower-level Rust lifecycle paths expose backend
  selectors at the target/load/run layer. They can select only from the backend
  set available to that artifact and host.

Practical examples:

| Active artifact/process | CPU engine | CUDA engine | Metal engine | Vulkan engine |
| --- | --- | --- | --- | --- |
| CUDA-only native artifact | Yes, where the surface exposes CPU selection | Yes, if the CUDA device is available | No | No |
| Metal-only native artifact | Yes, where the surface exposes CPU selection | No | Yes, on macOS | No |
| Vulkan-only native artifact | Yes, where the surface exposes CPU selection | No | No | Yes, if the Vulkan device is available |
| Multi-backend source build | Yes | Yes, if compiled and available | Yes, if compiled and available | Yes, if compiled and available |

CLI examples:

```bash
# Build a CUDA-capable CLI artifact.
sipp build cli --backend cuda

# Use CUDA when the CUDA device is available.
sipp ./models/model.gguf "Explain this model." --chat --backend cuda

# Force CPU for a run; this disables GPU offload for that engine.
sipp ./models/model.gguf "Explain this model." --chat --backend cpu

# This requires a Vulkan-capable artifact; a CUDA-only artifact is not enough.
sipp ./models/model.gguf "Explain this model." --chat --backend vulkan
```

Gateway target examples:

```toml
# Same gateway process, different local targets.
# Each GPU backend must be compiled into the active gateway artifact.
[[targets]]
name = "local-cuda"
type = "local"
model = "./models/model.gguf"
backend = "cuda"

[[targets]]
name = "local-cpu"
type = "local"
model = "./models/model.gguf"
backend = "cpu"
```

Browser examples:

```ts
// Browser local supports CPU and WebGPU backend selection per local endpoint.
await client.add('local-webgpu', {
  kind: 'local',
  model: './models/model.gguf',
  options: { backend: 'webgpu' },
});

await client.add('local-cpu', {
  kind: 'local',
  model: './models/model.gguf',
  options: { backend: 'cpu' },
});
```

Node.js and Python examples:

```powershell
# PowerShell: choose the native binding before starting the process.
$env:SIPP_NODE_BACKEND = "cuda"
node .\examples\node\chat.mjs .\models\model.gguf "Explain this model."

$env:SIPP_NODE_BACKEND = "cpu"
node .\examples\node\chat.mjs .\models\model.gguf "Explain this model."
```

```bash
# Bash: choose the native binding before starting the process.
SIPP_PYTHON_BACKEND=cuda \
  python examples/python/chat.py ./models/model.gguf "Explain this model."

SIPP_PYTHON_BACKEND=cpu \
  python examples/python/chat.py ./models/model.gguf "Explain this model."
```

## Build Matrix

| Build command | Backend argument | Result |
| --- | --- | --- |
| `sipp build wasm` | none | Browser WASM package with CPU and WebGPU runtime support. |
| `sipp build node --backend cpu` | `cpu`, `cuda`, `metal`, `vulkan`, `all` | Node native binding artifacts for the selected backend set. |
| `sipp build python --backend cpu` | `cpu`, `cuda`, `metal`, `vulkan`, `all` | Python native binding artifacts for the selected backend set. |
| `sipp build cli --backend cpu` | `cpu`, `cuda`, `metal`, `vulkan`, `all` | Local `sipp` CLI distribution for the selected backend set. |
| `sipp build gateway-server --backend cpu` | `cpu`, `cuda`, `metal`, `vulkan`, `all` | Gateway server distribution for the selected backend set. |
| `sipp build all` | none | Core, WASM, Python CPU, Node CPU, and CLI CPU targets. |

`sipp build all` is intentionally conservative. Use an explicit backend build
when you need CUDA, Metal, or Vulkan artifacts.

## Operation Support

ggml backends do not all implement the same operation set. Common transformer
inference paths are covered by the backends Sipp exposes, but support for a
specific model family depends on the ggml operations used by that model and the
selected backend.

Use these rules when diagnosing backend issues:

- If a model works on `cpu` but fails on a GPU backend, check the upstream
  ggml operations matrix for the missing operation.
- If a GPU backend lacks an operation, llama.cpp/ggml may fall back for some
  paths, keep tensors on CPU for that operation, or fail depending on the graph
  and backend policy.
- If a package cannot see a backend at runtime, check that the artifact was
  built or installed for that backend and that the device driver/runtime is
  visible to the process.
- Browser `webgpu` depends on both compiled WebGPU support and browser adapter
  availability. Use `backend: 'cpu'` to force the browser CPU path.

For local verification from a source checkout:

```bash
sipp doctor --target node --backend vulkan
sipp run llama backend-ops --backend vulkan --mode support
sipp run llama backend-ops --backend cuda --mode perf --op MUL_MAT
```

The `llama backend-ops` command builds llama.cpp's backend operation tool for
the selected backend and is useful when investigating operation coverage or
performance outside the Sipp client path.

## Practical Selection

Use `cpu` first when validating a model or reproducing correctness issues. Move
to a GPU backend after the model, prompt format, and runtime config are known to
work.

Use `webgpu` for browser-local acceleration when the application can require a
modern WebGPU browser. Keep a CPU fallback for browsers, drivers, and devices
that do not expose a compatible adapter.

Use `cuda` for NVIDIA-heavy native deployments and `metal` for Apple Silicon or
validated AMD macOS deployments. On Intel Macs with integrated GPUs, use `cpu`
unless you have measured a stable Metal path for the target model. Use `vulkan`
when you want a cross-vendor native GPU path and have validated the target
driver stack.
