# Models And Backends

CogentLM local inference uses GGUF model files. Text workflows need a text GGUF
model, embedding workflows need a model that reports embedding support, and
vision chat workflows need both a model GGUF and a projector GGUF.

## Model Sources

The example and smoke workflows can use a cached sample model under
`.build/models`. For manual local inference, pass an explicit model path:

```bash
cargo run -p cogentlm-rust-examples --bin query -- <model.gguf> "Hello"
```

Browser demos can also load model URLs, then cache data in browser storage when
the runtime path supports it.

## Native Backends

Backend names are shared across build and runtime selection:

- `cpu`: portable default backend.
- `vulkan`: GPU backend for Vulkan-capable systems.
- `cuda`: NVIDIA CUDA backend.
- `metal`: Apple Metal backend on macOS.

Build native artifacts with xtask:

```bash
cargo xtask build node --backend cpu
cargo xtask build python --backend vulkan
cargo xtask build cli --backend all
```

Runtime selection is package-specific:

- Node.js: `COGENTLM_NODE_BACKEND=cpu|vulkan|cuda|metal`
- Python: `COGENTLM_PYTHON_BACKEND=cpu|vulkan|cuda|metal`
- CLI: `--backend auto|cpu|cuda|metal|vulkan`

Leave runtime backend variables unset for automatic selection.
