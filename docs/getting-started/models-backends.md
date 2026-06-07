# Models And Backends

CogentLM local inference uses GGUF model files. Text workflows need a text GGUF
model, embedding workflows need a model that reports embedding support, and
vision chat workflows need both a model GGUF and a projector GGUF.

## Model Sources

For local package usage, pass an explicit GGUF model path in Node.js, Python,
or Rust, or serve a GGUF model URL to browser code:

- Browser: `source: '/models/model.gguf'`
- Node.js: `modelPath: '/path/to/model.gguf'`
- Python: `LocalModelDescriptor('/path/to/model.gguf')`
- Rust: `EndpointDescriptor::local(model_path, config)`

Source examples and smoke workflows can use a cached sample model under
`.build/models`; see [Source Builds](../maintainers/source-builds.md).

## Native Backends

Backend names are shared across build and runtime selection:

- `cpu`: portable default backend.
- `vulkan`: GPU backend for Vulkan-capable systems.
- `cuda`: NVIDIA CUDA backend.
- `metal`: Apple Metal backend on macOS.

Runtime selection is package-specific:

- Node.js: `COGENTLM_NODE_BACKEND=cpu|vulkan|cuda|metal`
- Python: `COGENTLM_PYTHON_BACKEND=cpu|vulkan|cuda|metal`
- CLI: `--backend auto|cpu|cuda|metal|vulkan`

Leave runtime backend variables unset for automatic selection.

Maintainer builds can produce backend-specific artifacts with `clm` or
`cargo xtask`; see [Source Builds](../maintainers/source-builds.md).
