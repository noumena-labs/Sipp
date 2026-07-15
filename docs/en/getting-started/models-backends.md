# Models And Backends

Sipp local inference uses GGUF model files. Text workflows need a text GGUF
model, embedding workflows need a model that reports embedding support, and
vision chat workflows need both a model GGUF and a projector GGUF.

## Model Sources

Every package uses an explicit installed, local, or remote source. Native
packages also require the lifecycle asset-store root:

- Browser: `source: { kind: 'remote', modelUrls: ['https://models.example/model.gguf'] }`
- Node.js: `source: { kind: 'local', modelPaths: ['/path/model.gguf'] }, storageRoot: '.sipp-models'`
- Python: `LocalModelDescriptor(ModelSource.local([model]), '.sipp-models', config)`
- Rust: `LocalModelDescriptor { source: ModelSource::Local { .. }, storage_root, config }`

Remote acquisition uses one Rust policy in browser and native runtimes: exact
validator matching, bounded retries for transient HTTP failures, no stale-cache
fallback, and cleanup of assets created by a failed or cancelled acquisition.

Source examples and smoke workflows can use a cached sample model under
`.build/models`; see [Source Builds](../maintainers/source-builds.md).

## Native Backends

Backend names are shared across build and runtime selection:

- `cpu`: portable default backend.
- `vulkan`: GPU backend for Vulkan-capable systems.
- `cuda`: NVIDIA CUDA backend.
- `metal`: Apple Metal backend on macOS.

Runtime selection is package-specific:

- Node.js: `SIPP_NODE_BACKEND=cpu|vulkan|cuda|metal`
- Python: `SIPP_PYTHON_BACKEND=cpu|vulkan|cuda|metal`
- CLI: `--backend auto|cpu|cuda|metal|vulkan`

Leave runtime backend variables unset for automatic selection.

Maintainer builds can produce backend-specific artifacts with `sipp` or
`cargo xtask`; see [Source Builds](../maintainers/source-builds.md).

For the full package/backend matrix and llama.cpp/ggml operation support
guidance, see [Backend Matrix](../guides/backend-matrix.md).
