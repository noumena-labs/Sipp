# Models And Backends

Sipp local inference uses GGUF model files. Text workflows need a text GGUF
model, embedding workflows need a model that reports embedding support, and
vision chat workflows need both a model GGUF and a projector GGUF.

## Model Store

Every client owns one model store. `models.add` returns a `ManagedModel`; pass
its `id` to a local endpoint descriptor. Native local paths are referenced in
place, remote sources are stored under the client storage root, and browser
sources are persisted in OPFS. Endpoint descriptors select models and contain
only load-time options.

Native clients store models under `.sipp-models` by default. Browser clients
use the `sipp-models` directory in OPFS. Override the root only when an
application needs a separate store:

- Browser: `new SippClient({ storageRoot: 'tenant/models' })` selects an OPFS directory.
- Node.js: `new SippClient({ storageRoot: '/custom/models' })`.
- Python: `SippClient(storage_root='/custom/models')`.
- Rust: `SippClient::with_storage_root("/custom/models")?`.

Remove a local endpoint with `client.remove(endpointId)` before removing its
model with `client.models.remove(modelId)`. A model used by an endpoint cannot
be removed.

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
