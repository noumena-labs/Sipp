# Installation

Install the published package for the runtime your application uses. All
public client packages use the same endpoint model: register an endpoint, keep
the returned endpoint reference, and choose that endpoint for `query`, `chat`,
or `embed`.

## Package Installs

| Surface | Install | Use for |
| --- | --- | --- |
| Browser | `npm install cogentlm` | Browser-local GGUF inference and browser gateway clients. |
| Node.js | `npm install cogentlm-server` | Server-side local inference and framework route handlers. |
| Python | `pip install cogentlm` | Python scripts, services, and gateway clients. |
| Rust | `cargo add cogentlm` | Rust applications and services. |

The current release workflow publishes browser npm, Node npm, Python wheel,
and Rust source artifacts. It does not yet publish a standalone gateway-server
binary, container image, or `cargo install` target. Use the source checkout and
Dockerfile when deploying the gateway server until a public server artifact is
added.

## Runtime Requirements

- Local inference needs a compatible GGUF model file or browser-served GGUF
  asset.
- Browser-local inference needs a modern browser with WebAssembly support;
  WebGPU acceleration depends on the browser and device. For details, please refer to [Gateway](../reference/device-support.md).
- Node and Python native packages select a backend from packaged native
  artifacts. Use `COGENTLM_NODE_BACKEND` or `COGENTLM_PYTHON_BACKEND` when you
  need to force `cpu`, `vulkan`, `cuda`, or `metal`.
- Gateway clients need only the gateway base URL, public target name, and
  application-owned authentication value.

## Next Steps

- [clm CLI for source checkouts](../clm/README.md)
- [Browser package](../packages/browser.md)
- [Node.js package](../packages/node.md)
- [Python package](../packages/python.md)
- [Rust package](../packages/rust.md)
- [Gateway](../gateway/README.md)
- [Maintainer source builds](../maintainers/source-builds.md)
