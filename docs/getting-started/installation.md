# Installation

Install the published package for the runtime your application uses. All
public client packages use the same endpoint model: register an endpoint, keep
the returned endpoint reference, and choose that endpoint for `query`, `chat`,
or `embed`.

## Package Installs

| Surface | Install | Use for |
| --- | --- | --- |
| Browser | `npm install @sipp/sipp` | Browser-local GGUF inference and browser gateway clients. |
| Node.js | `npm install @sipp/sipp-server` | Server-side local inference and framework route handlers. |
| Python | `pip install sipp-py` | Python scripts, services, and gateway clients. |
| Python CUDA | `pip install "sipp-py[cuda]"` | Python local inference with CUDA backend wheels. |
| Python Vulkan | `pip install "sipp-py[vulkan]"` | Python local inference with Vulkan backend wheels. |
| Python Metal | `pip install "sipp-py[metal]"` | Python local inference with Metal backend wheels on macOS. |
| Rust | `cargo add sipp-rs` | Rust applications and services. |

The current release workflow publishes browser npm, Node npm, Python wheels,
and Rust crates. It does not yet publish a standalone gateway-server
binary, container image, or `cargo install` target. Use the source checkout and
Dockerfile when deploying the gateway server until a public server artifact is
added.

## Runtime Requirements

- Local inference needs a compatible GGUF model file or browser-served GGUF
  asset.
- Browser-local inference needs a modern browser with WebAssembly support;
  WebGPU acceleration depends on the browser and device. For details, please refer to [Gateway](../reference/device-support.md).
- Node installs use `@sipp/sipp-server`; npm resolves the matching optional
  platform binary package automatically. Python installs use `sipp-py` for CPU
  and extras such as `sipp-py[cuda]` to pull matching backend distributions
  like `sipp-py-backend-cuda`. Python code still imports `sipp`. Use
  `SIPP_NODE_BACKEND` or `SIPP_PYTHON_BACKEND` when you need to force `cpu`,
  `vulkan`, `cuda`, or `metal`.
- Gateway clients need only the gateway base URL, public target name, and
  application-owned authentication value.

## Next Steps

- [sipp CLI for source checkouts](../sipp/README.md)
- [Browser package](../packages/browser.md)
- [Node.js package](../packages/node.md)
- [Python package](../packages/python.md)
- [Rust package](../packages/rust.md)
- [Gateway](../gateway/README.md)
- [Maintainer source builds](../maintainers/source-builds.md)
