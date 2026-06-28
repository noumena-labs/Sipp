# Device Support

Sipp runs across a range of devices, operating systems, browsers, and GPU accelerators. This page documents which configurations are supported, at what level, and any known limitations.

## Compute Backends

Backend names are shared across build configuration and runtime selection. The same name selects the backend in each surface.

| Backend | Status | Feature flag | Default | Platforms | Notes |
| --- | --- | --- | --- | --- | --- |
| CPU | Supported | `native` | Yes | All | Portable fallback, no accelerator required |
| CUDA | Supported | `cuda` | No | Linux, Windows | NVIDIA GPUs, compute capability 7.5+ |
| Metal | Supported | `metal` | No | macOS | Apple Silicon and AMD GPUs; use CPU on Intel integrated GPUs |
| Vulkan | Supported | `vulkan` | No | Linux, Windows | Vulkan 1.2+ GPU required |
| WebGPU | Supported | `GGML_WEBGPU` (CMake) | No | WASM browsers | Browser-only, requires `shader-f16` |

Runtime selection:

* **CLI:** `--backend auto|cpu|cuda|metal|vulkan`
* **Node.js:** `SIPP_NODE_BACKEND=cpu|vulkan|cuda|metal`
* **Python:** `SIPP_PYTHON_BACKEND=cpu|vulkan|cuda|metal`
* **Browser:** `backend: 'auto' | 'cpu' | 'webgpu'` in model load options

Leave the variable unset for automatic backend selection.

### Backend Availability by Package

| Backend | Node.js | Python | Rust | Browser (WASM) | Gateway |
| --- | --- | --- | --- | --- | --- |
| CPU | Yes | Yes | Yes | Yes | Yes |
| CUDA | Yes | Yes | Yes | — | Yes |
| Metal | Yes | Yes | Yes | — | — |
| Vulkan | Yes | Yes | Yes | — | Yes |
| WebGPU | — | — | — | Yes | — |

### Additional llama.cpp Backends (Not Yet Exposed)

The vendored llama.cpp supports additional backends that Sipp does not currently expose as feature flags. Community contributions are welcome.

* SYCL (Intel oneAPI)
* HIP / ROCm (AMD)
* OpenCL
* OpenVINO
* CANN (Huawei Ascend)
* MUSA (Moore Threads)
* Hexagon (Qualcomm DSP)
* ZenDNN (AMD)
* RPC (remote backend)

These backends require custom CMake flags on top of the vendored llama.cpp build and are not available through Sipp's standard build or package commands.

---

## Desktop Browser Support Matrix

The table below shows the first browser version where each feature is available for desktop operating systems. A dash (`—`) means the feature is not supported.

| Browser | Support | WASM st | WASM pthread¹ | WebGPU | WebGPU + f16² | OPFS³ | Workers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chrome (Windows) | ✅ Tested 149.0.7827.200 | 57 | 92⁴ | 113 | 113 | 86 | 4 |
| Chrome (macOS) | ✅ Tested 149.0.7827.199 | 57 | 92⁴ | 113 | 113 | 86 | 4 |
| Edge (Windows) | ✅ Tested 149.0.4022.80 | 79⁵ | 92⁴ | 113 | 113 | 86 | 79⁵ |
| Firefox (Windows) | 🟡 CPU tested 152.0.2 | 52 | 79⁴ | 141 | 141 | 111 | 3.5 |
| Firefox (macOS) | 🟡 CPU tested 152.0.3 | 52 | 79⁴ | 145⁶ | 145⁶ | 111 | 3.5 |
| Firefox (Linux) | ❌ Untested | 52 | 79⁴ | ⚠ Nightly | ⚠ Nightly | 111 | 3.5 |
| Safari (macOS) | ✅ Tested 26.4 (CPU) · STP (WebGPU) | 11 | 15.2⁴ | 26 | 26 | 16.4 | 4 |
| Opera (Win, Mac, Linux) | ❌ Untested | 44 | 78⁴ | 99 | 99 | 72 | 11.5 |
| ChromeOS | ❌ Untested | 57 | 92⁴ | 113 | 113 | 86 | 4 |
| Other Chromium-based⁷ | ❌ Untested | 57+ | 92⁴ | 113 | 113 | 86+ | 4+ |

**Footnotes:**

* ¹ WASM pthread requires the server to send `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` (or `credentialless`) HTTP headers. See [WASM Threading](https://www.google.com/search?q=%23wasm-threading) below.
* ² The `shader-f16` WebGPU feature is required by Sipp's browser WebGPU backend. Availability depends on GPU and driver support in addition to the browser version.
* ³ Origin Private File System. Used for model data caching. Requires a secure context (HTTPS). Firefox support is behind the `dom.fs.enabled` preference until version 111.
* ⁴ Version listed is when `SharedArrayBuffer` became available with cross-origin isolation headers. Earlier versions may have had the feature without the header requirement.
* ⁵ Edge switched to a Chromium engine at version 79. The Chromium-based Edge supports WASM single-thread from 79, Workers from 79. The legacy EdgeHTML engine supported Workers from version 12 and WASM from version 16.
* ⁶ Firefox 145 enables WebGPU on macOS version 26 (ARM64). Intel Mac support is in progress in Nightly.
* ⁷ Includes Brave, Vivaldi, Arc, and other Chromium-derived browsers. Versions match their underlying Chromium release.

---

## Mobile Browser Support Matrix

| Browser | Support | WASM st | WASM pthread¹ | WebGPU | WebGPU + f16² | OPFS³ | Workers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chrome (Android) | 🟡 Pending | 57 | 92⁴ | 121⁵ | 121⁵ | 86 | 56 |
| Safari (iOS / iPadOS) | ❌ Untested | 11 | 15.2⁴ | 26 | 26 | 16.4 | 5 |
| Safari (visionOS) | ❌ Untested | 11 | 15.2⁴ | 26 | 26 | 16.4 | 5 |
| Samsung Internet (Android) | ❌ Untested | 8 | 16⁴ | 24 | 24 | 21 | 4 |
| Opera (Android) | ❌ Untested | 44 | 78⁴ | 80 | 80 | 72 | 11.5 |
| Firefox (Android) | ❌ Untested | 52 | 79⁴ | ⚠ Beta/Nightly | ⚠ Beta/Nightly | 150 | 52 |
| Android WebView | ❌ Untested | 57 | 92⁴ | ⚠ Flag⁶ | ⚠ Flag⁶ | 86 | 56 |

**Footnotes:**

* ¹ Requires COOP/COEP HTTP headers as described in [WASM Threading](https://www.google.com/search?q=%23wasm-threading).
* ² The `shader-f16` feature may not be available on all mobile GPU/driver combinations even when the browser version supports it.
* ³ Origin Private File System. Chrome for Android and Samsung Internet support OPFS. iOS Safari supports OPFS from 16.4.
* ⁴ Version listed is when `SharedArrayBuffer` became available with cross-origin isolation headers.
* ⁵ Chrome 121 on Android 12+ with Qualcomm or ARM GPUs. Support on other GPU vendors (Imagination, Samsung Xclipse) is still rolling out.
* ⁶ Android WebView requires the `--enable-unsafe-webgpu` flag. Not recommended for production use.

---

## WASM Threading

Sipp ships pthread WASM runtime artifacts by default:

| Artifact | Backend/runtime | Thread count | Token streaming | Requirements |
| --- | --- | --- | --- | --- |
| `sipp-wasm-pthread.js` | WebGPU + JSPI | up to 4⁷ | `SharedArrayBuffer` ring | COOP + COEP headers, secure context |
| `sipp-wasm-pthread-cpu-nojspi.js` | CPU-only, no JSPI | up to 4⁷ | `SharedArrayBuffer` ring | COOP + COEP headers, secure context |

> ⁷ Defaults to `min(4, navigator.hardwareConcurrency)`. Override with `runtime.context.n_threads` in model load options.

The client auto-selects the CPU non-JSPI artifact for Firefox-like runtimes and
for any runtime that does not expose JSPI (`WebAssembly.Suspending`), and the
WebGPU+JSPI artifact elsewhere. Current Safari lacks JSPI, so it loads the CPU
non-JSPI artifact; a Safari build that ships JSPI (Safari Technology Preview /
27 beta, or 26.4+ with the experimental flag enabled) is detected at runtime and
upgraded to the WebGPU+JSPI artifact. The bundled runtime requires pthread
availability:

```ts
function supportsWasmPthreads(): boolean {
  return (
    typeof SharedArrayBuffer !== 'undefined' &&
    globalThis.crossOriginIsolated === true &&
    typeof Worker !== 'undefined'
  );
}

```

Single-thread artifacts are not included in the default browser package. Hosts
that cannot serve COOP/COEP headers must provide a custom single-thread
runtime with `wasmThreading: 'single-thread'`, `moduleUrl`, and `wasmUrl`.

---

## Platform & OS Support

| OS | x64 | arm64 | Other architectures | Available bindings |
| --- | --- | --- | --- | --- |
| Linux (glibc) | Yes | Yes | arm, loong64, riscv64, ppc64, s390x | Node.js, Python, Rust |
| Linux (musl) | Yes | Yes | arm, loong64, riscv64 | Node.js |
| Windows (MSVC) | Yes | Yes | ia32 | Node.js, Python, Rust |
| Windows (GNU) | Yes | — | — | Node.js |
| macOS | Yes | Yes | universal2 | Node.js, Python, Rust |
| Android | — | Yes | arm (eabi) | Node.js |
| FreeBSD | Yes | Yes | — | Node.js |
| OpenHarmony | Yes | Yes | arm | Node.js |

### Docker Containers

| Profile | Backend | Host OS | Notes |
| --- | --- | --- | --- |
| CPU | CPU | Linux, macOS, Windows | Works everywhere, no GPU passthrough |
| CUDA | CUDA | Linux, Windows (WSL2) | Requires NVIDIA Container Toolkit |
| Vulkan | Vulkan | Linux only | Windows Docker Desktop does not support Vulkan passthrough |
| Metal | — | — | Metal unavailable inside Linux containers |

---

## GPU & Accelerator Support

### NVIDIA CUDA

Sipp targets NVIDIA GPUs with compute capability 7.5 and above. CUDA 13 removes support for architectures below 7.5.

| Architecture | Compute Capability | Target GPUs |
| --- | --- | --- |
| Turing | 7.5 | T4, Quadro RTX, GeForce RTX 20-series |
| Ampere | 8.0, 8.6 | A100, A10, A40, RTX A6000, GeForce RTX 30-series |
| Ada Lovelace | 8.9 | L4, L40S, GeForce RTX 40-series |
| Hopper | 9.0 | H100, H200 |
| Blackwell (Data Center) | 10.0 | B100, B200, GB200 |
| Blackwell (Consumer/Edge) | 12.0, 12.1 | GeForce RTX 50-series, RTX PRO Blackwell |

### Vulkan

Any GPU with Vulkan 1.2 or later driver support works on Linux and Windows. Tested on:

* **NVIDIA:** Turing, Ampere, Ada Lovelace, Hopper (proprietary driver)
* **AMD:** RDNA 2 and later (AMDGPU PRO or RADV)
* **Intel:** Gen12/Xe and later (ANV)

Windows Docker Desktop does not support the Vulkan backend.

macOS source builds can compile Vulkan through the LunarG SDK, but LunarG's
macOS drivers translate Vulkan to Metal. Sipp does not publish macOS Vulkan
packages because the native Metal backend is simpler for normal macOS use and
macOS Vulkan adds loader/ICD runtime requirements.

### Metal

* **Apple Silicon:** M1, M2, M3, M4 series
* **AMD:** GPUs supported by macOS (Radeon Pro series)

Metal is macOS-only and unavailable inside Docker containers. Intel integrated
GPUs expose Metal, but Sipp does not treat them as a recommended Metal target;
use the CPU backend on those Macs unless you have tested the exact model,
context size, and device and confirmed that Metal is stable and faster than CPU.

Apple Silicon can run x64 processes through Rosetta 2. A `darwin-x64` Node or
Python native package is only used by an x64 Node/Python process; native arm64
Node/Python installations use the `darwin-arm64` packages and are the preferred
path on Apple Silicon.

### WebGPU (Browser)

Any GPU that the host browser exposes as a WebGPU adapter may work, but Sipp requires the `shader-f16` feature for WebGPU acceleration. Common configurations:

| GPU Family | Chrome (D3D12) | Chrome (Vulkan) | Firefox (wgpu) | Safari (Metal) |
| --- | --- | --- | --- | --- |
| NVIDIA | Yes | Yes (Linux) | Yes | — |
| AMD | Yes | Yes (Linux) | Yes | Yes |
| Intel integrated | Yes | Yes (Linux) | Yes | Yes |
| Apple Silicon | — | — | Yes | Yes |
| Qualcomm (Android) | Yes | — | — | — |
| ARM Mali | Yes (Android) | — | — | — |

#### Firefox Runtime Findings

Firefox 152.0.2 exposes the required WASM pthread, worker, WebGPU, and
`shader-f16` capabilities on tested Windows configurations. With Firefox's
experimental JSPI support enabled, the JSPI runtime path was functional: models
loaded and generated tokens. It was not performant enough to ship. The Firefox
browser runtime uses the pthread CPU no-JSPI artifact.

The Firefox browser path is pthread CPU no-JSPI. It still requires
`SharedArrayBuffer`, workers, and COOP/COEP headers.

#### Safari Runtime Findings

Most shipping Safari versions do not expose JSPI. The client detects this and selects the pthread CPU no-JSPI artifact instead, so Safari runs on the CPU backend rather than failing to boot. JSPI is being introduced experimentally (Safari 26.4 behind a flag, Safari Technology Preview / 27 beta) Safari still requires `SharedArrayBuffer`, workers, and COOP/COEP headers.

---

## Language Binding Support

| Package | Install command | Status | Run time | Primary use |
| --- | --- | --- | --- | --- |
| Browser (`@sipphq/sipp`) | `npm install @sipphq/sipp` | Published (npm) | WASM / WebGPU | Browser-local GGUF inference, gateway clients |
| Node.js (`@sipphq/sipp-server`) | `npm install @sipphq/sipp-server` | Published (npm) | N-API native | Server processes, route handlers, backend services |
| Python (`sipppy`) | `pip install sipppy` | Published (PyPI) | PyO3 native | Python services, scripts, gateway clients |
| Rust (`sipp-rs`) | `cargo add sipp-rs` | Published (crates.io) | Native-backed Rust crate | Rust applications and services |
| Gateway server | Source-built | Source only | Axum binary | HTTP gateway for local and provider targets |
| Gateway Docker | Docker from source | Source only | Container | Production container workflows |
| Gateway toolkit | Source artifact | Source only | Rust crate | Custom gateway applications |

---

## Limitations & Work in Progress

* **Gateway server** does not have a published binary or public container image yet. It must be built from source.
* **Windows Docker Vulkan** is not supported. Use the CUDA or CPU profiles on Windows with WSL2.
* **macOS Docker** is CPU-only. Metal cannot run inside a Linux Docker container.
* **Android and iOS** are not first-class package targets. The browser WASM package works on mobile web browsers, but no native Android or iOS packages are published.
* **Chrome (desktop)** is the primary tested browser target. Other desktop browsers (Edge, Firefox, Safari, Opera, Chromium derivatives) are untested.
* **Mobile browser support** has not been validated yet. Chrome (Android) is the next target for testing.
* **Firefox WebGPU on Linux and Android** is in active development (Nightly / Beta). Firefox WebGPU on macOS Intel is also in progress.
* **Gateways** are compatible with OpenAI and OpenAI-compatible providers plus Anthropic. Additional provider support is added over time.
