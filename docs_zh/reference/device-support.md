# 设备支持

Sipp 支持在多种设备、操作系统、浏览器和 GPU 加速器上运行。本页记录了受支持的配置、支持级别和已知限制。

## 计算后端

构建配置和运行时环境使用相同的名称选择计算后端。

| 计算后端 | 状态 | 特性标志 | 默认 | 平台 | 备注 |
| --- | --- | --- | --- | --- | --- |
| CPU | 支持 | `native` | 是 | 所有 | 便携方案，无需加速器 |
| CUDA | 支持 | `cuda` | 否 | Linux, Windows | NVIDIA GPU，计算能力 7.5+ |
| Metal | 支持 | `metal` | 否 | macOS | Apple Silicon 和 AMD GPU |
| Vulkan | 支持 | `vulkan` | 否 | Linux, Windows | 需 Vulkan 1.2+ 的 GPU |
| WebGPU | 支持 | `GGML_WEBGPU` (CMake) | 否 | WASM 浏览器 | 仅限浏览器，需 `shader-f16` 特性 |

运行时选择：

* **CLI:** `--backend auto|cpu|cuda|metal|vulkan`
* **Node.js:** `SIPP_NODE_BACKEND=cpu|vulkan|cuda|metal`
* **Python:** `SIPP_PYTHON_BACKEND=cpu|vulkan|cuda|metal`
* **Browser:** 模型加载选项中的 `backend: 'auto' | 'cpu' | 'webgpu'`

若要自动选择后端，请将变量保持未设置状态。

### 各语言包对后端的支持情况

| 计算后端 | Node.js | Python | Rust | 浏览器 (WASM) | 网关 |
| --- | --- | --- | --- | --- | --- |
| CPU | 是 | 是 | 是 | 是 | 是 |
| CUDA | 是 | 是 | 是 | — | 是 |
| Metal | 是 | 是 | 是 | — | — |
| Vulkan | 是 | 是 | 是 | — | 是 |
| WebGPU | — | — | — | 是 | — |

### 其他 llama.cpp 后端（尚未公开）

内置的 llama.cpp 支持其他后端，但 Sipp 尚未公开对应的特性标志。欢迎社区提交贡献。

* SYCL (Intel oneAPI)
* HIP / ROCm (AMD)
* OpenCL
* OpenVINO
* CANN (华为昇腾)
* MUSA (摩尔线程)
* Hexagon (高通 DSP)
* ZenDNN (AMD)
* RPC (远程后端)

这些后端需要在内置 llama.cpp 编译时添加自定义 CMake 标志，无法直接通过 Sipp 的标准构建或打包命令使用。

---

## 桌面端浏览器支持矩阵

下表显示了桌面操作系统上各项功能开始可用的最低浏览器版本。横线（`—`）表示不支持该功能。

| 浏览器 | 支持情况 | WASM st | WASM pthread¹ | WebGPU | WebGPU + f16² | OPFS³ | Workers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chrome (Win, Mac, Linux) | ✅ 已测试 | 57 | 92⁴ | 113 | 113 | 86 | 4 |
| Edge (Win, Mac, Linux) | ❌ 未测试 | 79⁵ | 92⁴ | 113 | 113 | 86 | 79⁵ |
| Firefox (Windows) | ❌ 未测试 | 52 | 79⁴ | 141 | 141 | 111 | 3.5 |
| Firefox (macOS) | ❌ 未测试 | 52 | 79⁴ | 145⁶ | 145⁶ | 111 | 3.5 |
| Firefox (Linux) | ❌ 未测试 | 52 | 79⁴ | ⚠ Nightly | ⚠ Nightly | 111 | 3.5 |
| Safari (macOS) | ❌ 未测试 | 11 | 15.2⁴ | 26 | 26 | 16.4 | 4 |
| Opera (Win, Mac, Linux) | ❌ 未测试 | 44 | 78⁴ | 99 | 99 | 72 | 11.5 |
| ChromeOS | ❌ 未测试 | 57 | 92⁴ | 113 | 113 | 86 | 4 |
| 其他基于 Chromium 的浏览器⁷ | ❌ 未测试 | 57+ | 92⁴ | 113 | 113 | 86+ | 4+ |

**脚注：**

* ¹ WASM pthread 需要服务器返回 `Cross-Origin-Opener-Policy: same-origin` 和 `Cross-Origin-Embedder-Policy: require-corp`（或 `credentialless`）HTTP 响应头。详情见下文 [WASM 线程机制](#wasm-线程机制)。
* ² Sipp 的 WebGPU 后端需要 `shader-f16` WebGPU 特性。其可用性除了取决于浏览器版本外，还取决于 GPU 和驱动程序的支持。
* ³ Origin Private File System（源私有文件系统），用于缓存模型数据。需要安全上下文（HTTPS）。Firefox 在 111 版本之前需要手动开启 `dom.fs.enabled` 选项才能支持 OPFS。
* ⁴ 此版本表示 `SharedArrayBuffer` 在包含跨源隔离响应头时开始可用。更早的版本可能具备此功能但没有响应头要求。
* ⁵ Edge 从 79 版本开始改用 Chromium 引擎。基于 Chromium 的 Edge 自 79 版本开始支持 WASM 单线程（single-thread）和 Worker。旧版 EdgeHTML 引擎自 12 版本支持 Worker，自 16 版本支持 WASM。
* ⁶ Firefox 145 在 macOS 26（ARM64）上启用了 WebGPU。Nightly 版本正在测试 Intel Mac 的支持。
* ⁷ 包括 Brave、Vivaldi、Arc 和其他基于 Chromium 的浏览器。支持情况与底层的 Chromium 版本一致。

---

## 移动端浏览器支持矩阵

| 浏览器 | 支持情况 | WASM st | WASM pthread¹ | WebGPU | WebGPU + f16² | OPFS³ | Workers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chrome (Android) | 🟡 待定 | 57 | 92⁴ | 121⁵ | 121⁵ | 86 | 56 |
| Safari (iOS / iPadOS) | ❌ 未测试 | 11 | 15.2⁴ | 26 | 26 | 16.4 | 5 |
| Safari (visionOS) | ❌ 未测试 | 11 | 15.2⁴ | 26 | 26 | 16.4 | 5 |
| 三星浏览器 (Android) | ❌ 未测试 | 8 | 16⁴ | 24 | 24 | 21 | 4 |
| Opera (Android) | ❌ 未测试 | 44 | 78⁴ | 80 | 80 | 72 | 11.5 |
| Firefox (Android) | ❌ 未测试 | 52 | 79⁴ | ⚠ Beta/Nightly | ⚠ Beta/Nightly | 150 | 52 |
| Android WebView | ❌ 未测试 | 57 | 92⁴ | ⚠ 标志⁶ | ⚠ 标志⁶ | 86 | 56 |

**脚注：**

* ¹ 需要前文 [WASM 线程机制](#wasm-线程机制) 提及的 COOP/COEP HTTP 响应头。
* ² 即使浏览器支持，`shader-f16` 特性在某些移动端 GPU 或驱动程序上依然不可用。
* ³ Origin Private File System（源私有文件系统）。Android 版 Chrome 和三星浏览器支持 OPFS。iOS Safari 从 16.4 版本开始支持。
* ⁴ 此版本表示 `SharedArrayBuffer` 在包含跨源隔离响应头时开始可用。
* ⁵ 搭载高通或 ARM GPU 且系统版本为 Android 12+ 的 Chrome 121 浏览器。目前正逐步支持其他 GPU 厂商（Imagination、三星 Xclipse）。
* ⁶ Android WebView 需要手动开启 `--enable-unsafe-webgpu` 标志。不建议用于生产环境。

---

## WASM 线程机制

Sipp 提供了两种 WASM 运行时产物：

| 产物 | 线程数 | Token 流式传输 | 要求 |
| --- | --- | --- | --- |
| `sipp-wasm.js` (单线程) | 1 | `postMessage` | 无 |
| `sipp-wasm-pthread.js` (pthread) | 最多 4 个⁷ | `SharedArrayBuffer` 环形缓冲区 | COOP + COEP 响应头、安全上下文 |

> ⁷ 默认为 `min(4, navigator.hardwareConcurrency)`。可以通过模型加载选项中的 `runtime.context.n_threads` 覆盖此默认值。

客户端会在运行时自动检测是否支持 pthread：

```ts
function supportsWasmPthreads(): boolean {
  return (
    typeof SharedArrayBuffer !== 'undefined' &&
    globalThis.crossOriginIsolated === true &&
    typeof Worker !== 'undefined'
  );
}
```

托管环境（如 GitHub Pages 或无法控制响应头的共享主机）无法返回 COOP/COEP 响应头时，在客户端选项中配置 `wasmThreading: 'single-thread'`。

---

## 平台与操作系统支持

| 操作系统 | x64 | arm64 | 其他架构 | 可用绑定 |
| --- | --- | --- | --- | --- |
| Linux (glibc) | 是 | 是 | arm, loong64, riscv64, ppc64, s390x | Node.js, Python, Rust |
| Linux (musl) | 是 | 是 | arm, loong64, riscv64 | Node.js |
| Windows (MSVC) | 是 | 是 | ia32 | Node.js, Python, Rust |
| Windows (GNU) | 是 | — | — | Node.js |
| macOS | 是 | 是 | universal2 | Node.js, Python, Rust |
| Android | — | 是 | arm (eabi) | Node.js |
| FreeBSD | 是 | 是 | — | Node.js |
| OpenHarmony | 是 | 是 | arm | Node.js |

### Docker 容器

| 配置文件 | 计算后端 | 宿主操作系统 | 备注 |
| --- | --- | --- | --- |
| CPU | CPU | Linux, macOS, Windows | 处处可用，无 GPU 直通 |
| CUDA | CUDA | Linux, Windows (WSL2) | 需要安装 NVIDIA Container Toolkit |
| Vulkan | Vulkan | 仅限 Linux | Windows Docker Desktop 不支持 Vulkan 直通 |
| Metal | — | — | Linux 容器内无法使用 Metal |

---

## GPU 与加速器支持

### NVIDIA CUDA

Sipp 需要计算能力 7.5 或以上的 NVIDIA GPU。CUDA 13 移除了对 7.5 以下架构的支持。

| 架构 | 计算能力 | 目标 GPU |
| --- | --- | --- |
| Turing | 7.5 | T4, Quadro RTX, GeForce RTX 20 系列 |
| Ampere | 8.0, 8.6 | A100, A10, A40, RTX A6000, GeForce RTX 30 系列 |
| Ada Lovelace | 8.9 | L4, L40S, GeForce RTX 40 系列 |
| Hopper | 9.0 | H100, H200 |
| Blackwell (数据中心) | 10.0 | B100, B200, GB200 |
| Blackwell (消费级/边缘) | 12.0, 12.1 | GeForce RTX 50 系列, RTX PRO Blackwell |

### Vulkan

在 Linux 和 Windows 上，任何搭载 Vulkan 1.2+ 驱动程序的 GPU 均可使用。已在以下硬件上测试通过：

* **NVIDIA:** Turing, Ampere, Ada Lovelace, Hopper (专有驱动)
* **AMD:** RDNA 2 及更新版本 (AMDGPU PRO 或 RADV)
* **Intel:** Gen12/Xe 及更新版本 (ANV)

Windows Docker Desktop 不支持 Vulkan 后端。

### Metal

* **Apple Silicon:** M1, M2, M3, M4 系列
* **AMD:** macOS 支持的 GPU (Radeon Pro 系列)

Metal 仅限 macOS，Docker 容器内不可用。

### WebGPU (浏览器)

几乎所有宿主浏览器支持的 WebGPU GPU 都可以工作，但 Sipp 需要 `shader-f16` 特性才能启用 WebGPU 加速。常见配置如下：

| GPU 系列 | Chrome (D3D12) | Chrome (Vulkan) | Firefox (wgpu) | Safari (Metal) |
| --- | --- | --- | --- | --- |
| NVIDIA | 是 | 是 (Linux) | 是 | — |
| AMD | 是 | 是 (Linux) | 是 | 是 |
| Intel 集成显卡 | 是 | 是 (Linux) | 是 | 是 |
| Apple Silicon | — | — | 是 | 是 |
| 高通 (Android) | 是 | — | — | — |
| ARM Mali | 是 (Android) | — | — | — |

---

## 语言绑定支持

| 软件包 | 安装命令 | 状态 | 运行时 | 主要用途 |
| --- | --- | --- | --- | --- |
| 浏览器 (`@sipp/sipp`) | `npm install @sipp/sipp` | 已发布 (npm) | WASM / WebGPU | 浏览器本地 GGUF 推理、网关客户端 |
| Node.js (`@sipp/sipp-server`) | `npm install @sipp/sipp-server` | 已发布 (npm) | N-API 原生 | 服务器进程、路由处理程序、后端服务 |
| Python (`sipppy`) | `pip install sipppy` | 已发布 (PyPI) | PyO3 原生 | Python 服务、脚本、网关客户端 |
| Rust (`sipp-rs`) | `cargo add sipp-rs` | 已发布 (crates.io) | 纯 Rust 门面 | Rust 应用程序和服务 |
| 网关服务器 | 源码构建 | 仅限源码 | Axum 二进制文件 | 本地和提供商目标的 HTTP 网关 |
| 网关 Docker | 从源码构建 Docker 镜像 | 仅限源码 | 容器 | 生产环境容器工作流 |
| 网关工具包 | 源码制品 | 仅限源码 | Rust crate | 自定义网关应用 |

---

## 局限性与待办事项

* **Rust crates.io 发布**：由于 `sipp-sys` 依赖于私有 llama.cpp 子模块，目前处于阻塞状态。仅发布了源码制品。
* **网关服务器**：尚未提供预编译的二进制文件或公共容器镜像。必须从源码构建。
* **Windows Docker Vulkan**：暂不支持。在开启了 WSL2 的 Windows 上请改用 CUDA 或 CPU 配置文件。
* **macOS Docker**：仅支持 CPU。Linux Docker 容器内无法运行 Metal。
* **Android 与 iOS**：目前不是首要目标。浏览器 WASM 包可以在移动端 Web 浏览器上运行，但尚未提供原生的 Android 或 iOS 软件包。
* **Chrome (桌面端)**：主要测试对象。其他桌面端浏览器（Edge、Firefox、Safari、Opera 及 Chromium 衍生版本）尚未测试。
* **移动端浏览器支持**：尚未测试验证。Android 版 Chrome 是下一个测试目标。
* **Linux 和 Android 上的 Firefox WebGPU**：活跃开发中 (Nightly / Beta)。macOS Intel 上的 Firefox WebGPU 也在推进中。
* **网关**：目前支持 OpenAI、兼容 OpenAI 的提供商以及 Anthropic。后续会继续增加对其他提供商的支持。
