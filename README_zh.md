<p align="center">
  <img src="docs/assets/sipp_logo_no_text.svg" alt="Sipp Logo" width="200">
</p>

<div id="user-content-toc" align="center">
  <ul style="list-style: none;">
    <summary>
      <h1>Sipp</h1>
    </summary>
  </ul>
</div>

---

<p align="center">
  <a href="https://github.com/noumena-labs/Sipp/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/noumena-labs/Sipp/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/noumena-labs/Sipp/actions/workflows/docs.yml"><img alt="Docs" src="https://github.com/noumena-labs/Sipp/actions/workflows/docs.yml/badge.svg"></a>
  <a href="https://github.com/noumena-labs/Sipp/actions/workflows/coverage.yml"><img alt="Coverage" src="https://github.com/noumena-labs/Sipp/actions/workflows/coverage.yml/badge.svg"></a>
  <a href="https://github.com/noumena-labs/Sipp/actions/workflows/release.yml"><img alt="Release" src="https://github.com/noumena-labs/Sipp/actions/workflows/release.yml/badge.svg"></a>
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue">
</p>

<div align="center">
  <a href="docs/zh/getting-started/quickstarts.md">文档</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://discord.gg/abzgfghhrq">Discord</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://github.com/noumena-labs/Sipp/issues">问题反馈</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="docs/zh/roadmap.md">路线图</a>
  <br />
  <a href="README.md">English README</a>
  <br />
</div>

> [!WARNING]
> Sipp 正在积极开发中。随着运行时层持续优化，API 或者运行时可能会出现明显。它目前可能还不适合生产环境。如果你发现问题、或者功能缺失，请提交 GitHub issue。

### [阅读文档 →](docs/zh/README.md)
### [English README →](README.md)

## 什么是 Sipp？

Sipp 是一个一体化的 AI 框架，用于构建 Web、桌面和边缘应用。它以统一 SDK 的形式，为本地推理、服务商推理和云网关推理提供一致的 API。

Sipp 的核心是 **Sipp Engine**：一个可在浏览器、桌面和裸金属云基础设施中运行的高速运行时, 同时具备低启动延迟和较小的内存占用。

```javascript
import { SippClient } from '@sipphq/sipp';
const blender = new SippClient();

// 1. 初始化高速本地 WebGPU 或 CUDA 推理
const juice = await blender.add('edge', { kind: 'local', source: '/models/llama3.gguf' });

// 2. 或使用同一套接口连接到安全的云代理
const ice = await blender.add('cloud', { kind: 'gateway', baseUrl: 'https://gateway.example.com/v1/' });

// 用对称 API 在任意端点上无缝运行推理
const [smoothie, snowcone] = await Promise.all([
  blender.chat([{ role: 'user', content: 'Explain Sipp.' }], { endpoint: juice }),
  blender.chat([{ role: 'user', content: 'Create a Sipp app.' }], { endpoint: ice })
]);
```

统一 SDK 让你可以在本地计算和云端计算之间动态拆分并优化复杂应用逻辑。你不再需要处理碎片化的 Web 运行时、彼此割裂的桌面原生封装，或用于保护 API key 的自定义中间件；只需要使用 Sipp。

它把**高性能 WebGPU 引擎**和安全的容器化网关代理打包成一个简洁的工具包。后续版本将重点推进嵌入式向量记忆、设备端 PII 脱敏，以及自动智能路由。参见[路线图](docs/zh/roadmap.md)。

```bash
sipp build wasm                # 编译高性能 WebGPU 资源
sipp run demos serve chat      # 启动本地硬件加速测试画布
```

## 性能基准

你可以在这里自行运行：[benchmark.sipp.sh/benchmark](https://benchmark.sipp.sh/benchmark)

| 运行时 / 框架 | TTFT (ms) ↓ | Decode (tok/s) ↑ | E2E Latency (ms) ↓ |
| --- | --- | --- | --- |
| **Sipp** | **24.3** *(最佳)* | **77.07** *(最佳)* | **6,655** *(最佳)* |
| **WebLLM** | 160.0 *(6.55x)* | 25.80 *(2.99x)* | 19,930 *(2.99x)* |
| **Transformers.js** | 301.0 *(12.38x)* | 33.25 *(2.32x)* | 15,670 *(2.35x)* |

---

> **免责与指标说明：**
> * **TTFT (Time to First Token)：** 首 token 时间，单位为毫秒 (ms)。**越低越好**。
> * **Decode：** 解码速度，单位为 tokens per second (tok/s)。**越高越好**。
> * **E2E Latency (End-to-End Latency)：** 端到端延迟，单位为毫秒 (ms)。**越低越好**。
> * *测试环境为 Nvidia GTX 3080，1 次预热，3 次正式测量。结果为所有测量运行的平均值。*

## 安装

Sipp 支持 Web 浏览器、桌面应用封装、服务端环境和原生运行时。请根据目标运行环境安装对应实现层：

```sh
# 适用于 Web 浏览器、Next.js 和 TanStack 应用
npm install @sipphq/sipp

# 适用于 Node.js 后端部署（带原生 CUDA/Metal 编译）
npm install @sipphq/sipp-server

# 适用于原生系统开发和应用嵌入
cargo add sipp-rs

# 适用于 Python 自动化和数据工程管线
# （sippy wheel 目前从 GitHub Releases 发布；完整 PyPI 正在推进中）
# pip install sipppy

# 通过 Docker 部署安全的云网关服务实例
# （云网关未来会开放，目前从源码构建）
# docker pull noumena/sipp-gateway
```

---

## 运行时与版本形态

多数开发者应优先使用预构建、已发布的包，而不是直接从 monorepo 源码编译。

| 运行环境 | 模块 | 安装 | 文档 |
| --- | --- | --- | --- |
| **Browser** | Sipp Edge | `npm install @sipphq/sipp` | [浏览器包](docs/zh/packages/browser.md) |
| **Node.js** | Sipp Core | `npm install @sipphq/sipp-server` | [Node.js 包](docs/zh/packages/node.md) |
| **Rust** | Sipp Core | `cargo add sipp-rs` | [Rust 包](docs/zh/packages/rust.md) |
| **Python** | Sipp Core | Release 页面提供 wheel | [Python 包](docs/zh/packages/python.md) |
| **Gateway Server** | Sipp Cloud | 从源码构建 | [网关服务](docs/zh/gateway/server.md) |
| **Gateway Toolkit** | Sipp Cloud | 从源码构建 | [网关工具包](docs/zh/gateway/toolkit.md) |

---

## 快速上手

### 1. Edge 快速上手（硬件加速客户端推理）

初始化本地引擎客户端，直接在客户端的 shader 上使用 WebGPU 执行模型权重。

```bash
npm install @sipphq/sipp
```

```javascript
import { SippClient } from '@sipphq/sipp';

const messages = [
  { role: 'system', content: 'Answer concisely.' },
  { role: 'user', content: 'Explain Sipp in one sentence.' },
];

const client = new SippClient();
const endpoint = await client.add('default', {
  kind: 'local',
  source: '/models/model.gguf',
});

const run = client.chat(messages, {
  endpoint,
  maxTokens: 64,
});

console.log((await run.response).text);
await client.close();
```

### 2. 云网关快速上手（预防式云代理）

云网关客户端使用完全相同的 `SippClient` API。网关负责管理模型路径、服务商凭证、访问策略和集中式指标追踪；你的客户端应用代码只需要网关路由目标 URL。

```javascript
import { SippClient } from '@sipphq/sipp';

const client = new SippClient();
const endpoint = await client.add('gateway', {
  kind: 'gateway',
  target: 'upstream-cluster',
  baseUrl: 'https://gateway.example.com/v1/',
  authentication: { kind: 'bearer', value: await getGatewayToken() },
});

const run = client.query('Explain gateway inference.', {
  endpoint,
  maxTokens: 64,
});

console.log((await run.response).text);
await client.close();
```

---

## 原生 Web 框架蓝图

Sipp 包含集成蓝图，可处理 Server-Sent Events (SSE) 流式输出、serverless 路由编排和客户端 hydration 模式。

- [Next.js](docs/zh/packages/frameworks/nextjs.md)：App Router route handlers、Client Components、网关代理和流式输出。
- [TanStack](docs/zh/packages/frameworks/tanstack.md)：TanStack Start server functions 和 TanStack Query 模式。
- [React And Vite](docs/zh/packages/frameworks/vite-react.md)：浏览器包设置、WASM 资源、OPFS 模型加载和网关示例。

## 文档

完整文档位于 [docs/zh](docs/zh/README.md)。在源码 checkout 中，可以使用 `sipp docs` CLI 工具构建或启动文档站点：

```bash
sipp docs build
sipp docs serve
```

`sipp docs` 会在缺失时自动评估并安装所需的 mdBook 工具，并配置技术文档页面所使用的 Mermaid 编译资源。

---

## 技术路线图

我们的核心开发方向是扩展端云混合系统所需的边缘-云基础设施，让本地和云端资源可以无缝编排。

如需查看架构和长期研究计划的详细结构，请阅读完整的 [Sipp 技术路线图](docs/zh/roadmap.md)。

---

## 维护者与贡献者

要初始化 workspace 环境、启用跨平台 profile 并运行单元测试，请使用集成 CLI 环境脚本：

```bash
source ./setup.sh
sipp doctor
sipp test list
```

*(在 Windows 平台上，如果不使用 Git Bash 或 WSL，请在 PowerShell 中执行 `.\setup.ps1`，或通过传统 CMD 执行 `setup.cmd`。)*

### 常见架构编译任务：

```bash
sipp build wasm && sipp run examples serve browser
sipp build node --backend cpu && node examples/node/query.mjs <model.gguf> "Explain Sipp."
sipp build python --backend cpu && python examples/python/query.py <model.gguf> "Explain Sipp."
sipp run demos serve chat
```

更完整的验证步骤请参阅[源码构建文档](docs/zh/maintainers/source-builds.md)和完整的[测试框架套件](docs/zh/testing.md)。

---

## 仓库结构

* [crates](crates%2FREADME.md)：发布的核心 `sipp-rs` 和底层后端 `sipp-sys` Rust crate。
* [lib](lib%2Fgateway%2FREADME.md)：高级语言包表面和网关代理工具包。
* [bindings](bindings%2FREADME.md)：原生 Node.js 绑定、Python 扩展和编译到浏览器的 WASM 目标。
* [apps](apps%2FREADME.md)：第一方用户界面和监控实现。
* [examples](examples%2FREADME.md)：小型、可运行的框架集成蓝图。
* [demos](demos%2FREADME.md)：基于公共包表面的高级浏览器沙箱。
* [tools/playground](tools%2Fplayground%2FREADME.md)：实时浏览器运行时分析和硬件执行诊断。
* `xtask/`：内部 cargo 自动化引擎，用于驱动构建、测试和包发布管线。

## 许可证

Sipp 基于 Apache-2.0 License 授权。第三方依赖保留其各自上游开源许可约束和文档要求；详见[第三方声明](THIRD_PARTY_NOTICES.md)。
