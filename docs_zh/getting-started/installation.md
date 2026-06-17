# 安装指南

根据应用所使用的运行时安装相应包。所有客户端包使用相同的模型：注册端点，保存返回的端点引用，在执行 `query`、`chat` 或 `embed` 时指定该端点。

## 软件包安装

| 平台 | 安装命令 | 适用场景 |
| --- | --- | --- |
| 浏览器 | `npm install @sipp/sipp` | 浏览器本地 GGUF 推理及网关客户端。 |
| Node.js | `npm install @sipp/sipp-server` | Node.js 本地推理及网关客户端。 |
| Python | `pip install sipp-py` | Python 本地推理及网关客户端。 |
| Python CUDA | `pip install "sipp-py[cuda]"` | 使用 CUDA 后端 Wheel 的 Python 本地推理。 |
| Python Vulkan | `pip install "sipp-py[vulkan]"` | 使用 Vulkan 后端 Wheel 的 Python 本地推理。 |
| Python Metal | `pip install "sipp-py[metal]"` | macOS 上使用 Metal 后端 Wheel 的 Python 本地推理。 |
| Rust | `cargo add sipp-rs` | Rust 本地推理及网关客户端。 |

当前发布工作流会发布浏览器 npm 包、Node npm 包、Python Wheel 和 Rust 源码 Crate, 但尚未发布独立的 gateway-server 二进制文件、容器镜像或 `cargo install` 目标。在官方服务器制品发布前，部署网关服务请使用源码签出及 Dockerfile。

## 运行时要求

- 本地推理需要兼容的 GGUF 模型文件或浏览器端提供的 GGUF 资源。
- 浏览器本地推理需要支持 WebAssembly 的现代浏览器；WebGPU 加速取决于浏览器和设备支持。具体请查看 [设备支持](../references/device-support.md)。
- Node 安装使用 `@sipp/sipp-server`，npm 会自动解析匹配的平台二进制包。Python CPU 包使用 `sipp-py`，CUDA、Vulkan、Metal 后端通过 `sipp-py[cuda]` 等 extras 拉取匹配的 `sipp-py-backend-*` 分发包；Python 代码仍然导入 `sipp`。如需强制指定 `cpu`、`vulkan`、`cuda` 或 `metal`，设置环境变量 `SIPP_NODE_BACKEND` 或 `SIPP_PYTHON_BACKEND`。
- 网关客户端只需要网关基础 URL、公共目标名称和应用专属的认证凭据。

## 后续步骤

- [源码安装对应的 sipp CLI](../sipp/README.md)
- [Browser 包](../packages/browser.md)
- [Node.js 包](../packages/node.md)
- [Python 包](../packages/python.md)
- [Rust 包](../packages/rust.md)
- [网关服务](../gateway/README.md)
- [维护者源码构建](../maintainers/source-builds.md)
