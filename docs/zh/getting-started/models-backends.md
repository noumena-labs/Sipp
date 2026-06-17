# 模型与后端

Sipp 本地推理依赖 GGUF 模型文件。文本工作流需要文本 GGUF 模型，嵌入工作流需要模型明确支持并开启嵌入特性，视觉对话工作流则同时需要模型 GGUF 和其专属的 projector 文件。

## 指定模型路径

在本地环境使用 Node.js、Python 或 Rust 包时，传入显式的 GGUF 模型路径；浏览器端代码则传入 GGUF 模型的 URL：

- 浏览器：`source: '/models/model.gguf'`
- Node.js：`modelPath: '/path/to/model.gguf'`
- Python：`LocalModelDescriptor('/path/to/model.gguf')`
- Rust：`EndpointDescriptor::local(model_path, config)`

源码示例和冒烟测试可直接加载 `.build/models` 目录下缓存的示例模型，详情见[源码构建](../maintainers/source-builds.md)。

## 原生后端

构建选项和运行时配置中统一使用相同的后端名称：

- `cpu`：高兼容性默认后端。
- `vulkan`：适用于支持 Vulkan 驱动的 GPU。
- `cuda`：NVIDIA CUDA 后端。
- `metal`：macOS 平台的 Apple Metal 后端。

运行时指定后端的方式取决于所用包：

- Node.js：`SIPP_NODE_BACKEND=cpu|vulkan|cuda|metal`
- Python：`SIPP_PYTHON_BACKEND=cpu|vulkan|cuda|metal`
- CLI：`--backend auto|cpu|cuda|metal|vulkan`

不设置任何环境变量或启动参数时，系统会自动选择合适的后端。

维护者可用 `sipp` 或 `cargo xtask` 编译特定后端的构建产物，详情见[源码构建](../maintainers/source-builds.md)。

包与后端之间的兼容矩阵及 llama.cpp/ggml 操作层面的指南，见[后端兼容矩阵](../guides/backend-matrix.md)。
