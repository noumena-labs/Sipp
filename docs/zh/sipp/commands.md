# 常用命令

`sipp` 将代码库的自动化任务划分为若干命令族。运行 `sipp <group> --help` 可查看自动生成的帮助信息和当前支持的选项。

## 健康检查

```bash
sipp doctor
sipp doctor --target wasm
sipp doctor --target node --backend vulkan
sipp toolchain status
```

`doctor` 在不安装或删除任何内容的前提下检查本地环境是否就绪。
`toolchain status` 报告由 xtask 管理的工具（Bun、Python、uv、Emscripten、Ninja）状态。CUDA 为系统级安装，xtask 仅报告状态，不负责安装或卸载。

## 构建

```bash
sipp build core
sipp build wasm
sipp build node --backend cpu
sipp build python --backend vulkan
sipp build cli --backend all
sipp build gateway-server --backend cpu
sipp build all
```

`build all` 构建所有主要目标，默认生成 CPU 原生输出，不为每个包构建所有后端变体。

支持的后端：

- `cpu`：移植性最好的默认后端。
- `cuda`：NVIDIA CUDA 后端，需本地安装 CUDA Toolkit。
- `metal`：macOS Apple Metal 后端。
- `vulkan`：Vulkan 后端，xtask 会在需要时自动准备 Vulkan SDK。
- `all`：为选定目标构建当前主机支持的所有后端。

## 运行

```bash
sipp run examples serve browser --port 5173
sipp run examples serve gateway-local --model .build/models/model.gguf --bind 127.0.0.1:8787
sipp run examples gateway rust --case query
sipp run demos serve chat
sipp run tools serve playground
sipp run gateway-server check --config apps/gateway-server/config/local.toml
sipp run gateway-server serve --config apps/gateway-server/config/local.toml --backend cpu
```

`run` 用于执行当前有的程序、网关服务、服务器示例以及非测试诊断。执行测试请用 `sipp test`。

## 文档

```bash
sipp docs build
sipp docs serve
sipp docs build --lang zh
```

`docs build` 在缺失时自动安装 `mdbook` 和 `mdbook-mermaid`，将 Mermaid JavaScript 资源提取到 `theme/`，生成的静态页面写入 `book/`。默认构建还会在 `book/zh` 目录下生成中文版文档。`docs serve` 执行相同的准备工作，构建相同的目录树，并在 `localhost:3000` 上启动服务。页面顶部提供 `/` 和 `/zh/` 之间的语言切换开关。使用 `--lang zh` 可以只构建中文版本，或优先显示中文预览地址。

## 测试

```bash
sipp test list
sipp test list --group unit --layer interface --cases --search router --format json
sipp test unit group full
sipp test unit suite rust-crates --package sipp
sipp test unit suite node-package --backend cpu
sipp test unit suite browser-package
sipp test smoke suite example-node --backend cpu
sipp test smoke group local-model --backend cpu
sipp test verify --changed
sipp test verify --target public-docs
```

执行需要模型的冒烟测试时，省略 `--model` 参数会默认使用 setup 阶段下载到 `.build/models` 的缓存模型。完整的测试套件目录见[测试](../testing.md)。

## 清理

```bash
sipp clean --dry-run
sipp clean
sipp clean --purge
sipp clean --toolchains
```

`clean` 清理构建输出，默认保留已下载的工具链和依赖。`--purge` 额外清理工作区中的 `node_modules`。`--toolchains` 移除 `.build/toolchain` 下由 xtask 管理的工具链。

## 输出参数

大多数命令组支持以下通用输出参数：

- `--verbose`：直接流式输出子进程日志。
- `--no-banner`：禁用装饰性横幅。
- `--plain`：禁用控制台边框渲染。
