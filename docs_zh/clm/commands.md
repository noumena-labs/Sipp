# 常用命令

`clm` 将代码库的自动化任务划分为若干命令族。运行 `clm <group> --help` 可查看自动生成的帮助信息和当前支持的选项。

## 健康检查

```bash
clm doctor
clm doctor --target wasm
clm doctor --target node --backend vulkan
clm toolchain status
```

`doctor` 在不安装或删除任何内容的前提下检查本地环境是否就绪。
`toolchain status` 报告由 xtask 管理的工具（Bun、Python、uv、Emscripten、Ninja）状态。CUDA 为系统级安装，xtask 仅报告状态，不负责安装或卸载。

## 构建

```bash
clm build core
clm build wasm
clm build node --backend cpu
clm build python --backend vulkan
clm build cli --backend all
clm build gateway-server --backend cpu
clm build all
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
clm run examples serve browser --port 5173
clm run examples serve gateway-local --model .build/models/model.gguf --bind 127.0.0.1:8787
clm run examples gateway rust --case query
clm run demos serve chat
clm run tools serve playground
clm run gateway-server check --config apps/gateway-server/config/local.toml
clm run gateway-server serve --config apps/gateway-server/config/local.toml --backend cpu
```

`run` 用于执行当前有的程序、网关服务、服务器示例以及非测试诊断。执行测试请用 `clm test`。

## 文档

```bash
clm docs build
clm docs serve
clm docs build --lang zh
```

`docs build` 在缺失时自动安装 `mdbook` 和 `mdbook-mermaid`，将 Mermaid JavaScript 资源提取到 `theme/`，生成的静态页面写入 `book/`。默认构建还会在 `book/zh` 目录下生成中文版文档。`docs serve` 执行相同的准备工作，构建相同的目录树，并在 `localhost:3000` 上启动服务。页面顶部提供 `/` 和 `/zh/` 之间的语言切换开关。使用 `--lang zh` 可以只构建中文版本，或优先显示中文预览地址。

## 测试

```bash
clm test list
clm test list --group unit --layer interface --cases --search router --format json
clm test unit group full
clm test unit suite rust-crates --package cogentlm-engine
clm test unit suite node-package --backend cpu
clm test unit suite browser-package
clm test smoke suite example-node --backend cpu
clm test smoke group local-model --backend cpu
clm test verify --changed
clm test verify --target public-docs
```

执行需要模型的冒烟测试时，省略 `--model` 参数会默认使用 setup 阶段下载到 `.build/models` 的缓存模型。完整的测试套件目录见[测试](../testing.md)。

## 清理

```bash
clm clean --dry-run
clm clean
clm clean --purge
clm clean --toolchains
```

`clean` 清理构建输出，默认保留已下载的工具链和依赖。`--purge` 额外清理工作区中的 `node_modules`。`--toolchains` 移除 `.build/toolchain` 下由 xtask 管理的工具链。

## 输出参数

大多数命令组支持以下通用输出参数：

- `--verbose`：直接流式输出子进程日志。
- `--no-banner`：禁用装饰性横幅。
- `--plain`：禁用控制台边框渲染。
