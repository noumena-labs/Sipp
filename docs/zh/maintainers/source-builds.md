# 源码构建

如需开发 Sipp 自身功能、验证打包产物、运行源码示例，或在官方发布独立网关服务器二进制文件前自行部署网关，请直接使用源码版本。

## 引导环境

在仓库根目录下运行以下命令初始化环境：

```bash
source ./setup.sh
sipp doctor
sipp test list
```

Windows 上在 PowerShell 中执行 `.\setup.ps1`，或在 CMD 中执行 `setup.cmd`。初始化完成后，`sipp` 别名指向仓库本地的 `cargo xtask`。别名未生效时，直接用相同参数执行 `cargo xtask ...`。

## 构建目标产物

编译 Sipp 时务必使用 `xtask` 编排工具，不要直接执行底层构建命令。该工具自动管理原生依赖库、后端工具链并完成打包暂存。

```bash
sipp build core
sipp build node --backend cpu
sipp build python --backend cpu
sipp build gateway-server --backend cpu
sipp build wasm
sipp build all
```

支持硬件加速的原生目标可使用 `--backend vulkan`、`--backend cuda`、`--backend metal` 或 `--backend all` 等参数。

默认情况下，CUDA 会编译一系列相关 GPU 架构的产物。为加快本地编译速度，可在构建前设置 `SIPP_CUDA_ARCHITECTURES` 环境变量（使用分号分隔的 CMake 架构字符串，例如仅针对 A100 设置 `80`）。完整架构列表见[网关 Docker](../gateway/docker.md)。

## 示例与演示

使用 `sipp` 命令运行基于浏览器的示例和演示应用。这些命令启动 Vite 开发服务器，不接收原生后端参数：

```bash
sipp run examples serve browser
sipp run demos serve avatar
sipp run demos serve simulation
```

## 网关 Hello World 示例

网关示例工作流自动启动本地网关服务，运行客户端代码示例，客户端执行完毕后关闭网关。内部流程为先启动 `examples/gateway`，再拉起 `examples/rust`、`examples/node` 或 `examples/python` 中的客户端应用。

通过 `--case query|chat|embed` 选择客户端要执行的具体用例。网关进程需使用特定原生引擎后端时，附加 `--backend cpu|vulkan|cuda|metal`。

```bash
sipp run examples gateway rust --case query
sipp run examples gateway node --case chat
sipp run examples gateway python --case embed --backend vulkan
```

## Playground

浏览器 Playground 位于 `tools/playground` 目录。该工具可用于验证本地推理功能、调试视觉模型配置、测试 GGUF 加载过程、查看运行时可观测性指标，以及执行可重复的浏览器运行时冒烟测试。

```bash
sipp run tools serve playground
```

## 网关服务器

当前发布工作流尚未提供预编译的 `gateway-server` 二进制文件或容器镜像。请使用 `sipp` 基于源码验证，并使用标准 Docker 命令完成容器化部署。权威指南见[网关服务器](../gateway/server.md)；Docker 部署细节见[网关 Docker](../gateway/docker.md)。

```bash
cp apps/gateway-server/config/local.toml.example apps/gateway-server/config/local.toml
cp apps/gateway-server/.env.example apps/gateway-server/.env
set -a
. apps/gateway-server/.env
set +a
sipp run gateway-server check --config apps/gateway-server/config/local.toml --backend cpu
sipp run gateway-server serve --config apps/gateway-server/config/local.toml --backend cpu
```

复制的本地配置文件要求在 `.build/models` 目录下存放一个本地 GGUF 模型，并根据 TOML 配置好管理面板的密码环境变量。妥善保管 `.env` 文件，其中包含管理员密码和提供商 API 密钥等敏感信息。

## 验证

在[测试](../testing.md)中寻找范围最窄的目标命令进行验证。常用入口命令：

```bash
sipp test list
sipp test unit group full
sipp test smoke group examples --backend cpu
sipp test verify --target public-docs
```
