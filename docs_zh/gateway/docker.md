# 网关 Docker

网关的 Docker 工作流使用 Compose 文件、网关 TOML 配置文件以及仅含密钥的 `.env` 文件。

配置职责严格划分：

- `.env` 文件只存放密钥。
- TOML 文件负责网关应用配置。
- Compose YAML 文件负责 Docker 构建、镜像、端口、挂载、健康检查和容器编排。

容器内启动命令：

```bash
cogentlm-gateway serve --config /etc/cogentlm/gateway.toml
```

## 相关文件

- `apps/gateway-server/Dockerfile`：构建网关发布版。
- `apps/gateway-server/.env.example`：仅包含密钥的环境变量模板。
- `apps/gateway-server/development.yml.example`：构建并运行含本地模型挂载的开发镜像。
- `apps/gateway-server/development-provider-only.yml.example`：构建并运行无模型挂载的纯服务商开发镜像。
- `apps/gateway-server/production.yml.example`：运行预构建的生产级本地模型镜像。
- `apps/gateway-server/production-provider-only.yml.example`：运行预构建的纯服务商生产镜像。
- `apps/gateway-server/config/*.toml.example`：网关应用的配置模板。

## 本地模型 Docker 运行

在项目根目录下执行：

```bash
cp apps/gateway-server/.env.example apps/gateway-server/.env
cp apps/gateway-server/development.yml.example apps/gateway-server/development.yml
cp apps/gateway-server/config/development.toml.example apps/gateway-server/config/development.toml
```

编辑 `apps/gateway-server/.env`，只填入密钥：

```bash
COGENTLM_GATEWAY_ADMIN_PASSWORD=replace-me
COGENTLM_GATEWAY_TOKEN=replace-me
OPENAI_API_KEY=replace-me
ANTHROPIC_API_KEY=replace-me
```

编辑 `apps/gateway-server/config/development.toml`：

- 将本地目标的 `model` 改为容器内路径，通常为 `/models/<file>.gguf`。
- 保持 `public_bind = "0.0.0.0:8080"` 和 `management_bind = "0.0.0.0:9090"`，确保网关在容器内监听。
- 除非在 `.env` 中修改了密钥名称，否则保留 `admin_password_env = "COGENTLM_GATEWAY_ADMIN_PASSWORD"`。

编辑 `apps/gateway-server/development.yml` 适配镜像标签、构建后端、模型挂载、端口映射和健康检查。

构建并使用对应的后端配置运行。CPU 模式可在 Windows、macOS 和 Linux 之间移植。GPU 容器需要主机显卡驱动支持。

> [!WARNING]
> Windows Docker Desktop 不支持官方 Vulkan 网关路径。
> 配备 NVIDIA GPU 的 Windows 主机必须使用 `cuda` 配置。请勿使用废弃的 `vulkan-windows` 配置；如遇到 `ggml_vulkan: No devices found` 错误，说明容器未能枚举到可用 Vulkan 设备。

```bash
# CPU 模式，跨平台兼容
docker compose --profile cpu --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml config
docker compose --profile cpu --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml build gateway-cpu
docker compose --profile cpu --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml up gateway-cpu

# CUDA 模式，适用于 Linux 或 Windows Docker Desktop
docker compose --profile cuda --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml config
docker compose --profile cuda --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml build gateway-cuda
docker compose --profile cuda --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml up gateway-cuda

# 原生 Linux Vulkan 模式，需使用 /dev/dri
docker compose --profile vulkan-linux --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml config
docker compose --profile vulkan-linux --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml build gateway-vulkan-linux
docker compose --profile vulkan-linux --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml up gateway-vulkan-linux
```

切换服务后 Compose 提示存在孤立容器时清理：

```bash
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development.yml down --remove-orphans
```

## 纯服务商模式 Docker

纯服务商模式不挂载模型：

```bash
cp apps/gateway-server/.env.example apps/gateway-server/.env
cp apps/gateway-server/development-provider-only.yml.example apps/gateway-server/development-provider-only.yml
cp apps/gateway-server/config/provider-only.toml.example apps/gateway-server/config/provider-only.toml
```

在 `apps/gateway-server/.env` 中设置密钥并运行：

```bash
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development-provider-only.yml config
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development-provider-only.yml build
docker compose --env-file apps/gateway-server/.env -f apps/gateway-server/development-provider-only.yml up
```

纯服务商模板会构建 CPU 网关镜像，因为推理在上游执行。

## 生产环境 Docker

将生产环境的 TOML、Compose 和 `.env` 文件独立存放在项目仓库之外：

```bash
mkdir -p /opt/cogentlm/gateway
cp apps/gateway-server/.env.example /opt/cogentlm/gateway/.env
cp apps/gateway-server/production.yml.example /opt/cogentlm/gateway/production.yml
cp apps/gateway-server/config/production.toml.example /opt/cogentlm/gateway/production.toml
```

编辑 `/opt/cogentlm/gateway/.env` 填写敏感值，`production.toml` 配置网关，`production.yml` 配置镜像、挂载、端口映射、重启策略和健康检查。

根据需求选择一种后端部署：

```bash
# CPU 模式
docker compose --profile cpu --env-file /opt/cogentlm/gateway/.env -f /opt/cogentlm/gateway/production.yml config
docker compose --profile cpu --env-file /opt/cogentlm/gateway/.env -f /opt/cogentlm/gateway/production.yml up -d gateway-cpu

# CUDA 模式，需要 NVIDIA Container Toolkit
docker compose --profile cuda --env-file /opt/cogentlm/gateway/.env -f /opt/cogentlm/gateway/production.yml config
docker compose --profile cuda --env-file /opt/cogentlm/gateway/.env -f /opt/cogentlm/gateway/production.yml up -d gateway-cuda

# Vulkan 模式，需 /dev/dri 支持
docker compose --profile vulkan-linux --env-file /opt/cogentlm/gateway/.env -f /opt/cogentlm/gateway/production.yml config
docker compose --profile vulkan-linux --env-file /opt/cogentlm/gateway/.env -f /opt/cogentlm/gateway/production.yml up -d gateway-vulkan-linux
```

部署纯服务商模式时，复制并使用 `production-provider-only.yml.example` 和 `config/provider-only.toml.example`。

## 绑定与挂载机制

网关总是使用相同的 TOML 架构，但绑定地址与路径的解析方式会因环境而异。

| 运行环境 | TOML 绑定值 | 主机曝光行为 | 本地目标 `model` 路径 |
| --- | --- | --- | --- |
| 源码/可执行文件 | 主机地址，通常为 `127.0.0.1:*` | 进程直接在主机上监听 | 基于进程工作目录的相对/绝对路径 |
| 本地 Compose | 容器地址，通常为 `0.0.0.0:8080` 和 `0.0.0.0:9090` | 本地模板的 `ports` 将主机端口映射至 `127.0.0.1` | 容器挂载路径，如 `/models/<file>.gguf` |
| 生产级 Compose | 容器地址，通常为 `0.0.0.0:8080` 和 `0.0.0.0:9090` | 生产模板公开公共端口，管理端口仅限本地访问 | 容器挂载路径，如 `/models/<file>.gguf` |
| 纯服务商 Compose | 容器地址，通常为 `0.0.0.0:8080` 和 `0.0.0.0:9090` | 遵循一致的端口映射规则 | 无本地模型路径 |

生产环境中务必将管理接口保持为私有。在公共监听器前，按需添加公共入口、TLS 以及外部身份验证控制。

## 原生 Docker 构建

可使用原生 Docker 命令构建，需显式提供所有参数：

```bash
docker build \
  --build-arg COGENTLM_GATEWAY_BACKEND=vulkan \
  --build-arg COGENTLM_GATEWAY_BUILDER_IMAGE=rust:bookworm \
  --build-arg COGENTLM_GATEWAY_RUNTIME_IMAGE=ubuntu:22.04 \
  --build-arg COGENTLM_GATEWAY_INSTALL_RUSTUP=0 \
  -f apps/gateway-server/Dockerfile \
  -t cogentlm-gateway:vulkan .
```

## 后端硬件与 Docker 限制

构建的网关镜像带有后端标签：`latest-cpu`、`latest-cuda`、`latest-vulkan`。

支持的官方 Docker 配置文件：

| 主机环境 | GPU 厂商 | 配置文件 | 后端 | 备注 |
| --- | --- | --- | --- | --- |
| Linux Docker | NVIDIA | `cuda` | CUDA | 推荐的 NVIDIA 方案。需驱动及容器支持。 |
| Linux Docker | AMD / Intel | `vulkan-linux` | Vulkan | 需 `/dev/dri` 挂载及 Vulkan 驱动栈。 |
| Linux Docker | 无 GPU | `cpu` | CPU | 跨平台的诊断与回退方案。 |
| Windows Docker Desktop | NVIDIA | `cuda` | CUDA | 需开启 WSL2 GPU 支持及直通。 |
| Windows Docker Desktop | AMD / Intel | `cpu` | CPU | 官方 Docker 暂不支持 Windows Vulkan 推理。 |
| macOS Docker | 任意 | `cpu` | CPU | Metal 需在 macOS 本地运行，Linux 容器不支持。 |

### CPU 后端（`latest-cpu` / `cpu` 配置）
- 标准的可移植环境，无需特殊驱动依赖。
- 适用于 macOS 本地 Docker 开发。

### CUDA 后端（`latest-cuda` / `cuda` 配置）
- 主机必须配置 **NVIDIA Container Toolkit** 与 NVIDIA 驱动。
- 通过 Compose 的 GPU 预留功能进行分配。
- 兼容配备 NVIDIA GPU 的 Linux 和 Windows Docker Desktop 主机。

### CUDA 架构选择

通过 `COGENTLM_CUDA_ARCHITECTURES` 控制 GPU 编译架构。该值直接传给 CMake，用分号分隔。Docker 镜像通过构建参数接受此值。

架构默认策略：

- `cargo xtask build` 默认选用以下兼容云端 GPU 的列表，保证不同主机的包一致性。Docker 网关通过 xtask 构建，因此参数为空时继承该默认值。
- 如果绕过 xtask 原生构建 `cogentlm-sys`，CMake 不会设定默认架构，llama.cpp 会尝试自动匹配本地机器的 CUDA 架构。

可移植的云端 GPU 默认包含：

```text
75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real;121a-real
```

| 标识 | 目标 GPU |
| --- | --- |
| `75-virtual` | T4 等 Turing 架构云端显卡 |
| `80-virtual` | A100 等 Ampere 架构显卡 |
| `86-real` | A10, A40, RTX A6000 级 Ampere 显卡 |
| `89-real` | L4, L40S, Ada 架构显卡 |
| `90-virtual` | H100, H200 Hopper 架构显卡 |
| `120a-real` | 专用 Blackwell 架构 |
| `121a-real` | 较新的专用 Blackwell 架构 |

缩减该列表可显著加快构建速度。例如仅填写 `80`（A100）或 `89`（L40S）。

CUDA 13 不再支持计算能力低于 7.5 的架构。若需支持 `61`（Pascal）或 `70`（Volta），请使用 CUDA 12.x 镜像，并显式指定架构列表。

Blackwell 目标（带 `a` 后缀）为专有架构。仅在需要 TensorRT 时才使用 TensorRT 镜像，默认使用纯 CUDA。

### Vulkan 后端（`latest-vulkan` 镜像）
- 官方仅在 Linux 环境提供支持：`vulkan-linux`。
- 必须通过 `/dev/dri:/dev/dri` 将渲染设备挂载至容器。
- Windows Docker Desktop 不支持。NVIDIA 用户应使用 `cuda` 配置。
- 容器已预装 `libvulkan1` 与 `mesa-vulkan-drivers`。

### Apple Metal 后端限制
> [!WARNING]
> **Metal 无法在 Linux Docker 容器中运行。**
> macOS Docker 运行在 Linux 虚拟机中，Apple 并不支持将 Metal API 穿透至 Linux。
>
> 影响如下：
> 1. **Docker 限制**：在 macOS 上运行 Docker 网关只能回退至纯 CPU 计算，无法获得 Metal 加速。
> 2. **原生方案**：macOS 用户如需 Metal 加速，必须在本地编译并直接运行服务器：
>    ```bash
>    cargo xtask build gateway-server --backend metal
>    ./.build/artifacts/gateway-server/cogentlm-gateway serve --config apps/gateway-server/config/development.toml
>    ```

## 健康检查

Compose 模板通过以下命令探测管理端就绪路由：

```bash
curl --fail --silent http://127.0.0.1:9090/readyz
```

如果在 TOML 中更改了就绪路由地址，请同步更新 Compose 探针。
