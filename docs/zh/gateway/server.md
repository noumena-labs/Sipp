# 网关服务器

Sipp 网关服务器是官方提供的 HTTP 应用，为本地 GGUF 目标和外部服务商后端目标提供统一的推理边界。代码位于 `apps/gateway-server`。

本文介绍如何通过检出源码运行以及运行编译后的二进制文件。容器工作流见 [Docker](docker.md)，TOML 配置说明见[配置](configuration.md)。

当前版本暂不发布独立二进制文件、公开容器镜像或 `cargo install` 目标。请通过检出源码构建。

## 本地工作流

工作流中使用 `sipp` 命令。`sipp` 是已配置好的 `cargo xtask` 启动器。无法使用 `sipp` 时，可用等效参数运行 `cargo xtask`。

```bash
cp apps/gateway-server/config/local.toml.example apps/gateway-server/config/local.toml
cp apps/gateway-server/.env.example apps/gateway-server/.env
set -a
. apps/gateway-server/.env
set +a
sipp run gateway-server check --config apps/gateway-server/config/local.toml --backend vulkan
sipp run gateway-server serve --config apps/gateway-server/config/local.toml --backend vulkan
```

运行本地推理测试前，在本地 TOML 配置文件中更新 Token 环境变量名、管理员密码环境变量名及模型路径（该文件已被 Git 忽略）。敏感密钥只需在 `.env` 中更新。

`sipp run gateway-server check` 为选定后端构建网关发布版，然后运行 `sipp-gateway check`。该命令仅验证并解析 TOML 配置，不读取 Bearer Token、不加载模型、不连接服务商、不绑定端口。

`sipp run gateway-server serve` 构建网关发布版，在工作区根目录运行生成的 `sipp-gateway`。它读取 TOML 中指定的密钥，加载目标，绑定监听器，按下 Ctrl-C 时优雅退出。

用 `--backend cpu|vulkan|cuda|metal|all` 选择编译进网关发布版的后端。

## 纯服务商模式工作流

纯服务商模式网关将请求路由到上游 API，不加载本地 GGUF 模型。推理在外部服务商执行，使用 CPU 后端构建即可：

```bash
cp apps/gateway-server/config/provider-only.toml.example apps/gateway-server/config/provider-only.toml
cp apps/gateway-server/.env.example apps/gateway-server/.env
set -a
. apps/gateway-server/.env
set +a
sipp run gateway-server check --config apps/gateway-server/config/provider-only.toml --backend cpu
sipp run gateway-server serve --config apps/gateway-server/config/provider-only.toml --backend cpu
```

Anthropic 和 OpenAI 兼容的目标配置示例见[配置](configuration.md)。

## 运行二进制文件

`sipp build gateway-server --backend <backend>` 将可运行的发布版部署到 `.build/artifacts/gateway-server` 目录。该目录包含 `sipp-gateway` 可执行文件、基础运行时库及指定 GGML 后端插件。此外还会编译 `apps/gateway-server/admin-ui` 下的 React 管理面板，并将其 Vite 输出复制到 `.build/artifacts/gateway-server/admin-ui`。请确保可执行文件、管理面板资源和运行时库在同一目录下。

直接运行二进制时，必须将工件目录加入动态链接库加载路径。默认情况下可执行文件读取同级的 `admin-ui` 目录，除非通过 `SIPP_GATEWAY_ADMIN_ASSETS_DIR` 指定了其他 Vite `dist` 目录。

Linux：

```bash
set -a
. apps/gateway-server/.env
set +a
export LD_LIBRARY_PATH="$(pwd)/.build/artifacts/gateway-server${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.build/artifacts/gateway-server/sipp-gateway check --config apps/gateway-server/config/local.toml
.build/artifacts/gateway-server/sipp-gateway serve --config apps/gateway-server/config/local.toml
```

macOS：

```bash
set -a
. apps/gateway-server/.env
set +a
export DYLD_LIBRARY_PATH="$(pwd)/.build/artifacts/gateway-server${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
.build/artifacts/gateway-server/sipp-gateway check --config apps/gateway-server/config/local.toml
.build/artifacts/gateway-server/sipp-gateway serve --config apps/gateway-server/config/local.toml
```

Windows PowerShell：

```powershell
Get-Content apps\gateway-server\.env | ForEach-Object {
    if ($_ -and -not $_.StartsWith("#")) {
        $name, $value = $_.Split("=", 2)
        Set-Item -Path "Env:$name" -Value $value
    }
}
$dist = Join-Path (Get-Location) ".build\artifacts\gateway-server"
$env:PATH = "$dist;$env:PATH"
.\.build\artifacts\gateway-server\sipp-gateway.exe check --config apps\gateway-server\config\local.toml
.\.build\artifacts\gateway-server\sipp-gateway.exe serve --config apps\gateway-server\config\local.toml
```

TOML 中的相对 `model` 路径基于进程工作目录解析。`sipp run gateway-server ...` 默认在工作区根目录运行。从其他目录运行可执行文件时，请使用绝对路径或从工作区根目录启动。

## 推理后端

网关服务器支持与其他原生目标一致的后端选项：

- `cpu`：纯服务商模式构建或本地推理诊断。
- `cuda`：NVIDIA CUDA 后端。
- `metal`：macOS Apple Metal 后端。
- `vulkan`：Vulkan 后端。
- `all`：构建主机支持的所有后端。

配置本地目标时，`backend = "auto"` 会按顺序选择可用且已编译的最佳后端：CUDA、Metal、Vulkan、CPU。生产环境使用 `auto` 或明确指定 GPU 后端。指定 `cpu` 会禁用 GPU 卸载，仅用于诊断。显式指定的 GPU 后端未编译或不受支持时，进程启动失败。

## 管理面板

管理面板密码从 TOML 指定的环境变量读取：

```toml
admin_password_env = "SIPP_GATEWAY_ADMIN_PASSWORD"
```

真实密码保存在环境密钥文件或生产环境的机密管理器。

## 相关文档

- [Docker](docker.md)
- [配置](configuration.md)
- [测试](testing.md)
- [运维](operations.md)
