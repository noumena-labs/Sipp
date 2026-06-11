# 网关故障排除

官方网关在启动、请求处理或响应阶段出现异常时参考此页。

## `check` 成功但 `serve` 失败

`check` 命令的作用仅限于解析并验证 TOML 配置文件。它不会尝试读取 Token 环境变量、加载模型文件、建立服务商连接或绑定网络端口。

`check` 通过但 `serve` 启动失败时排查以下方面：

- `[[tokens]].env` 指定的环境变量是否已正确设置且值不为空。
- `admin_password_env` 指定的环境变量是否已正确设置且值不为空。
- 外部服务商所需密钥环境变量（如 `OPENAI_API_KEY`）是否已设置。
- TOML 中本地 GGUF 模型路径是否真实存在，当前进程是否有读取权限。
- `public_bind` 和 `management_bind` 端口是否可用，未被其他进程占用。
- 指定 GPU 推理后端是否已编译支持，并在当前主机上可用。

## 缺少 DLL 或共享库

直接运行编译出的可执行文件时，必须将 `.build/artifacts/gateway-server` 加入系统动态库加载路径。生成的网关可执行文件依赖该目录下的运行时库和 GGML 后端插件。

- Windows：将目录路径追加到 `PATH` 环境变量前端。
- Linux：将目录路径追加到 `LD_LIBRARY_PATH` 环境变量前端。
- macOS：将目录路径追加到 `DYLD_LIBRARY_PATH` 环境变量前端。

使用 `clm run gateway-server ...` 工作流启动时，工具会自动处理这些路径。

## 相对模型路径解析错误

本地目标 TOML 配置中的相对 `model` 路径基于进程当前工作目录解析。`clm run gateway-server ...` 默认将工作区根目录设为工作目录；在终端中直接运行可执行文件时，工作目录为当前 shell 所在目录。

跨目录运行可执行文件时改用绝对路径。Docker 环境下路径必须是容器内的绝对路径，而非主机的本地路径。

## Docker 端口已发布，但主机无法连接

Docker 模式下，`public_bind` 和 `management_bind` 配置的是容器内部监听地址。请务必使用：

```toml
public_bind = "0.0.0.0:8080"
management_bind = "0.0.0.0:9090"
```

随后，依靠 Compose 的 `ports` 配置控制这些端口如何映射至主机。在本地开发用的 Compose 模板中，这两个端口默认均映射至 `127.0.0.1`，仅限本地回环访问。

## `401 Unauthorized` 错误

该错误表明公共路由未接收到合法 Bearer Token。检查：

- 请求头格式是否为 `Authorization: Bearer <token>`。
- Token 值是否与 TOML 中 `[[tokens]]` 块绑定的环境变量值完全一致。
- Token 字符串中是否混入多余空格或换行。
- 修改 Token 环境变量后是否已重启网关进程。

## `403 Forbidden` 错误

该错误表明 Bearer Token 有效，但受限于 `targets` 允许列表，无法访问请求的 `model` 推理目标。在 TOML 中将目标名称追加到对应的 `[[tokens]]` 允许列表，或在请求中使用有访问权限的 Token。

## `404 Target Not Found` 错误

请求中的 `model` 值无法匹配任何已配置的 `[[targets]].name`。公共 HTTP 请求的 `model` 字段是网关向外公开的目标标识符，不一定与外部服务商使用的模型名或本地 GGUF 文件名一致。

## 浏览器跨域 (CORS) 失败

浏览器发起的请求必须得到公共监听器 CORS 授权。将完整匹配的源地址添加到 `allowed_origins`：

```toml
allowed_origins = ["http://localhost:5173"]
```

若 `allowed_origins` 数组为空，网关将禁用跨域支持，拦截所有预检请求。

## GPU 后端运行失败

指定的本地目标后端在编译时未包含，或运行时环境不支持时，网关启动失败。推荐 `backend = "auto"`，网关会自动回退到最优可用后端。指定 `cpu` 会禁用 GPU offload，仅用于排查本地推理问题。

Docker 中使用 GPU 后端必须配置相应主机环境：

- CUDA 需要 NVIDIA 主机驱动和 Container Toolkit 支持。
- Vulkan 需要挂载主机 GPU 设备节点，并提供 Vulkan 加载器和驱动栈。
- Metal 仅限 macOS 本地原生环境，无法穿透到 Linux Docker 容器。

Docker 日志提示 `ggml_vulkan: No devices found` 表示已加载 Vulkan 后端但容器内无法枚举到物理设备。Windows Docker Desktop 的 NVIDIA GPU 用户请改用 `cuda` 配置。

## 管理面板登录失败

管理面板的登录密码从 TOML 中 `admin_password_env` 指定的环境变量提取。检查环境变量配置文件或机密管理器中是否已正确设定该值，并确认启动时 `--config` 参数指向了正确的 TOML 文件。

管理面板仅允许通过管理端监听器（默认 9090 端口）访问。
