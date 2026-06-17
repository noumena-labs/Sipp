# 后端兼容矩阵

Sipp 的本地推理构建在 llama.cpp 和 ggml 之上。Sipp 负责客户端 API、端点模型、调度、各语言包的绑定、浏览器的生命周期管理以及网关集成；llama.cpp 和 ggml 负责提供 GGUF 运行时的底层后端计算内核。

因此后端支持情况分为两层：

- Sipp 层：各语言包支持哪些后端，以及如何编译或选择这些后端。
- ggml 层：特定 ggml 后端实现了哪些张量运算。

ggml 运算级别的兼容矩阵参考上游的 [llama.cpp GGML 运算表](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)。

## Sipp 后端名称

| 后端 | 设备类别 | 暴露位置 | 备注 |
| --- | --- | --- | --- |
| `cpu` | 主机 CPU | 浏览器、Node.js、Python、Rust/源码、CLI、网关服务 | 移植性最好的默认后端。原生构建用 ggml CPU；浏览器构建用 WASM CPU。 |
| `webgpu` | 浏览器 GPU | Browser 包 | 仅限浏览器，通过 `options.backend` 激活。需要浏览器及其底层系统支持 WebGPU。 |
| `cuda` | NVIDIA GPU | 原生构建、Node.js、Python、CLI、网关服务 | 需本地安装 CUDA Toolkit 和对应 NVIDIA 驱动。xtask 检测状态但不安装 CUDA。 |
| `metal` | Apple GPU | macOS 原生构建、Node.js、Python、CLI、网关服务 | 仅限 macOS 原生环境。适合 Apple Silicon 和已验证的 AMD Mac；Intel 集成显卡建议使用 CPU。 |
| `vulkan` | 兼容 Vulkan 的 GPU | 原生构建、Node.js、Python、CLI、网关服务 | 需系统和驱动支持。xtask 会在编译时按需准备 Vulkan SDK。 |

上游 llama.cpp/ggml 还支持 BLAS、CANN、OpenCL、SYCL 等后端（可在运算表中看到），但目前 Sipp 官方暴露的后端仅为上述五种。

## 包与运行时选择

| 环境 | 支持的后端 | 选择方式 |
| --- | --- | --- |
| 浏览器 | `auto`, `cpu`, `webgpu` | `client.add(..., { kind: 'local', options: { backend: 'webgpu' } })` |
| Node.js | `cpu`, `vulkan`, `cuda`, `metal` | `SIPP_NODE_BACKEND=cpu|vulkan|cuda|metal` |
| Python | `cpu`, `vulkan`, `cuda`, `metal` | `SIPP_PYTHON_BACKEND=cpu|vulkan|cuda|metal` |
| CLI | `auto`, `cpu`, `cuda`, `metal`, `vulkan` | `sipp ... --backend <backend>` |
| 网关服务 | `auto`, `cpu`, `cuda`, `metal`, `vulkan` | 通过 `sipp ... --backend <backend>` 编译或运行；目标配置 TOML 可以设定 `backend = "auto"` 或特定名称。 |
| Rust / 源码工作流 | 编译后的产物集合 | 通过 `sipp` 或 `cargo xtask` 编译；运行时可用后端取决于链接的原生产物。 |

注意：`auto` 是运行时策略。`all` 是 `sipp` 和 `xtask` 在构建/测试时使用的参数，表示编译当前主机支持的所有后端，不是运行时的后端名称。

## 后端混合使用

编译产物和引擎运行时的选择是分离的：

- **编译产物**决定哪些 GPU 后端打包进二进制、可供进程加载。例如，仅含 CUDA 的构建无法运行 Vulkan，仅含 Metal 的构建也无法运行 CUDA。
- **`cpu`** 是特例。明确指定引擎使用 `cpu` 时，Sipp 会禁用 GPU 计算、显存分配、KV 缓存卸载、运算卸载、Flash Attention 和 GPU 驻留管理。
- **GPU 引擎**（`cuda`、`metal`、`vulkan`、`webgpu`）必须同时满足两个条件：编译时已包含进产物，且运行时环境可用。
- Node.js 和 Python 在进程启动时通过环境变量（`SIPP_NODE_BACKEND` 或 `SIPP_PYTHON_BACKEND`）加载特定原生绑定。同一进程内只能使用一种原生后端。如需不同后端，需启动独立进程或加载不同产物。
- 网关、CLI、浏览器和更底层的 Rust API 允许在目标/加载阶段传入后端选择器，但只能从已编译且当前主机可用的后端中挑选。

匹配示例：

| 当前的编译产物/进程 | CPU 引擎 | CUDA 引擎 | Metal 引擎 | Vulkan 引擎 |
| --- | --- | --- | --- | --- |
| 仅 CUDA 的产物 | 支持（如果在暴露 CPU 选项的平台上） | 支持（如果设备可用） | 不支持 | 不支持 |
| 仅 Metal 的产物 | 支持（如果在暴露 CPU 选项的平台上） | 不支持 | 支持（在 macOS 上） | 不支持 |
| 仅 Vulkan 的产物 | 支持（如果在暴露 CPU 选项的平台上） | 不支持 | 不支持 | 支持（如果设备可用） |
| 多后端的产物 | 支持 | 支持（若编译并可用） | 支持（若编译并可用） | 支持（若编译并可用） |

CLI 示例：

```bash
# 编译包含 CUDA 的 CLI
sipp build cli --backend cuda

# 如果环境支持，使用 CUDA 推理
sipp ./models/model.gguf "Explain this model." --chat --backend cuda

# 强制使用 CPU，禁用所有 GPU 加速
sipp ./models/model.gguf "Explain this model." --chat --backend cpu

# 该命令将失败，因为之前的编译产物中没有 Vulkan
sipp ./models/model.gguf "Explain this model." --chat --backend vulkan
```

网关配置示例：

```toml
# 同一个网关进程挂载不同后端的本地模型。
# 前提：编译出的网关必须同时包含 CUDA 和 CPU 后端支持。
[[targets]]
name = "local-cuda"
type = "local"
model = "./models/model.gguf"
backend = "cuda"

[[targets]]
name = "local-cpu"
type = "local"
model = "./models/model.gguf"
backend = "cpu"
```

浏览器示例：

```ts
// 浏览器可以通过选项分别指定 WebGPU 和 CPU 终端
await client.add('local-webgpu', {
  kind: 'local',
  model: './models/model.gguf',
  options: { backend: 'webgpu' },
});

await client.add('local-cpu', {
  kind: 'local',
  model: './models/model.gguf',
  options: { backend: 'cpu' },
});
```

Node.js 和 Python 示例：

```powershell
# PowerShell：在启动进程前选择原生绑定。
$env:SIPP_NODE_BACKEND = "cuda"
node .\examples\node\chat.mjs .\models\model.gguf "Explain this model."

$env:SIPP_NODE_BACKEND = "cpu"
node .\examples\node\chat.mjs .\models\model.gguf "Explain this model."
```

```bash
# Bash：在启动进程前选择原生绑定。
SIPP_PYTHON_BACKEND=cuda \
  python examples/python/chat.py ./models/model.gguf "Explain this model."

SIPP_PYTHON_BACKEND=cpu \
  python examples/python/chat.py ./models/model.gguf "Explain this model."
```

## 构建矩阵

| 构建命令 | 后端参数 | 产出物 |
| --- | --- | --- |
| `sipp build wasm` | 无 | 包含 CPU 与 WebGPU 运行时的浏览器 WASM 包。 |
| `sipp build node --backend cpu` | `cpu`, `cuda`, `metal`, `vulkan`, `all` | 针对选定后端编译的 Node 原生绑定。 |
| `sipp build python --backend cpu` | `cpu`, `cuda`, `metal`, `vulkan`, `all` | 针对选定后端编译的 Python 原生绑定。 |
| `sipp build cli --backend cpu` | `cpu`, `cuda`, `metal`, `vulkan`, `all` | 针对选定后端编译的 `sipp` CLI 工具。 |
| `sipp build gateway-server --backend cpu` | `cpu`, `cuda`, `metal`, `vulkan`, `all` | 针对选定后端编译的网关服务程序。 |
| `sipp build all` | 无 | Core, WASM, Python CPU, Node CPU 以及 CLI CPU 目标。 |

`sipp build all` 仅生成基础组合。如果需要 CUDA、Metal 或 Vulkan 支持，必须显式传入对应的 `--backend` 参数。

## 运算支持情况

并非所有 ggml 后端都实现了完整的运算集。Sipp 提供的主流后端已覆盖了常用的 Transformer 架构，但不同模型所需的特殊运算可能会受限于你所选的后端。

诊断后端问题时，请遵循以下步骤：

- 如果模型在 `cpu` 上正常工作，但在 GPU 上报错，请检查上游 ggml 运算表中是否缺少该 GPU 后端的对应运算。
- 若 GPU 缺失某项运算，llama.cpp/ggml 可能会降级到 CPU 执行、保留张量在内存中，或直接由于计算图策略抛出错误。
- 如果运行时无法识别某个后端，请确保二进制产物在编译时包含了该后端，并且对应的设备驱动对当前进程可见。
- 浏览器中的 `webgpu` 既依赖于编译时的支持，也依赖于浏览器中实际适配器的可用性。可使用 `backend: 'cpu'` 强制降级。

源码环境下的本地验证方法：

```bash
sipp doctor --target node --backend vulkan
sipp run llama backend-ops --backend vulkan --mode support
sipp run llama backend-ops --backend cuda --mode perf --op MUL_MAT
```

`llama backend-ops` 会为指定的后端构建 llama.cpp 的测试工具，非常适合在 Sipp 客户端之外排查运算覆盖率和性能瓶颈。

## 实际选择建议

在测试模型兼容性或排查正确性问题时，建议先使用 `cpu` 后端。验证模型文件、提示词格式和配置均无误后，再切换至 GPU。

如果需要在现代浏览器中加速，请优先使用 `webgpu`。同时，为不支持 WebGPU 的浏览器或设备保留 CPU 回退逻辑。

针对原生部署：NVIDIA 硬件环境使用 `cuda`；Apple Silicon 或已验证的 AMD
macOS 环境使用 `metal`。Intel 集成显卡的 Mac 建议使用 `cpu`，除非你已经
为目标模型测过稳定的 Metal 路径。如果需要兼容多厂商的 GPU，并在目标驱动栈
上验证通过，则使用 `vulkan`。
