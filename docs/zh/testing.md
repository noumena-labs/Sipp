# 测试

Sipp 的测试套件通过 `cargo xtask test list` 统一管理。需要查找测试目标或查看 CI 运行任务时，优先使用此命令。

## 命令

`cargo xtask test` 提供四个核心操作：

- `list`：列出所有单元测试和冒烟测试套件，支持搜索和发现轻量级测试用例。
- `unit`：按套件（suite）或组（group）执行确定性的代码逻辑和 API 接口测试。
- `smoke`：按套件或组执行端到端的集成冒烟测试。
- `verify`：分析已有的覆盖率数据，验证测试目录结构。

## 常用命令

```bash
cargo xtask test list
cargo xtask test list --group unit --layer interface --cases --search router --format json
cargo xtask test unit group full
cargo xtask test unit group whitebox
cargo xtask test unit group interface
cargo xtask test unit suite xtask
cargo xtask test unit suite rust-crates --package sipp
cargo xtask test unit suite browser-package
cargo xtask test unit suite demos
cargo xtask test unit suite node-package --backend cpu
cargo xtask test unit suite python-package --backend cpu
cargo xtask test smoke suite example-node --backend cpu
cargo xtask test smoke suite example-gateway --backend cpu --case query
cargo xtask test smoke suite playground-browser
cargo xtask test smoke group examples --backend cpu
cargo xtask test smoke group local-model --backend cpu
cargo xtask test smoke group full --backend cpu
cargo xtask test verify --target whitebox
cargo xtask test verify --changed
```

`test unit` 执行确定性测试，命名空间分类如下：

- `test unit suite <name>` 只运行指定的单个确定性单元测试套件。
- `test unit group <name>` 运行预配置的一组确定性单元测试套件。

单元测试套件名还支持附加专用参数，如 `test unit suite rust-crates --package <crate>` 或 `test unit suite node-package --backend cpu`。

## 单元测试套件

| 命令 | 测试内容 | 代码路径 |
| --- | --- | --- |
| `cargo xtask test unit suite xtask` | xtask 工具及编排逻辑测试 | `xtask/src/tests` |
| `cargo xtask test unit suite rust-crates` | 工作区 Crate 的单元测试 | `crates`, `lib/gateway`, `apps` |
| `cargo xtask test unit suite rust-bindings` | Rust 绑定 Crate 的单元测试 | `bindings/node`, `bindings/python`, `bindings/wasm` |
| `cargo xtask test unit suite browser-package` | 浏览器包 TypeScript 测试 | `lib/web/tests` |
| `cargo xtask test unit suite demos` | 浏览器端演示项目的 TypeScript 测试 | `demos` |
| `cargo xtask test unit suite api` | 库级别公共 API 集成测试 | `crates/sipp/tests` |
| `cargo xtask test unit suite cli` | CLI 的黑盒集成测试 | `apps/cli/tests` |
| `cargo xtask test unit suite node-package` | Node 库的确定性 API 测试 | `lib/node`, `bindings/node` |
| `cargo xtask test unit suite python-package` | Python 库的确定性 API 测试 | `lib/python`, `bindings/python` |

## 单元测试组

| 命令 | 包含的套件 |
| --- | --- |
| `cargo xtask test unit group whitebox` | `xtask`、`rust-crates`、`rust-bindings`、`browser-package`、`demos` |
| `cargo xtask test unit group interface` | `api`、`cli`、`node-package`、`python-package` |
| `cargo xtask test unit group full` | 所有确定性单元测试套件 |

`test smoke` 用于端到端集成验证，同样有明确的命名空间分类：

- `test smoke suite <name>` 只运行指定的单个冒烟测试套件。
- `test smoke group <name>` 运行预配置的一组冒烟测试套件。

未指定 `--model` 时，基于模型的冒烟测试默认加载 `.build/models` 下的示例模型缓存。Rust、Node、Python、网关和浏览器的冒烟测试支持重复传入 `--case query|chat|embed` 参数以执行不同类型的请求。注意，embed（嵌入）用例要求模型或后端明确支持嵌入生成。

## 冒烟测试套件

| 命令 | 测试内容 | 代码路径 |
| --- | --- | --- |
| `cargo xtask test smoke suite cli` | CLI 本地推理生成流程 | `apps/cli` |
| `cargo xtask test smoke suite example-rust` | Rust 的 `query`/`chat`/`embed` 示例 | `examples/rust` |
| `cargo xtask test smoke suite example-node` | Node.js 的 `query.mjs`/`chat.mjs`/`embed.mjs` 示例 | `examples/node` |
| `cargo xtask test smoke suite example-python` | Python 的 `query.py`/`chat.py`/`embed.py` 示例 | `examples/python` |
| `cargo xtask test smoke suite example-gateway` | 嵌入式本地网关代理 + Rust/Node/Python 客户端示例 | `examples/gateway`, `examples/rust`, `examples/node`, `examples/python` |
| `cargo xtask test smoke suite example-browser` | Playwright 自动化执行浏览器示例（`query.html`/`chat.html`/`embed.html`） | `examples/web` |
| `cargo xtask test smoke suite playground-browser` | Playwright 自动化执行 Playground 测试 | `tools/playground` |
| `cargo xtask test smoke suite llama-backend-ops` | 验证 llama.cpp 后端操作正确性 | `crates/sys/llama.cpp` |

## 冒烟测试组

| 命令 | 包含的套件 |
| --- | --- |
| `cargo xtask test smoke group examples` | `example-rust`、`example-node`、`example-python`、`example-gateway`、`example-browser` |
| `cargo xtask test smoke group local-model` | `cli`、`example-rust`、`example-node`、`example-python` |
| `cargo xtask test smoke group full` | 所有冒烟测试套件（含 Playground、网关、llama 检查） |

手动启动浏览器端示例：`cargo xtask run examples serve browser`。

启动极简版本地网关代理：`cargo xtask run examples serve gateway-local --model <model.gguf>`。如需验证完整网关服务和生产环境配置，使用 `apps/gateway-server`。运行 `sipp run gateway-server check --config <path>` 校验网关配置。Docker 容器测试见[网关 Docker 指南](gateway/docker.md)，curl 和 Postman 的网络验证见[网关测试指南](gateway/testing.md)。Playground 环境由 `test smoke suite playground-browser` 负责验证。

`test unit` 和 `test smoke` 执行完毕后会输出总结报告，保存到 `.build/test/run-report.json` 和 `.build/test/run-report.md`。支持覆盖率的单元测试套件还会将覆盖率数据写入 `.build/coverage/`。

`test verify` 不运行测试代码，只负责校验测试结构、验证目录归属、检查测试代码与运行时代码是否分离、分析变更文件的覆盖情况，以及读取已有的覆盖率产物。

## 包路径说明

- `lib/web` 构建并发布 `@noumena-labs/sipp` 及公开浏览器包 `@sipp/sipp`。
- `lib/node` 构建并发布 `@noumena-labs/sipp-server` 及公开 Node 服务端包 `@sipp/sipp-server`。
- `lib/python` 构建并发布 Python 分发包 `sipppy`，导入包名仍为 `sipp`。
- `crates/sipp` 发布 Rust 包 `sipp-rs`，库 crate 名仍为 `sipp`。
