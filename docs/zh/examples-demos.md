# 示例与演示

示例（Examples）是精简可运行的集成代码。演示（Demos）是完整的浏览器端应用，用于测试运行时行为和用户交互流程。

## 示例

- `examples/rust`：Rust 的 query、chat、embed、视觉、网关及服务商示例。
- `examples/node`：Node.js 的 query、chat、embed、视觉及网关示例。
- `examples/python`：Python 的 query、chat、embed、视觉及网关示例。
- `examples/web`：用于验证本地和网关工作流的 Vite 浏览器页面。
- `examples/gateway`：极简的 Axum 网关路由配置。

启动示例：

```bash
cargo xtask run examples gateway rust --case query
cargo xtask run examples serve browser
```

## 演示

- `demos/chat`：面向本地 GGUF 模型的极简浏览器聊天界面。
- `demos/avatar`：基于 React 和 three.js 的数字人演示。
- `demos/proactive-ui`：带运行时跟踪（tracing）的画板视觉演示。
- `demos/simulation`：利用 director 工具构建的多智能体模拟场景。
- `tools/playground`：浏览器运行时的诊断和自动化测试工具。

启动演示：

```bash
cargo xtask run demos serve chat
cargo xtask run tools serve playground
```

如需验证各环境下的运行时表现，运行 `cargo xtask test smoke group examples --backend cpu` 执行全面的示例冒烟测试。
