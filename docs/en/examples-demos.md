# Examples And Demos

Examples are small, runnable integrations. Demos are broader browser
experiences for inspecting runtime behavior and user-facing workflows.

## Examples

- `examples/rust`: Rust query, chat, embed, vision, gateway, and provider
  examples.
- `examples/node`: Node.js query, chat, embed, vision, and gateway examples.
- `examples/python`: Python query, chat, embed, vision, and gateway examples.
- `examples/web`: Vite browser pages for local and gateway workflows.
- `examples/gateway`: minimal Axum gateway route composition.

Start with:

```bash
cargo xtask run examples gateway rust --case query
cargo xtask run examples serve browser
```

## Demos

- `demos/chat`: focused browser chat interface for local GGUF models.
- `demos/avatar`: React and three.js character demo.
- `demos/proactive-ui`: drawing-to-vision demo with runtime tracing.
- `demos/simulation`: multi-agent simulation demo using director helpers.
- `tools/playground`: browser runtime diagnostics and automation tool.

Start with:

```bash
cargo xtask run demos serve chat
cargo xtask run tools serve playground
```

Use `cargo xtask test smoke group examples --backend cpu` for model-backed
example smoke coverage when validating broader runtime behavior.
