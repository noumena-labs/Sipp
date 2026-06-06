# CogentLM Gateway Server

`apps/gateway-server` is the full-featured Axum gateway implementation. It is
the production-style server for running a CogentLM Remote Gateway from a TOML
config:

```bash
cargo run -p cogentlm-gateway-server -- serve --config <gateway.toml>
```

It builds on the framework-neutral `crates/gateway` adapter and adds HTTP
server concerns:

- bearer auth for inference requests
- admin auth for protected status and history APIs
- CORS
- `/healthz` and `/readyz` probes
- `/v1/query`, `/v1/chat`, and `/v1/embed` routes
- a root dashboard page
- bounded in-memory request history

For the smallest learning example, use `examples/gateway`. That example is
self-contained and shows how to wire `crates/gateway` into an Axum proxy without
the dashboard, admin APIs, or request history.
