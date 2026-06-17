# Gateway Route Example

This Axum example shows the canonical gateway composition pattern:

- Create a `SippClient`.
- Register endpoints with `client.add(...)`.
- Define Axum routes in application code.
- Decode request bodies with `GatewayCodec`.
- Select the endpoint and call `client.query()`, `client.chat()`, or
  `client.embed()`.
- Encode JSON or SSE responses explicitly.

The example exposes `/v1/query`, `/v1/chat`, and `/v1/embed`. Sipp does
not own those routes; they are ordinary application handlers.

## Run

```bash
cargo xtask run examples serve gateway-local \
  --model .build/models/model.gguf \
  --bind 127.0.0.1:8787
```

Use `apps/gateway-server` for the first-party authenticated application, or
compose `lib/gateway` helpers in your own framework routes.

See [../../docs/en/gateway/architecture.md](../../docs/en/gateway/architecture.md)
for gateway layering.
