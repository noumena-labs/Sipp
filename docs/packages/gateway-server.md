# Gateway Server

The CogentLM Gateway Server is the first-party HTTP gateway application for
teams that want a central inference boundary. It can expose local GGUF targets
or provider-backed targets while keeping model paths, provider credentials,
target access policy, concurrency limits, and operational metrics inside the
gateway process.

The gateway server is a user-facing deployment surface, but the current release
workflow does not publish a standalone binary, public container image, or
`cargo install` target. Build and deploy it from the source checkout until a
public server artifact is added.

## When To Use It

- Browser applications need inference without exposing provider credentials.
- Multiple clients should share one local model host or provider routing
  policy.
- Operators need a stable HTTP boundary with health, readiness, metrics, CORS,
  body limits, and concurrency admission.
- Applications want the same `query`, `chat`, and `embed` client API while
  moving endpoint ownership to a server process.

## Run From Source

```bash
export COGENTLM_GATEWAY_TOKEN="replace-me"
cargo run -p cogentlm-gateway-server -- \
  check --config apps/gateway-server/config/production.toml
cargo run -p cogentlm-gateway-server -- \
  serve --config apps/gateway-server/config/production.toml
```

`check` validates TOML without reading secrets or loading endpoints. `serve`
loads targets, reads token environment variables, binds the public and
management listeners, and shuts down gracefully on Ctrl-C.

The checked-in Dockerfile and compose file are source deployment helpers:

```bash
docker build -f apps/gateway-server/Dockerfile -t cogentlm-gateway:cpu .
docker compose -f apps/gateway-server/compose.yaml up
```

Treat `cogentlm-gateway:cpu` as a local image name from this build command, not
as a published image.

## Configuration

The production example config uses separate public and management listeners:

```toml
public_bind = "0.0.0.0:8080"
management_bind = "0.0.0.0:9090"
max_request_bytes = 1048576
max_concurrent_requests = 4
allowed_origins = []

[routes]
query = "/v1/query"
chat = "/v1/chat"
embed = "/v1/embed"
index = "/"
health = "/healthz"
readiness = "/readyz"
metrics = "/metrics"

[[tokens]]
env = "COGENTLM_GATEWAY_TOKEN"
caller = "production-client"
targets = ["local"]

[[targets]]
name = "local"
type = "local"
model = "/models/model.gguf"
```

The public listener serves `query`, `chat`, and `embed`. The management
listener serves optional `index`, `health`, `readiness`, and `metrics` routes.
Keep the management listener private in production.

Tokens are read from environment variables. Each token gets a stable caller
label and an allowlist of public target names. Targets can be local GGUF,
OpenAI, OpenAI-compatible, or Anthropic endpoints. Provider credentials stay in
the gateway environment and are never needed by browser or client applications.

## Client Shape

Gateway clients register a gateway endpoint, keep the returned endpoint
reference, and call the normal client methods. The client needs only:

- `target`: the public target name from gateway config.
- `baseUrl`: the public gateway URL.
- `authentication`: the bearer or header value issued by the application.
- Optional route overrides when the gateway config changes from the defaults.

### Browser

```ts
import { CogentClient } from 'cogentlm';

const client = new CogentClient();
const endpoint = await client.add('gateway', {
  kind: 'gateway',
  target: 'local',
  baseUrl: 'https://gateway.example.com',
  authentication: {
    kind: 'bearer',
    valueProvider: getShortLivedGatewayToken,
  },
});

const run = client.query('Explain gateway inference.', {
  endpoint,
  emitTokens: true,
  maxTokens: 64,
});

for await (const batch of run.tokens) {
  console.log(batch.text);
}
console.log((await run.response).text);
await client.close();
```

### Node.js

```ts
import { CogentClient } from 'cogentlm-server';

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

const client = new CogentClient();
const endpoint = await client.add('gateway', {
  kind: 'gateway',
  target: requiredEnv('COGENTLM_GATEWAY_TARGET'),
  baseUrl: requiredEnv('COGENTLM_GATEWAY_URL'),
  authentication: {
    kind: 'bearer',
    value: requiredEnv('COGENTLM_GATEWAY_TOKEN'),
  },
});

const run = client.query({
  endpoint,
  prompt: 'Explain gateway inference.',
  options: { maxTokens: 64 },
  emitTokens: true,
});
console.log((await run.response).text);
```

### Python

```python
import os

from cogentlm import CogentClient, CogentTextOptions, GatewayDescriptor


client = CogentClient()
endpoint = client.add(
    "gateway",
    GatewayDescriptor(
        "local",
        os.environ["COGENTLM_GATEWAY_URL"],
        authentication_kind="bearer",
        authentication_value=os.environ["COGENTLM_GATEWAY_TOKEN"],
    ),
)
run = client.query(
    "Explain gateway inference.",
    endpoint=endpoint,
    options=CogentTextOptions(max_tokens=64),
)
print(run.result()["text"])
```

### Rust

```rust
use cogentlm::{
    CogentClient, CogentQueryRequest, CogentTextOptions, EndpointDescriptor,
    GatewayAuthentication, GatewayEndpointConfig, GatewayRoutes, GatewaySecret,
    GatewayTimeoutPolicy,
};

let mut client = CogentClient::new();
let endpoint = client
    .add(
        "gateway",
        EndpointDescriptor::gateway(GatewayEndpointConfig {
            target: "local".to_string(),
            base_url: std::env::var("COGENTLM_GATEWAY_URL")?,
            routes: GatewayRoutes::default(),
            authentication: GatewayAuthentication::Bearer(GatewaySecret::new(
                std::env::var("COGENTLM_GATEWAY_TOKEN")?,
            )),
            static_headers: Default::default(),
            timeouts: GatewayTimeoutPolicy::default(),
            protocol_options: Default::default(),
        }),
    )
    .await?;

let response = client
    .query(CogentQueryRequest {
        endpoint: Some(endpoint),
        prompt: "Explain gateway inference.".to_string(),
        options: CogentTextOptions {
            max_tokens: Some(64),
            ..Default::default()
        },
        ..Default::default()
    })
    .await?;
println!("{}", response.text);
```

## Operations

- Health and readiness run on the management listener.
- Metrics use Prometheus text exposition with per-operation request and error
  counters.
- CORS applies to the public listener when `allowed_origins` is non-empty.
- `max_request_bytes` bounds incoming request bodies.
- `max_concurrent_requests` applies application-wide admission control.
- JSON tracing is enabled through the gateway process and can be filtered with
  `RUST_LOG`.

## Related Docs

- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
- [Gateway Server Reference](../reference/gateway-server.md)
- [Configuration](../reference/configuration.md)
- [Source Builds](../maintainers/source-builds.md)
