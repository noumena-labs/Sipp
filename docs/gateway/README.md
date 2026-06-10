# Gateway

CogentLM gateway workflows put one HTTP boundary in front of local GGUF
targets and provider-backed targets. Applications still use the same client
model: register an endpoint with `CogentClient.add`, keep the returned
endpoint reference, and choose that reference for `query`, `chat`, or `embed`.

Use a gateway when you want a separate process to own model paths, provider
credentials, target access policy, concurrency limits, metrics, and operational
routes.

## Notices

> [!WARNING]
> The gateway server is in active development. Changes will be made frequently, and things will break. 
> If you use it for production, be cautious and watch for release updates. You can join [our Discord](https://discord.gg/abzgfghhrq) server and follow up on development.


## What To Use

| Need | Start here |
| --- | --- |
| Run the first-party server from a checkout | [Server](server.md) |
| Build and run the Docker image | [Docker](docker.md) |
| Understand the TOML file | [Configuration](configuration.md) |
| Test with curl, Postman, or raw HTTP | [Testing](testing.md) |
| Operate health, metrics, admin, and ingress | [Operations](operations.md) |
| Build your own gateway application | [Toolkit](toolkit.md) |
| Understand package boundaries | [Architecture](architecture.md) |
| Debug common failures | [Troubleshooting](troubleshooting.md) |

The current release workflow publishes browser npm, Node npm, Python wheel,
and Rust source artifacts. It does not yet publish a standalone gateway-server
binary, public container image, or `cargo install` target. Build the
first-party server from the source checkout or with the provided Dockerfile.

## Gateway Shapes

- **First-party server**: `apps/gateway-server` provides TOML configuration,
  bearer-token policy, local and provider targets, management routes, metrics,
  and an Admin Dashboard.
- **Docker image**: `apps/gateway-server/Dockerfile` builds the same staged
  gateway distribution and runs
  `cogentlm-gateway serve --config /etc/cogentlm/gateway.toml`.
- **Gateway toolkit**: `lib/gateway` provides codecs, HTTP error helpers,
  authentication traits, observability traits, and the first-party JSON/SSE
  profile for custom applications.
- **Gateway clients**: Browser, Node, Python, and Rust packages all register
  gateway endpoints through the same `.add` path used for local and provider
  endpoints.

## Deployment Shapes

- **On-board GPU inference**: configure a local GGUF target, build or run the
  gateway with `vulkan`, `cuda`, or `metal`, and mount or point at the model
  path the process can read.
- **Provider-only router**: configure only provider targets such as `openai`,
  `openai_compatible`, or `anthropic`. No local model path or `/models` mount
  is required, and a CPU gateway image is sufficient because inference runs at
  the provider.
- **Hybrid**: configure both a local GPU target and provider targets. Clients
  still send the public gateway target name in the request `model` field.

## Default Routes

The first-party server examples use:

- Public: `/v1/query`, `/v1/chat`, `/v1/embed`.
- Management: `/`, `/healthz`, `/readyz`, `/metrics`, `/admin`.

Those paths are application configuration, not core library behavior. Custom
gateway applications can choose their own routes.
