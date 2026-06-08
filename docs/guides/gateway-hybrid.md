# Gateway And Hybrid Inference

Gateway inference lets an application call a separate CogentLM gateway over
HTTP. Hybrid inference registers local and gateway endpoints in the same client
so each request can choose where it runs.

## When To Use A Gateway

- Keep provider credentials out of browser or edge clients.
- Centralize target access policy and concurrency limits.
- Serve local models from a controlled machine.
- Expose a stable HTTP boundary to multiple language clients.

## Endpoint Model

The client does not route implicitly. Every application registers descriptors
and selects an endpoint reference:

- Local descriptor: a GGUF model loaded by the current runtime.
- Gateway descriptor: a base URL, target name, routes, and authentication.
- Provider descriptor: direct provider adapter where the package supports it.

Gateway descriptors send the target as the first-party profile `model` field.
The gateway process resolves that public target name to a local or provider
endpoint.

## Authentication

Server and script environments use bearer values from environment variables.
Browser applications use short-lived tokens supplied at runtime through a
provider callback.

## Related Docs

- [Gateway Server](../packages/gateway-server.md)
- [Gateway Architecture](../gateway.md)
- [Gateway Server Reference](../reference/gateway-server.md)
- [Gateway Toolkit](../packages/gateway.md)
