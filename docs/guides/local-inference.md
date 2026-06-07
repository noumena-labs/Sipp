# Local Inference

Local inference runs a GGUF model in the current process or browser runtime.
Applications register a local endpoint, keep the returned endpoint reference,
and pass it to `query`, `chat`, or `embed`.

## Endpoint Flow

1. Build or load the package for the desired runtime.
2. Choose a GGUF model that supports the requested capability.
3. Register the model with `CogentClient.add`.
4. Pass the returned endpoint reference on each request.
5. Close the client when the runtime is no longer needed.

## Capability Notes

- Query and chat require text generation support.
- Embed requires a model/runtime that reports embedding support.
- Vision chat requires a text/vision model plus projector data where the model
  family requires it.
- Local-only request options, such as context keys and media inputs, belong on
  local endpoints.

Use examples for copyable integrations and demos for broader runtime behavior.
