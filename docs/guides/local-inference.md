# Local Inference

Local inference runs a GGUF model inside the current browser, Node.js, Python,
Rust, or CLI process. The application owns model selection, runtime lifecycle,
resource cleanup, and the request options that should be exposed to users.

Register a local endpoint with `SippClient.add`, keep the returned endpoint
reference, and pass that reference to `query`, `chat`, or `embed`.

## Endpoint Flow

1. Choose a GGUF model that supports the requested capability.
2. Register the model with a local descriptor.
3. Set load-time runtime options on the endpoint descriptor.
4. Pass request-time generation options to `query`, `chat`, or `embed`.
5. Stream tokens or await the final response.
6. Close the client when the page, worker, service, or script no longer needs
   the runtime.

Local endpoints do not route implicitly. A client can register multiple
endpoints, but every request that should use a specific destination should pass
the endpoint reference returned by `add`.

## Model Sources

Browser local endpoints can load:

- A model URL served by the application.
- A user-selected `File`.
- Multiple shard URLs or files.
- An installed model id returned by browser model-management APIs.
- A model plus projector pair for vision-capable models.

Node.js, Python, Rust, and CLI local endpoints use filesystem paths. Source
examples and smoke workflows can use cached sample models under `.build/models`
when running from a checkout.

## Runtime And Request Options

Keep option layers separate:

- Browser client options such as `executionMode`, `wasmThreading`, runtime
  asset URLs, and `browserCache` belong on `new SippClient(...)`.
- Local endpoint load options choose the model source, browser backend
  preference, progress callbacks, and `NativeRuntimeConfig`.
- Runtime config groups such as `context`, `sampling`, `scheduler`, `cache`,
  `placement`, `multimodal`, `residency`, and `observability` describe stable
  local endpoint behavior.
- Request options such as `maxTokens`, `temperature`, `topP`, `stop`,
  cancellation, and `emitTokens` belong on `query`, `chat`, or `embed`.
- Local-only request options such as context keys, grammars, media inputs, and
  embedding normalization should not be sent to gateway or provider endpoints.

See [Runtime Options](../reference/runtime-options.md) for the canonical option
map and field groups.

## Threads And Browser Execution

Browser execution has two separate choices:

- `executionMode: 'worker'` or `auto` keeps inference work off the UI thread
  when workers are available.
- `wasmThreading: 'pthread'` enables the pthread WASM runtime and requires
  `SharedArrayBuffer` plus cross-origin isolation headers.

Use `wasmThreading: 'single-thread'` when the app cannot serve COOP/COEP
headers. Use `executionMode: 'main-thread'` mainly for debugging or constrained
hosts.

Native Node.js, Python, and Rust local endpoints can tune CPU thread counts
with `context.n_threads` and `context.n_threads_batch`. Leave them unset for
runtime defaults unless the application has measured a better value.

## Text, Embeddings, And Vision

- Query and chat require text generation support.
- Embed requires a model/runtime that reports embedding support.
- Vision chat requires a text/vision model plus projector data where the model
  family requires it.
- Streaming text requires `emitTokens` and consuming the returned token
  iterable before or alongside the final response.
- GBNF grammars and media inputs are local-only request features.

## Related Docs

- [Runtime Options](../reference/runtime-options.md)
- [Browser Package](../packages/browser.md)
- [Node.js Package](../packages/node.md)
- [Python Package](../packages/python.md)
- [Rust Package](../packages/rust.md)
- [Browser Caching](browser-caching.md)
