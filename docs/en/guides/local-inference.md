# Local Inference

Local inference runs a GGUF model inside the current browser, Node.js, Python,
Rust, or CLI process. The application chooses the model and request options;
`SippClient` owns the model store, endpoint registry, and runtime resources.

Register a local endpoint with `SippClient.add`, keep the returned endpoint
reference, and pass that reference to `query`, `chat`, or `embed`.

## Endpoint Flow

1. Choose a GGUF model that supports the requested capability.
2. Install its files or URLs through `client.models`.
3. Create a local endpoint descriptor from the returned model ID.
4. Set load-time runtime options on the endpoint descriptor.
5. Pass request-time options to `query`, `chat`, or `embed`.
6. Stream tokens or await the final response.
7. Close the client when the page, worker, service, or script no longer needs
   the runtime.

Pass the endpoint reference returned by `add` whenever routing must be
explicit. Omit it only when the client can select one compatible local
endpoint unambiguously.

## Model Installation

All packages install models before creating local endpoints:

- Browser `installFiles` and `installUrls` persist models in OPFS.
- Node.js, Python, and Rust install filesystem paths or HTTP(S) URLs under the
  client's storage root.
- Both forms accept multiple shards and an optional projector for vision
  models.

Installation returns a stable model ID. `EndpointDescriptor.local(model.id)`
or `LocalDescriptor::new(model.id)` loads that installed model; it does not
download or copy files.

Source examples and smoke workflows can use cached sample models under
`.build/models` when running from a checkout.

## Runtime And Request Options

Keep option layers separate:

- Browser client options such as `executionMode`, `wasmThreading`, runtime
  asset URLs, and `browserCache` belong on `new SippClient(...)`.
- Install method arguments choose files or URLs; install options provide an
  optional projector, progress callback, and cancellation where supported.
- Local endpoint options select the installed model ID, browser backend
  preference, and `NativeRuntimeConfig`.
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

The bundled browser runtime requires COOP/COEP headers. Apps that cannot serve
those headers must set `wasmThreading: 'single-thread'` and provide custom
single-thread `moduleUrl` and `wasmUrl` assets. Use
`executionMode: 'main-thread'` mainly for debugging or constrained hosts.

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
