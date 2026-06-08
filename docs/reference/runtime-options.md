# Runtime Options

CogentLM keeps runtime configuration close to the endpoint that owns local
inference. Request options stay on `query`, `chat`, or `embed` calls. Gateway
and provider extensions use separate option buckets so applications can see
which boundary receives each field.

## Option Layers

| Layer | Browser package | Node.js package | Purpose |
| --- | --- | --- | --- |
| Client options | `new CogentClient(options)` | Environment and process setup | Browser assets, workers, browser cache, and backend selection. |
| Local endpoint load options | `client.add(..., { kind: 'local', options })` | `client.add(..., { kind: 'local', config })` | Model source, backend preference, progress, and native runtime config. |
| Text request options | `client.query(prompt, options)` | `client.query({ options })` | Output length, sampling shortcuts, streaming, cancellation, and stop strings. |
| Local request options | `session`, `grammar`, media, `normalize` | `local: { contextKey, grammar, media, normalize }` | Local-only prompt state, grammars, images, and embedding normalization. |
| Gateway extensions | `endpointOptions` | `endpointOptions` | Extra fields consumed by gateway endpoint implementations. |
| Provider extensions | `providerOptions` | `providerOptions` | Provider-only fields merged into direct provider requests. |

Python and Rust expose the same concepts with language-native descriptors and
runtime config classes or structs.

## Browser Client Options

Browser `CogentClientOptions` affect the WebAssembly runtime, worker transport,
and browser storage. They do not select a model by themselves.

| Option | Use |
| --- | --- |
| `executionMode` | `auto` uses a worker when available. `worker` forces worker transport. `main-thread` is useful for debugging or constrained hosts. |
| `wasmThreading` | `single-thread` loads the single-thread WASM runtime. `pthread` loads the pthread runtime. |
| `moduleUrl`, `wasmUrl` | Override single-thread runtime asset URLs when a bundler or deployment moves package assets. Provide both together. |
| `pthreadModuleUrl`, `pthreadWasmUrl` | Override pthread runtime asset URLs. Provide both together. |
| `browserCache` | Tune OPFS split thresholds and direct-load behavior for browser GGUF storage. |
| `trustedOrigins` | Allow runtime asset URLs from additional origins. Defaults allow same-origin package assets. |
| `workerUrl` | Override the worker entry URL when the bundler cannot resolve the packaged worker. |

`wasmThreading: 'pthread'` requires `SharedArrayBuffer`, cross-origin
isolation, and COOP/COEP headers. Use `single-thread` when the application
cannot serve those headers.

```ts
const client = new CogentClient({
  executionMode: 'worker',
  wasmThreading: 'single-thread',
});
```

## Local Endpoint Options

Browser local endpoints use `source` plus optional load options:

```ts
const endpoint = await client.add('browser-local', {
  kind: 'local',
  source: '/models/model.gguf',
  options: {
    backend: 'webgpu',
    runtime: {
      context: { n_ctx: 2048 },
    },
  },
});
```

Node.js local endpoints use `modelPath` and `config`:

```ts
const endpoint = await client.add('node-local', {
  kind: 'local',
  modelPath: '/models/model.gguf',
  config: {
    context: { n_ctx: 2048, n_threads: 8, n_threads_batch: 8 },
  },
});
```

Browser `backend` accepts `auto`, `cpu`, or `webgpu`. Native package backend
selection is package-specific: Node.js uses `COGENTLM_NODE_BACKEND`, Python
uses `COGENTLM_PYTHON_BACKEND`, and the CLI uses `--backend`.

## Native Runtime Config

`NativeRuntimeConfig` groups local runtime settings by responsibility.

| Group | Common fields | Use |
| --- | --- | --- |
| `placement` | `devices`, `gpu_layers`, `split_mode`, `main_gpu`, `tensor_split`, `use_mmap`, `use_mlock`, `fit_params` | Model placement, memory mapping, and GPU residency choices. |
| `context` | `n_ctx`, `n_batch`, `n_ubatch`, `n_parallel`, `n_threads`, `n_threads_batch`, `flash_attention`, `offload_kqv` | Context window, batch sizes, CPU thread counts, attention, and KV behavior. |
| `sampling` | `samplers`, `seed`, `top_k`, `top_p`, `min_p`, `temperature`, `repeat_penalty`, `mirostat`, `logit_bias` | Default local sampling behavior for text generation. |
| `scheduler` | `continuous_batching`, `policy`, `prefill_chunk_size`, `max_running_requests`, `max_queued_requests` | Request scheduling, batching, and queue limits. |
| `cache` | `mode`, `retained_prefix_tokens`, `snapshot_interval_tokens`, `max_snapshot_entries`, `max_snapshot_bytes` | Prefix KV reuse and snapshot behavior. |
| `multimodal` | `projector_path`, `use_gpu`, `image_min_tokens`, `image_max_tokens` | Vision projector and image-token settings. |
| `residency` | `max_gpu_models_per_device`, `allow_cpu_models_while_gpu_loaded`, `require_gpu_lease` | GPU model residency policy for native runtimes. |
| `observability` | `runtime_metrics`, `backend_profiling` | Runtime timing, throughput, and backend diagnostics. |

Use runtime config for stable endpoint behavior. Use request options for values
that should vary per prompt, user action, or UI control.

## Request Options

Text-producing calls share common generation controls:

| Option | Use |
| --- | --- |
| `maxTokens` | Maximum generated tokens for the response. |
| `temperature` | Request-local temperature shortcut. |
| `topP` | Request-local nucleus sampling shortcut. |
| `stop` | Stop strings for text generation. |
| `signal` | Cancellation through `AbortSignal` where supported. |
| `emitTokens` | Enables token streaming through the returned run handle. |

Local text calls can also use a prompt context key or session key, GBNF grammar,
and media inputs for vision-capable models. Embedding calls can set
normalization through local embedding options.

Gateway-specific fields belong in `endpointOptions`. Direct provider-specific
fields belong in `providerOptions`:

```ts
const run = client.chat({
  endpoint,
  messages,
  options: { maxTokens: 128, temperature: 0.2 },
  providerOptions: {
    reasoning_effort: 'low',
  },
});
```

Provider options cannot override typed fields such as `model`, `messages`,
`prompt`, `temperature`, or `topP`/`top_p`; set those through the typed request
options where CogentLM exposes them.

## Related Docs

- [Local Inference](../guides/local-inference.md)
- [Providers](../guides/providers.md)
- [Browser Caching](../guides/browser-caching.md)
- [Gateway And Hybrid Inference](../guides/gateway-hybrid.md)
