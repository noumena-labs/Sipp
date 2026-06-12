# Inference Operations

Sipp separates the operation from the endpoint. Choose `query`, `chat`, or
`embed` based on the input shape and expected output, then pass the endpoint
reference that decides where the request runs.

## Shared Contract

1. Register a local, gateway, or provider descriptor with `SippClient.add`.
2. Keep the returned endpoint reference.
3. Pass that reference to `query`, `chat`, or `embed`.

`query` and `chat` both produce text. They share `maxTokens`, `temperature`,
`topP`, `stop`, cancellation, and token streaming. `embed` produces vectors and
does not use generation options or token streaming.

| Operation | Input | Output | Best fit |
| --- | --- | --- | --- |
| `query` | One already-rendered prompt string. | Generated text. | Raw completions, custom templates, encoder-decoder text generation, few-shot prompts, and agent loops that render prompts themselves. |
| `chat` | Ordered `{ role, content }` messages. | Generated assistant text. | Conversation-shaped model calls where the endpoint owns the chat-template or provider-message mapping. |
| `embed` | One text input. | One embedding vector. | Retrieval, semantic search, ranking, clustering, and memory indexes. |

## Local Inference

Local endpoints run a GGUF model in the current browser, Node.js, Python, Rust,
or CLI process.

| Operation | What Sipp sends to the runtime | Template behavior | Local-only options |
| --- | --- | --- | --- |
| `query` | The prompt string exactly as supplied. Decoder-only models run the normal decode path; encoder-decoder models run an encoder pass and then the decoder loop. | No chat template is applied. Use this when the application owns a custom or generic prompt format. | Context keys, grammars, JSON schema, sampling overrides, media inputs. |
| `chat` | Messages are rendered to one prompt with llama.cpp chat-template support and `add_assistant = true`. | Requires the GGUF to declare `tokenizer.chat_template`. Sipp checks model metadata, not the llama.cpp fallback chain, before allowing local `chat`. | Same text options as `query`, including context keys and media inputs. |
| `embed` | The input text is encoded by the local embedding runtime. | No chat template and no generation. | Context key and embedding normalization. |

Local `chat` is a prompt renderer plus generation call, not a conversation
store. Pass prior turns in `messages` when they should be visible to the model.
Use a context key only for local KV-cache reuse.
Encoder-decoder text models, such as T5 or BART GGUF files, use `query` for
text generation. Encoder-only models do not generate text and should use
`embed` when they expose pooled embeddings.

### Local Query With A Custom Template

Use `query` when you want to own the full prompt shape, including a hand-written
or application-provided chat template.

```ts
const endpoint = await client.add('local', {
  kind: 'local',
  modelPath: '/models/model.gguf',
});

const prompt = [
  '<|system|>',
  'Answer with one concise paragraph.',
  '<|user|>',
  'Explain local query.',
  '<|assistant|>',
].join('\n');

const run = client.query({
  endpoint,
  prompt,
  options: { maxTokens: 128, temperature: 0.2 },
  local: { contextKey: 'docs-example' },
  emitTokens: true,
});
```

### Local Chat With The Model Template

Use `chat` when the GGUF model declares the chat template it expects. Sipp
passes the role messages to llama.cpp template rendering and then generates
from the rendered prompt.

```ts
const run = client.chat({
  endpoint,
  messages: [
    { role: 'system', content: 'Answer with one concise paragraph.' },
    { role: 'user', content: 'Explain local chat.' },
  ],
  options: { maxTokens: 128, temperature: 0.2 },
  local: { contextKey: 'docs-example' },
  emitTokens: true,
});
```

If the model has no `tokenizer.chat_template`, local `chat` fails. Use `query`
with an explicit prompt template for base models, legacy models, or any generic
template the application wants to control.

### Local Query With Encoder-Decoder Models

Use `query` for encoder-decoder GGUF models. The source prompt is encoded
first; Sipp then drives the decoder from the model's decoder-start token.

```ts
const endpoint = await client.add('t5-local', {
  kind: 'local',
  modelPath: '/models/t5-small-f16.gguf',
});

const run = client.query({
  endpoint,
  prompt: 'translate English to German: Hello, world.',
  options: { maxTokens: 64 },
});
```

Most encoder-decoder text models do not declare a GGUF chat template. In that
case `chat` is rejected even though `query` works.

### Local Embed

Use `embed` with a model/runtime that supports embeddings. Local embedding
normalization is a local-only option.

```ts
const run = client.embed({
  endpoint,
  input: 'Vectorize this sentence for retrieval.',
  local: { normalize: true },
});

const embedding = (await run.response).values;
```

## Remote Gateway

A gateway endpoint sends the operation over HTTP. The first-party profile uses
separate routes and payload shapes:

| Operation | Default route | Required body fields |
| --- | --- | --- |
| `query` | `/v1/query` | `model`, `prompt` |
| `chat` | `/v1/chat` | `model`, `messages` |
| `embed` | `/v1/embed` | `model`, `input` |

`model` is the public gateway target name. The gateway resolves that target to
a local GGUF endpoint, OpenAI endpoint, OpenAI-compatible endpoint, or
Anthropic endpoint.

Gateway calls accept shared text options for `query` and `chat`, such as
`max_tokens`, `temperature`, `top_p`, `stop`, and `stream`. Local-only fields
such as `contextKey`, `grammar`, `jsonSchema`, `sampling`, `media`, and
`normalize` are rejected by gateway endpoints. Direct-provider
`providerOptions` are also rejected by gateway endpoints; a custom gateway must
translate provider-specific extensions deliberately.

### Gateway Target Mapping

| Gateway target type | `query` behavior | `chat` behavior | `embed` behavior |
| --- | --- | --- | --- |
| Local GGUF | Runs local raw-prompt generation. Decoder-only models decode directly; encoder-decoder models run encoder prefill plus decoder generation. No chat template is added. | Runs local chat rendering with the GGUF-declared chat template. Fails if the model has no template, including many encoder-decoder models. | Runs local embedding if the loaded model/runtime supports embeddings. Encoder-decoder text models do not produce embeddings through this runtime. |
| OpenAI | Sends an OpenAI completions request with `prompt`. | Sends an OpenAI chat-completions request with `messages`. | Sends an OpenAI embeddings request with `input` and `encoding_format: "float"`. |
| OpenAI-compatible | Sends `/completions` with `prompt`. | Sends `/chat/completions` with `messages`. | Sends `/embeddings` with `input` and `encoding_format: "float"`. |
| Anthropic | Wraps the prompt as one user message and sends an Anthropic `/messages` request. | Sends Anthropic `/messages`; system role messages are joined into the top-level `system` field, and user/assistant messages remain in `messages`. | Unsupported by the native Anthropic adapter. |

Provider support still depends on the upstream model and provider. For example,
an OpenAI-compatible target may expose chat but not completions, so gateway
`chat` can work while gateway `query` fails for that target.

### Gateway Client Chat

```ts
const endpoint = await client.add('gateway-openai', {
  kind: 'gateway',
  target: 'openai-chat',
  baseUrl: process.env.SIPP_GATEWAY_URL!,
  authentication: {
    kind: 'bearer',
    value: process.env.SIPP_GATEWAY_TOKEN!,
  },
});

const run = client.chat({
  endpoint,
  messages: [
    { role: 'system', content: 'Answer for application developers.' },
    { role: 'user', content: 'When should I use gateway chat?' },
  ],
  options: { maxTokens: 128, temperature: 0.2 },
});
```

### First-Party Gateway HTTP Examples

Raw-prompt query:

```bash
curl -X POST "$SIPP_GATEWAY_URL/v1/query" \
  -H "Authorization: Bearer $SIPP_GATEWAY_TOKEN" \
  -H "content-type: application/json" \
  -d '{
    "model": "compatible-completion",
    "prompt": "Explain gateway query in one sentence.",
    "max_tokens": 64
  }'
```

Chat:

```bash
curl -X POST "$SIPP_GATEWAY_URL/v1/chat" \
  -H "Authorization: Bearer $SIPP_GATEWAY_TOKEN" \
  -H "content-type: application/json" \
  -d '{
    "model": "anthropic-chat",
    "messages": [
      { "role": "system", "content": "Answer briefly." },
      { "role": "user", "content": "Explain gateway chat." }
    ],
    "max_tokens": 128
  }'
```

Embedding:

```bash
curl -X POST "$SIPP_GATEWAY_URL/v1/embed" \
  -H "Authorization: Bearer $SIPP_GATEWAY_TOKEN" \
  -H "content-type: application/json" \
  -d '{
    "model": "openai-embed",
    "input": "Text to index for retrieval."
  }'
```

## Choosing Quickly

- Use local `query` when the application must control every token in the
  prompt, including custom or generic chat templates, or when the target is an
  encoder-decoder text model.
- Use local `chat` when the GGUF model declares its own chat template and the
  application already has role messages.
- Use local `embed` when vectors should be produced in the current process and
  local normalization matters.
- Use gateway `query` when the target supports raw-prompt generation, including
  local decoder-only or encoder-decoder GGUF targets and OpenAI-compatible
  completions targets.
- Use gateway `chat` for provider chat models and for local GGUF chat models
  with declared templates.
- Use gateway `embed` for local, OpenAI, or OpenAI-compatible embedding targets;
  do not use it with native Anthropic targets.

## Related Docs

- [Local Inference](local-inference.md)
- [Gateway And Hybrid Inference](gateway-hybrid.md)
- [Providers](providers.md)
- [Runtime Options](../reference/runtime-options.md)
