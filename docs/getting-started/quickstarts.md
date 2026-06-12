# Quickstarts

These snippets show the public call shapes for `query`, `chat`, and `embed`.
`query` sends the exact prompt string and never applies a chat template. A
plain prompt is only for completion-style/base models; for decoder-only chat or
instruct GGUFs, render the model's template yourself. Local `query` also supports encoder-decoder GGUF text models. `chat` sends role-tagged
messages. `embed` returns vectors and needs an embedding-capable local model
loaded with embedding mode enabled.

Local context naming differs only by language casing: browser and Node.js use
`contextKey`; Python and Rust use `context_key`.

See [Examples And Demos](../examples-demos.md) for runnable end-to-end files.

## Browser Local

```bash
npm install sipp
```

```ts
import { SippClient, type ChatMessage } from 'sipp';

const client = new SippClient();
const messages: readonly ChatMessage[] = [
  { role: 'system', content: 'Answer concisely.' },
  { role: 'user', content: 'Explain local browser inference.' },
];
const queryPrompt = [
  '<|system|>',
  'Answer concisely.',
  '<|user|>',
  'Explain local browser inference.',
  '<|assistant|>',
].join('\n');

const textEndpoint = await client.add('text', {
  kind: 'local',
  source: '/models/chat.gguf',
  options: { backend: 'webgpu', runtime: { context: { n_ctx: 2048 } } },
});

// query: raw prompt; replace markers with the target model's template.
const query = await client.query(queryPrompt, {
  endpoint: textEndpoint,
  maxTokens: 64,
  contextKey: 'browser-query',
}).response;

// chat: role messages; local runtime uses tokenizer.chat_template.
const chat = await client.chat(messages, {
  endpoint: textEndpoint,
  maxTokens: 64,
  contextKey: 'browser-chat',
}).response;

const embedEndpoint = await client.add('embed', {
  kind: 'local',
  source: '/models/embed.gguf',
  options: {
    backend: 'webgpu',
    runtime: { context: { n_ctx: 2048, embeddings: true, pooling: 'mean' } },
  },
});

// embed: vector output; local endpoint must be embedding-capable.
const embedding = await client.embed('Sipp embedding input.', {
  endpoint: embedEndpoint,
  contextKey: 'browser-embed',
  normalize: true,
}).response;

console.log(query.text, chat.text, embedding.values.length);
await client.close();
```

## Node.js Local

```bash
npm install sipp-server
```

```ts
import { SippClient } from 'sipp-server';

const client = new SippClient();
const messages = [
  { role: 'system', content: 'Answer concisely.' },
  { role: 'user', content: 'Explain local Node.js inference.' },
];
const queryPrompt = [
  '<|system|>',
  'Answer concisely.',
  '<|user|>',
  'Explain local Node.js inference.',
  '<|assistant|>',
].join('\n');
const textOptions = { maxTokens: 64 };
const textModel = process.argv[2] ?? 'chat.gguf';
const embedModel = process.argv[3] ?? 'embed.gguf';

const textEndpoint = await client.add('text', {
  kind: 'local',
  modelPath: textModel,
  config: { context: { n_ctx: 2048 } },
});

// query: raw prompt; replace markers with the target model's template.
const query = await client.query({
  endpoint: textEndpoint,
  prompt: queryPrompt,
  options: textOptions,
  local: { contextKey: 'node-query' },
}).response;

// chat: role messages; local runtime uses tokenizer.chat_template.
const chat = await client.chat({
  endpoint: textEndpoint,
  messages,
  options: textOptions,
  local: { contextKey: 'node-chat' },
}).response;

const embedEndpoint = await client.add('embed', {
  kind: 'local',
  modelPath: embedModel,
  config: { context: { n_ctx: 2048, embeddings: true, pooling: 'mean' } },
});

// embed: vector output; local endpoint must be embedding-capable.
const embedding = await client.embed({
  endpoint: embedEndpoint,
  input: 'Sipp embedding input.',
  local: { contextKey: 'node-embed', normalize: true },
}).response;

console.log(query.text, chat.text, embedding.values.length);
```

Local `query` also supports encoder-decoder GGUF text models, while many
encoder-decoder models cannot use `chat` because they do not declare
`tokenizer.chat_template`. Encoder-decoder text models do not produce
embeddings through this runtime.

## Python Local

```bash
pip install sipp
```

```python
from sipp import (
    ChatMessage,
    SippClient,
    SippTextOptions,
    ContextRuntimeConfig,
    LocalEmbedOptions,
    LocalTextOptions,
    LocalModelDescriptor,
    NativeRuntimeConfig,
)

client = SippClient()
messages = [
    ChatMessage("system", "Answer concisely."),
    ChatMessage("user", "Explain local Python inference."),
]
query_prompt = "\n".join(
    [
        "<|system|>",
        "Answer concisely.",
        "<|user|>",
        "Explain local Python inference.",
        "<|assistant|>",
    ]
)
text_options = SippTextOptions(max_tokens=64)

text_endpoint = client.add("text", LocalModelDescriptor("chat.gguf"))

# query: raw prompt; replace markers with the target model's template.
query = client.query(
    query_prompt,
    endpoint=text_endpoint,
    options=text_options,
    local=LocalTextOptions(context_key="python-query"),
).result()

# chat: role messages; local runtime uses tokenizer.chat_template.
chat = client.chat(
    messages,
    endpoint=text_endpoint,
    options=text_options,
    local=LocalTextOptions(context_key="python-chat"),
).result()

embed_endpoint = client.add(
    "embed",
    LocalModelDescriptor(
        "embed.gguf",
        NativeRuntimeConfig(
            context=ContextRuntimeConfig(
                n_ctx=2048,
                embeddings=True,
                pooling="mean",
            ),
        ),
    ),
)

# embed: vector output; local endpoint must be embedding-capable.
embedding = client.embed(
    "Sipp embedding input.",
    endpoint=embed_endpoint,
    local=LocalEmbedOptions(context_key="python-embed", normalize=True),
).result()

print(query["text"], chat["text"], len(embedding["values"]))
```

## Rust Local

```bash
cargo add sipp
```

```rust
use sipp::engine::{
    ChatMessage, ChatRole, ContextRuntimeConfig, NativeRuntimeConfig, PoolingType,
};
use sipp::{
    SippChatRequest, SippClient, SippEmbedRequest, SippQueryRequest,
    SippTextOptions, EndpointDescriptor, LocalEmbedOptions, LocalTextOptions,
};

let mut client = SippClient::new();
let messages = vec![
    ChatMessage::new(ChatRole::System, "Answer concisely."),
    ChatMessage::new(ChatRole::User, "Explain local Rust inference."),
];
let query_prompt = [
    "<|system|>",
    "Answer concisely.",
    "<|user|>",
    "Explain local Rust inference.",
    "<|assistant|>",
]
.join("\n");
let text_options = SippTextOptions {
    max_tokens: Some(64),
    ..Default::default()
};

let text_endpoint = client
    .add("text", EndpointDescriptor::local("chat.gguf", Default::default()))
    .await?;

// query: raw prompt; replace markers with the target model's template.
let query = client
    .query(SippQueryRequest {
        endpoint: Some(text_endpoint.clone()),
        prompt: query_prompt,
        options: text_options.clone(),
        local: LocalTextOptions {
            context_key: Some("rust-query".to_string()),
            ..Default::default()
        },
        ..Default::default()
    })
    .await?;

// chat: role messages; local runtime uses tokenizer.chat_template.
let chat = client
    .chat(SippChatRequest {
        endpoint: Some(text_endpoint),
        messages,
        options: text_options,
        local: LocalTextOptions {
            context_key: Some("rust-chat".to_string()),
            ..Default::default()
        },
        ..Default::default()
    })
    .await?;

let embed_endpoint = client
    .add("embed", EndpointDescriptor::local("embed.gguf", embed_config()))
    .await?;

// embed: vector output; local endpoint must be embedding-capable.
let embedding = client
    .embed(SippEmbedRequest {
        endpoint: Some(embed_endpoint),
        input: "Sipp embedding input.".to_string(),
        local: LocalEmbedOptions {
            context_key: Some("rust-embed".to_string()),
            normalize: Some(true),
        },
        ..Default::default()
    })
    .await?;

println!("{}, {}, {}", query.text, chat.text, embedding.values.len());

fn embed_config() -> NativeRuntimeConfig {
    NativeRuntimeConfig {
        context: ContextRuntimeConfig {
            n_ctx: Some(2048),
            embeddings: Some(true),
            pooling: Some(PoolingType::Mean),
            ..Default::default()
        },
        ..Default::default()
    }
}
```

## Gateway

Gateway clients keep model paths, provider credentials, target policy, and
metrics in the gateway process. The example uses the browser package shape;
Node.js uses the same request-object shape shown above.

```ts
import { SippClient, type ChatMessage } from 'sipp';

const client = new SippClient();
const endpoint = await client.add('gateway', {
  kind: 'gateway',
  target: 'local',
  baseUrl: 'https://gateway.example.com',
  authentication: { kind: 'bearer', value: await getGatewayToken() },
});
const messages: readonly ChatMessage[] = [
  { role: 'system', content: 'Answer concisely.' },
  { role: 'user', content: 'Explain gateway inference.' },
];
const queryPrompt = [
  '<|system|>',
  'Answer concisely.',
  '<|user|>',
  'Explain gateway inference.',
  '<|assistant|>',
].join('\n');

// query: gateway forwards the raw prompt to the selected target.
const query = await client.query(queryPrompt, {
  endpoint,
  maxTokens: 64,
}).response;

// chat: gateway maps role messages for the selected provider/local target.
const chat = await client.chat(messages, { endpoint, maxTokens: 64 }).response;

// embed: target must support embeddings.
const embedding = await client.embed('Sipp embedding input.', {
  endpoint,
}).response;

console.log(query.text, chat.text, embedding.values.length);
await client.close();
```

Gateway `query` preserves the raw prompt, so it is the gateway path for custom
templates or local encoder-decoder targets. Gateway `embed` requires the target
to support embeddings.

## Direct Provider

Use direct provider endpoints only in trusted server code (e.g. self-hosted service). Provider support is
model-specific: `query` needs a completion-compatible provider or model,
`chat` needs a chat model, and `embed` needs an embedding model.

```ts
import { SippClient } from 'sipp-server';

function env(name: string): string {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

const client = new SippClient();
const chatMessages = [
  { role: 'system', content: 'Answer concisely.' },
  { role: 'user', content: 'Explain provider inference.' },
];

const completionEndpoint = await client.add('completion', {
  kind: 'provider',
  provider: 'openai_compatible',
  model: env('COMPLETION_MODEL'),
  baseUrl: env('COMPLETION_BASE_URL'),
  apiKey: env('COMPLETION_API_KEY'),
});
const chatEndpoint = await client.add('chat', {
  kind: 'provider',
  provider: 'openai',
  model: env('OPENAI_CHAT_MODEL'),
  apiKey: env('OPENAI_API_KEY'),
});
const embedEndpoint = await client.add('embed', {
  kind: 'provider',
  provider: 'openai',
  model: env('OPENAI_EMBED_MODEL'),
  apiKey: env('OPENAI_API_KEY'),
});

// query: raw completion prompt for a completion-compatible provider.
const query = await client.query({
  endpoint: completionEndpoint,
  prompt: 'Write one provider inference sentence.',
  options: { maxTokens: 64 },
}).response;

// chat: provider-native role messages.
const chat = await client.chat({
  endpoint: chatEndpoint,
  messages: chatMessages,
  options: { maxTokens: 64 },
}).response;

// embed: provider-native embedding model.
const embedding = await client.embed({
  endpoint: embedEndpoint,
  input: 'Sipp embedding input.',
}).response;

console.log(query.text, chat.text, embedding.values.length);
```

## Runtime Tuning

Local endpoint tuning, browser WebGPU options, worker/threading choices,
generation options, and provider/gateway option buckets are documented in
[Runtime Options](../reference/runtime-options.md).

## Building and Running from Source Code

Runnable source examples and demos live in the maintainer lane:
[Source Builds](../maintainers/source-builds.md).
