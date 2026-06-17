# 快速上手

以下代码片段展示 `query`、`chat` 和 `embed` 的调用方式。`query` 接收并发送提示词字符串，不会自动套用对话模板。纯提示词仅适用于基础模型（completion-style / base model）；如果调用纯解码器（decoder-only）的对话或instruct GGUF 模型，需自行在应用层渲染好模型的指令模板。`query` 同样兼容编码器-解码器（encoder-decoder）架构的 GGUF 文本模型。`chat` 接口接收结构化角色消息，由底层自动处理模板。`embed` 返回向量数据，前提是加载的本地模型支持并开启了嵌入特性。

本地上下文标识符的命名在不同语言中仅有大小写差异：浏览器和 Node.js 使用驼峰 `contextKey`；Python 和 Rust 使用蛇形 `context_key`。

可运行示例见[示例与演示](../examples-demos.md)。

## 浏览器本地推理

```bash
npm install @sipp/sipp
```

```ts
import { SippClient, type ChatMessage } from '@sipp/sipp';

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

// query：传递原始提示词。请在应用层将提示词渲染为目标模型对应的模板。
const query = await client.query(queryPrompt, {
  endpoint: textEndpoint,
  maxTokens: 64,
  contextKey: 'browser-query',
}).response;

// chat：传递角色消息列表。本地运行时会自动读取并应用 tokenizer.chat_template。
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

// embed：返回向量数据。指定的本地端点必须具备嵌入生成能力。
const embedding = await client.embed('Sipp embedding input.', {
  endpoint: embedEndpoint,
  contextKey: 'browser-embed',
  normalize: true,
}).response;

console.log(query.text, chat.text, embedding.values.length);
await client.close();
```

## Node.js 本地推理

```bash
npm install @sipp/sipp-server
```

```ts
import { SippClient } from '@sipp/sipp-server';

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

// query：传递原始提示词。请在应用层将提示词渲染为目标模型对应的模板。
const query = await client.query({
  endpoint: textEndpoint,
  prompt: queryPrompt,
  options: textOptions,
  local: { contextKey: 'node-query' },
}).response;

// chat：传递角色消息列表。本地运行时会自动读取并应用 tokenizer.chat_template。
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

// embed：返回向量数据。指定的本地端点必须具备嵌入生成能力。
const embedding = await client.embed({
  endpoint: embedEndpoint,
  input: 'Sipp embedding input.',
  local: { contextKey: 'node-embed', normalize: true },
}).response;

console.log(query.text, chat.text, embedding.values.length);
```

虽然本地 `query` 同样兼容编码器-解码器（encoder-decoder）架构的 GGUF 文本模型, 但是由于许多编码器-解码器模型缺少 `tokenizer.chat_template` 声明，无法支持 `chat` 接口。当前版本的运行时不支持使用编码器-解码器文本模型来生成嵌入数据。

## Python 本地推理

```bash
pip install sipp-py
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

# query：传递原始提示词。请在应用层将提示词渲染为目标模型对应的模板。
query = client.query(
    query_prompt,
    endpoint=text_endpoint,
    options=text_options,
    local=LocalTextOptions(context_key="python-query"),
).result()

# chat：传递角色消息列表。本地运行时会自动读取并应用 tokenizer.chat_template。
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

# embed：返回向量数据。指定的本地端点必须具备嵌入生成能力。
embedding = client.embed(
    "Sipp embedding input.",
    endpoint=embed_endpoint,
    local=LocalEmbedOptions(context_key="python-embed", normalize=True),
).result()

print(query["text"], chat["text"], len(embedding["values"]))
```

## Rust 本地推理

```bash
cargo add sipp-rs
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

// query：传递原始提示词。请在应用层将提示词渲染为目标模型对应的模板。
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

// chat：传递角色消息列表。本地运行时会自动读取并应用 tokenizer.chat_template。
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

// embed：返回向量数据。指定的本地端点必须具备嵌入生成能力。
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

## 网关服务

网关客户端会在网关服务端进程内安全托管模型路径、服务商凭证、请求路由策略及各项性能指标。以下代码片段演示了浏览器端的网关调用，Node.js 同样可复用这套统一的请求对象结构。

```ts
import { SippClient, type ChatMessage } from '@sipp/sipp';

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

// query：网关将原始提示词原样透传至选定的目标后端。
const query = await client.query(queryPrompt, {
  endpoint,
  maxTokens: 64,
}).response;

// chat：网关负责映射并适配角色消息，发往选定的服务商或本地目标。
const chat = await client.chat(messages, { endpoint, maxTokens: 64 }).response;

// embed：请求的后端目标必须支持嵌入生成。
const embedding = await client.embed('Sipp embedding input.', {
  endpoint,
}).response;

console.log(query.text, chat.text, embedding.values.length);
await client.close();
```

通过网关调用 `query` 时，它会保留原始提示词不做修改。因此，当你需要使用自定义对话模板或者将请求发往本地运行的编码器-解码器模型时，推荐使用该接口路径。通过网关调用 `embed` 时，依然要求目标端点支持嵌入功能。

## 直连服务商

除非在绝对安全可控的服务端代码中 (比如自建本地服务)，否则请勿直连第三方服务商的端点。服务商的能力支持情况取决于你选用的具体模型：`query` 接口要求服务商或模型兼容补全（completion）模式，`chat` 支持大部分服务商chat模型端口，而 `embed` 则必须调用专门的嵌入模型。

```ts
import { SippClient } from '@sipp/sipp-server';

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

// query：传递原始补全提示词，发往兼容补全协议的服务商。
const query = await client.query({
  endpoint: completionEndpoint,
  prompt: 'Write one provider inference sentence.',
  options: { maxTokens: 64 },
}).response;

// chat：向服务商发送其原生支持的角色消息格式。
const chat = await client.chat({
  endpoint: chatEndpoint,
  messages: chatMessages,
  options: { maxTokens: 64 },
}).response;

// embed：调用服务商提供的原生嵌入模型。
const embedding = await client.embed({
  endpoint: embedEndpoint,
  input: 'Sipp embedding input.',
}).response;

console.log(query.text, chat.text, embedding.values.length);
```

## 运行时性能调优

有关如何优化本地端点、配置浏览器 WebGPU 参数、管理 Worker 及线程分配、控制生成过程选项，以及配置服务商/网关选项的完整清单，参阅[运行时选项](../reference/runtime-options.md)。

## 源码构建与运行

有关如何运行源码级别的示例和演示项目，参阅维护者专属的[源码构建](../maintainers/source-builds.md)指南。
