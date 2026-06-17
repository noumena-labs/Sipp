# 推理操作

Sipp 将操作与端点分离。根据输入格式和期望输出选择 `query`、`chat` 或 `embed`，然后传入决定请求执行位置的端点引用。

## 通用流程

1. 调用 `SippClient.add` 注册本地、网关或提供商端点。
2. 保存返回的端点引用。
3. 将引用传入 `query`、`chat` 或 `embed`。

`query` 和 `chat` 都生成文本，都支持 `maxTokens`、`temperature`、`topP`、`stop`、取消请求和流式 Token。`embed` 生成向量，不支持这些文本选项，也不流式。

| 操作 | 输入 | 输出 | 适用场景 |
| --- | --- | --- | --- |
| `query` | 一段提示词文本 | 生成的文本 | 原始补全、自定义模板、编码器-解码器模型、少样本提示、自行构建提示词的智能体 |
| `chat` | 有序的 `{ role, content }` 消息列表 | 生成的助手回复 | 对话式调用，由端点处理聊天模板或消息格式转换 |
| `embed` | 一段文本 | 一个嵌入向量 | 检索、语义搜索、排序、聚类、记忆索引 |

## 本地推理

本地端点在当前浏览器、Node、Python、Rust 或 CLI 进程中直接运行 GGUF 模型。

| 操作 | Sipp 传给运行时的内容 | 模板行为 | 仅限本地的选项 |
| --- | --- | --- | --- |
| `query` | 完整的提示词字符串。仅解码器模型直接解码；编码器-解码器模型先编码再解码。 | 不应用聊天模板。适合自行处理提示词格式的场景。 | 上下文键、文法约束、JSON Schema、采样覆盖、媒体输入 |
| `chat` | 使用 llama.cpp 的聊天模板和 `add_assistant = true` 将消息列表渲染为提示词。 | GGUF 必须在元数据中声明 `tokenizer.chat_template`。Sipp 会先检查模型元数据，不依赖 llama.cpp 的 fallback。 | 与 `query` 相同的文本选项，包括上下文键和媒体输入 |
| `embed` | 输入文本交给本地嵌入运行时编码。 | 不应用模板，不生成文本。 | 上下文键、嵌入归一化 |

本地 `chat` 只负责渲染提示词和生成文本，不保存对话历史。若需模型访问对话历史，在 `messages` 中显式传入即可。上下文键仅在需要复用本地 KV 缓存时有用。

编码器-解码器模型（如 T5、BART 的 GGUF）使用 `query` 生成文本。纯编码器模型不生成文本，若支持池化嵌入则使用 `embed`。

### 本地 query（原始提示词）

```ts
const run = client.query({
  endpoint,
  prompt,
  options: { maxTokens: 128, temperature: 0.2 },
  local: { contextKey: 'docs-example' },
  emitTokens: true,
});
```

### 本地 chat（对话消息）

仅当 GGUF 模型声明了聊天模板时才能使用 `chat`。Sipp 将角色消息交由 llama.cpp 渲染模板，然后基于渲染后的提示词生成文本。

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

如果模型没有 `tokenizer.chat_template`，本地 `chat` 会失败。基座模型、老旧模型，或需自行控制模板格式时，改用 `query` 传入显式拼接的提示词。

### 编码器-解码器模型的本地 query

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

绝大多数编码器-解码器文本模型没有 GGUF 聊天模板，因此 `query` 可用，`chat` 会报错。

### 本地嵌入

```ts
const run = client.embed({
  endpoint,
  input: 'Vectorize this sentence for retrieval.',
  local: { normalize: true },
});

const embedding = (await run.response).values;
```

## 远程网关

网关端点通过 HTTP 发送请求。官方网关使用固定的路由和请求体格式：

| 操作 | 默认路由 | 请求体必须包含 |
| --- | --- | --- |
| `query` | `/v1/query` | `model`, `prompt` |
| `chat` | `/v1/chat` | `model`, `messages` |
| `embed` | `/v1/embed` | `model`, `input` |

`model` 是网关公开的目标名称。网关收到请求后，将该名称解析为具体的本地 GGUF 模型、OpenAI 端点、兼容 OpenAI 的端点或 Anthropic 端点。

网关调用支持 `query` 和 `chat` 的通用文本选项：`max_tokens`、`temperature`、`top_p`、`stop`、`stream`。`contextKey`、`grammar`、`jsonSchema`、`sampling`、`media`、`normalize` 这些仅限本地的字段，网关端点会拒绝。`providerOptions` 也只对直连提供商有效，网关同样拒绝；自定义网关需自行处理提供商特定的扩展字段。

### 网关目标映射

| 网关目标类型 | `query` | `chat` | `embed` |
| --- | --- | --- | --- |
| 本地 GGUF | 运行原始提示词。仅解码器直接解码；编码器-解码器先编码预填充再解码。不应用聊天模板。 | 使用 GGUF 声明的聊天模板渲染消息。模型无模板（包括大多数编码器-解码器模型）时报错。 | 模型或运行时支持嵌入即可使用。编码器-解码器模型不生成嵌入。 |
| OpenAI | 发 OpenAI 补全请求（带 `prompt`）。 | 发 OpenAI 聊天补全请求（带 `messages`）。 | 发 OpenAI 嵌入请求（带 `input` 和 `encoding_format: "float"`）。 |
| 兼容 OpenAI | 发 `/completions` 请求（带 `prompt`）。 | 发 `/chat/completions` 请求（带 `messages`）。 | 发 `/embeddings` 请求（带 `input` 和 `encoding_format: "float"`）。 |
| Anthropic | 将提示词封装为用户消息，发送 Anthropic `/messages` 请求。 | 发送 Anthropic `/messages` 请求；system 角色合并到顶层 `system` 字段，user 和 assistant 保留在 `messages` 中。 | 原生 Anthropic 适配器不支持。 |

最终可用性取决于上游模型和提供商。例如，兼容 OpenAI 的目标可能只支持聊天接口而不支持补全接口，此时网关 `chat` 成功，`query` 失败。

### 网关客户端示例

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

### 直接发 HTTP 请求

Query: 

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

Embed: 

```bash
curl -X POST "$SIPP_GATEWAY_URL/v1/embed" \
  -H "Authorization: Bearer $SIPP_GATEWAY_TOKEN" \
  -H "content-type: application/json" \
  -d '{
    "model": "openai-embed",
    "input": "Text to index for retrieval."
  }'
```

## 快速抉择

- 自行控制提示词格式（自定义模板、编码器-解码器模型）→ 本地 `query`
- GGUF 有聊天模板且有消息列表 → 本地 `chat`
- 当前进程需要生成向量且需本地归一化 → 本地 `embed`
- 目标支持原始提示词（本地仅解码器、编码器-解码器、兼容 OpenAI 补全）→ 网关 `query`
- 目标支持聊天（服务商聊天模型、有模板的本地 GGUF）→ 网关 `chat`
- 目标支持嵌入（本地、OpenAI、兼容 OpenAI）→ 网关 `embed`；原生 Anthropic 不支持

## 相关文档

- [本地推理](local-inference.md)
- [网关与混合推理](gateway-hybrid.md)
- [服务商](providers.md)
- [运行时选项](../reference/runtime-options.md)
