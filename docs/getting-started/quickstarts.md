# Quickstarts

These examples use the published client packages. Use a local GGUF path or a
browser-served model URL for local inference.

## Browser Local Query

```bash
npm install cogentlm
```

```ts
import { CogentClient } from 'cogentlm';

const client = new CogentClient();
const endpoint = await client.add('default', {
  kind: 'local',
  source: '/models/model.gguf',
});
const run = client.query('Explain local inference in one sentence.', {
  endpoint,
  maxTokens: 64,
});
console.log((await run.response).text);
await client.close();
```

## Node.js Local Query

```bash
npm install cogentlm-server
```

```ts
import { CogentClient } from 'cogentlm-server';

const client = new CogentClient();
const endpoint = await client.add('default', {
  kind: 'local',
  modelPath: process.argv[2],
});
const run = client.query({
  endpoint,
  prompt: 'Explain local inference in one sentence.',
  options: { maxTokens: 64 },
});
console.log((await run.response).text);
```

## Python Local Query

```bash
pip install cogentlm
```

```python
from cogentlm import CogentClient, CogentTextOptions, LocalModelDescriptor

client = CogentClient()
endpoint = client.add("default", LocalModelDescriptor("model.gguf"))
run = client.query(
    "Explain local inference in one sentence.",
    endpoint=endpoint,
    options=CogentTextOptions(max_tokens=64),
)
print(run.result()["text"])
```

## Rust Local Query

```bash
cargo add cogentlm
```

```rust
use cogentlm::{
    CogentClient, CogentQueryRequest, CogentTextOptions, EndpointDescriptor,
};

let mut client = CogentClient::new();
let endpoint = client
    .add("default", EndpointDescriptor::local("model.gguf", Default::default()))
    .await?;
let response = client
    .query(CogentQueryRequest {
        endpoint: Some(endpoint),
        prompt: "Explain local inference in one sentence.".to_string(),
        options: CogentTextOptions {
            max_tokens: Some(64),
            ..Default::default()
        },
        ..Default::default()
    })
    .await?;
println!("{}", response.text);
```

## Gateway Query

Gateway clients use the same `query`, `chat`, and `embed` calls after
registering a gateway endpoint. The gateway owns local model paths, provider
credentials, target policy, and metrics.

```ts
import { CogentClient } from 'cogentlm';

const client = new CogentClient();
const endpoint = await client.add('gateway', {
  kind: 'gateway',
  target: 'local',
  baseUrl: 'https://gateway.example.com',
  authentication: { kind: 'bearer', value: await getGatewayToken() },
});
const run = client.query('Explain gateway inference.', {
  endpoint,
  maxTokens: 64,
});
console.log((await run.response).text);
await client.close();
```

## Provider Query

Trusted server code can register a direct provider endpoint with
`kind: 'provider'`. Keep provider credentials in server environment variables
such as `OPENAI_API_KEY`; never put long-lived provider keys in browser
bundles. See [Providers](../guides/providers.md) for copyable server and
gateway-backed provider patterns.

## Runtime Tuning

Local endpoint tuning, browser WebGPU options, worker/threading choices,
generation options, and provider/gateway option buckets are documented in
[Runtime Options](../reference/runtime-options.md).

## Building and Running from Source Code

Runnable source examples and demos live in the maintainer lane:
[Source Builds](../maintainers/source-builds.md).
