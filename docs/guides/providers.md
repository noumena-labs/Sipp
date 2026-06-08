# Providers

CogentLM can call external providers directly from trusted server-side
processes or indirectly through a CogentLM gateway. Both paths use the same
endpoint model: register a descriptor with `CogentClient.add`, keep the
endpoint reference, and pass it to `query`, `chat`, or `embed`.

Provider credentials must stay in trusted code. Do not ship long-lived provider
keys in browser bundles.

## Direct Provider Endpoints

Use a direct provider endpoint when the current server process owns the
credential lifecycle and application policy. This is the recommended framework
route pattern for Next.js and TanStack server code.

```ts
import { CogentClient } from 'cogentlm-server';

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

const client = new CogentClient();
const endpoint = await client.add('provider', {
  kind: 'provider',
  provider: 'openai',
  model: process.env.OPENAI_MODEL ?? 'gpt-5-mini',
  apiKey: requiredEnv('OPENAI_API_KEY'),
});

const run = client.chat({
  endpoint,
  messages: [{ role: 'user', content: 'Explain provider inference.' }],
  options: { maxTokens: 128, temperature: 0.2 },
});
console.log((await run.response).text);
```

Use `OPENAI_API_KEY="<mock-openai-key>"` only as a placeholder in docs and
examples. Real keys belong in environment variables or a secret manager.

## Provider Options

Typed request fields should use CogentLM's request options. Provider-only
fields belong in `providerOptions`:

```ts
const run = client.chat({
  endpoint,
  messages,
  options: { maxTokens: 128 },
  providerOptions: {
    reasoning_effort: 'low',
  },
});
```

`providerOptions` is for direct provider endpoints. Gateway-specific extensions
belong in `endpointOptions` or descriptor-level `protocolOptions`, because the
gateway implementation owns how those fields are interpreted.

## Provider-Backed Gateway Targets

Use the first-party gateway when multiple applications should share target
policy, provider credentials, local model hosting, admission control, metrics,
or a stable HTTP boundary.

OpenAI target:

```toml
[[targets]]
name = "openai-chat"
type = "openai"
model = "gpt-5-mini"
api_key_env = "OPENAI_API_KEY"
```

OpenAI-compatible target:

```toml
[[targets]]
name = "compatible-chat"
type = "openai_compatible"
model = "provider-model"
base_url = "https://provider.example/v1"
token_env = "COMPATIBLE_API_TOKEN"
correlation_header = "x-request-id"
```

Anthropic target:

```toml
[[targets]]
name = "anthropic-chat"
type = "anthropic"
model = "claude-3-5-sonnet-latest"
api_key_env = "ANTHROPIC_API_KEY"
```

Gateway clients receive only the public target name, gateway URL, and gateway
authentication value. Provider credentials stay in the gateway process.

## Browser Applications

Browser applications should usually call an application route or gateway, not a
provider directly. If a BYOK browser flow is required, use short-lived provider
keys supplied at runtime through the browser provider descriptor and keep the
user-facing risks explicit.

## Related Docs

- [Frameworks](../packages/frameworks/)
- [Gateway Server](../packages/gateway-server.md)
- [Gateway And Hybrid Inference](gateway-hybrid.md)
- [Runtime Options](../reference/runtime-options.md)
