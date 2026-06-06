# Web Examples

The web examples are Vite pages that demonstrate browser-local GGUF inference
and gateway calls. Shared code in `src/common.ts` handles DOM wiring and output
formatting; each page module owns its `CogentClient`, endpoint registration,
request construction, streaming, and cleanup.

Browser endpoints use the same unified descriptor API:

```ts
const endpoint = await client.add('local', {
  kind: 'local',
  source,
  options: { runtime },
});
```

Start the app:

```bash
cargo xtask run examples serve browser
```

Open:

- `/query.html`: local GGUF query
- `/chat.html`: local GGUF chat with streaming
- `/embed.html`: local GGUF embeddings
- `/gateway_local.html`: local browser GGUF and separately running local gateway side by side
- `/gateway_query.html`: gateway query
- `/gateway_chat.html`: gateway chat with streaming
- `/gateway_embed.html`: gateway embeddings

For gateway pages, start a gateway separately:

```bash
export COGENTLM_GATEWAY_TOKEN="dev-token"
cargo xtask run examples serve gateway-local --model <model.gguf> --bind 127.0.0.1:8787
```

Then enter URL `http://127.0.0.1:8787`, token `dev-token`, and alias `local`
for query/chat pages. Use alias `local-embed` for the embedding page.

Open `http://127.0.0.1:8787/` to inspect gateway status and request history.
The xtask gateway serve command uses `dev-token` as the dashboard admin token.

OpenAI gateway pages require the gateway process to have `OPENAI_API_KEY` set.
Use alias `openai-chat` for query/chat and `openai-embed` for embeddings.
