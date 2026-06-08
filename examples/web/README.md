# Web Examples

The web examples are Vite pages that demonstrate browser-local GGUF inference
and gateway calls. Shared code in `src/common.ts` handles DOM wiring and output
formatting; each page module owns its `CogentClient`, endpoint registration,
request construction, streaming, and cleanup.

## Serve

```bash
cargo xtask run examples serve browser
```

Open:

- `/query.html`: local GGUF query
- `/chat.html`: local GGUF chat with streaming
- `/embed.html`: local GGUF embeddings
- `/gateway_local.html`: browser-local GGUF and a separate local gateway side by side
- `/gateway_query.html`: gateway query
- `/gateway_chat.html`: gateway chat with streaming
- `/gateway_embed.html`: gateway embeddings

## Gateway Pages

Use the one-command gateway workflow when possible:

```bash
cargo xtask run examples gateway web --case query
```

For a manually started gateway:

```bash
export COGENTLM_GATEWAY_TOKEN="dev-token"
cargo xtask run examples serve gateway-local --model <model.gguf> --bind 127.0.0.1:8787
```

Then enter URL `http://127.0.0.1:8787`, any non-empty token, and target
`local` in the browser page.

See [../README.md](../README.md) for shared example workflow details and
[../../docs/packages/browser.md](../../docs/packages/browser.md) for browser
package docs.
