# CogentLM Examples

The example directories mirror the same public `CogentClient` workflows across
Rust, Node.js, Python, and the browser:

* `query`: single prompt text generation.
* `chat`: system and user chat messages with token streaming.
* `embed`: embedding generation with a compact vector preview.
* `remote_gateway_query`, `remote_gateway_chat`, and `remote_gateway_embed`:
  the same calls routed through a CogentLM Remote Gateway.

## Local Model Examples

Local Rust, Node.js, and Python examples take a GGUF model path followed by an
optional input string.

```powershell
cargo run -p cogentlm-rust-examples --bin query -- <model.gguf> [input]
node examples/node/query.mjs <model.gguf> [input]
python examples/python/query.py <model.gguf> [input]
```

Swap `query` for `chat` or `embed` to run the other local examples.

## Remote Gateway Examples

Remote examples take a gateway alias followed by an optional input string. Set
the gateway URL and bearer token in the environment before running them.

```powershell
$env:COGENTLM_GATEWAY_URL="http://127.0.0.1:8080"
$env:COGENTLM_GATEWAY_TOKEN="<token>"

cargo run -p cogentlm-rust-examples --features remote --bin remote_gateway_query -- <gateway-alias> [input]
node examples/node/remote_gateway_query.mjs <gateway-alias> [input]
python examples/python/remote_gateway_query.py <gateway-alias> [input]
```

Swap `remote_gateway_query` for `remote_gateway_chat` or
`remote_gateway_embed` to run the other gateway examples.

## Browser Examples

Build the browser package first when the staged package artifacts do not exist:

```powershell
cargo xtask build wasm
```

Then run the Vite example app:

```powershell
cd examples/web
bun run dev
```

Open the matching page:

* `/query.html`
* `/chat.html`
* `/embed.html`
* `/remote_gateway_query.html`
* `/remote_gateway_chat.html`
* `/remote_gateway_embed.html`

Browser gateway pages collect the alias, base URL, and token in the page because
browser code cannot read process environment variables. The token is held only
in memory and is not printed in the output.
