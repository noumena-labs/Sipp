# Quickstarts

These commands assume a source checkout at the repository root.

## Run A Local Example

Local examples take a GGUF model path and optional input:

```bash
cargo run -p cogentlm-rust-examples --bin query -- <model.gguf> "Explain local inference."
node examples/node/query.mjs <model.gguf> "Explain local inference."
python examples/python/query.py <model.gguf> "Explain local inference."
```

Use the matching `chat` or `embed` example for chat and embedding workflows.

## Run A Gateway Workflow

The one-command gateway workflow starts a local gateway, runs a client example,
and stops the gateway when the client exits.

```bash
cargo xtask run examples gateway rust --case query
cargo xtask run examples gateway node --case chat
cargo xtask run examples gateway python --case embed
```

The workflow uses token `dev-token`, binds the gateway to `127.0.0.1:8787`,
and uses the cached sample model under `.build/models` unless `--model` is
provided.

## Serve Browser Examples

```bash
cargo xtask run examples serve browser
```

Open the printed local URL and choose one of the example pages:

- `/query.html`
- `/chat.html`
- `/embed.html`
- `/gateway_query.html`
- `/gateway_chat.html`
- `/gateway_embed.html`

## Serve A Demo

```bash
cargo xtask run demos serve chat
cargo xtask run demos serve avatar
cargo xtask run demos serve simulation
cargo xtask run tools serve playground
```

Use demos for exploratory workflows and examples for small, copyable
integrations.
