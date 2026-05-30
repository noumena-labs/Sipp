# apps/proactive-ui

Speed-first proactive drawing demo for CogentClient's vision pipeline.

The app is a dynamic sketch pad. The user draws with fixed colors and a simple pen size control. After each stroke, the app exports an ink-cropped low-res JPEG and runs a two-pass model pipeline: first the vision model describes the sketch, then a text-only model pass writes an absurd British sketch-comedy heckle from that description.

## Quick start

From the repo root:

```bash
bun install
bun run proactive-ui:dev
```

Open the printed local URL, load the default vision model/projector pair, then draw on the canvas. Auto inference runs after each stroke by default. Use `AI_TRACE` for manual deeper inspection.

## What this demonstrates

- direct canvas capture without DOM screenshotting
- ink-cropped low-res vision frames for lower latency and stronger visual grounding
- sending JPEG bytes through `client.chat({ messages, media }).response`
- tiny line-protocol model perception responses to avoid slow or truncated JSON
- commentary-only model output, with no model-authored canvas writes
- dynamic model-authored heckles steered by parsed visual facts
- preserving `AI_TRACE` for crop metadata, capture timings, both model prompts, both raw outputs, parsed perception, heckle parsing, and runtime metrics
- latency-oriented runtime settings with small image token budgets and short output budgets

## Demo flow

The main loop is intentionally small:

- user draws on the canvas
- the app waits briefly after stroke end
- the app crops around the visible ink, then exports that crop at the `turbo` preset
- pass 1 returns subject, features, weird detail, and line quality
- pass 2 writes one original theatrical heckle from those parsed facts
- `AI_TRACE` records capture, encode, describe time, heckle time, total time, raw model outputs, parsed perception, and runtime observability

The developer drawer is separate from the captured canvas. It can be minimized and does not affect the image sent to the model.

## Drawing response

The first pass is asked to return visual facts only:

```text
SUBJECT: smiling cartoon head
FEATURES: round head, spiky hair, flat eyes, big smile
WEIRD: spiky hair
QUALITY: thick wobbly marker lines
```

The second pass receives those parsed facts and returns a single dynamic heckle:

```text
HECKLE: That lopsided grin has been inspected by the borough and found guilty of impersonating a mayoral biscuit.
```

The heckle pass is steered toward Monty-Python-like absurd British sketch-comedy energy without hardcoded joke templates or existing-comedy quotes.

## Director Config

The director config lives in:

```text
public/directors/sketch-lab/drawing-director.json
```

It controls the perception persona, heckle persona, perception instructions, and text length caps.

## Defaults

- Model: `https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/LFM2.5-VL-450M-F16.gguf`
- Projector: `https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/mmproj-LFM2.5-VL-450m-F16.gguf`

These are the same LiquidAI vision defaults used by `apps/examples`.
