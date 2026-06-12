# Proactive UI Demo

`demos/proactive-ui` is a speed-first proactive drawing demo for Sipp's
vision pipeline. The app captures a sketch, asks a vision model to describe it,
then asks a text pass to produce a short commentary response from the parsed
visual facts.

## Run

```bash
cargo xtask run demos serve proactive-ui
```

Open the printed local URL, load the default vision model/projector pair, then
draw on the canvas. Auto inference runs after each stroke by default.

## What It Demonstrates

- Canvas capture without DOM screenshotting.
- Ink-cropped low-resolution JPEG frames for lower latency.
- JPEG media sent through `client.chat({ messages, media }).response`.
- Short line-protocol perception output instead of slow JSON.
- Commentary-only model output with no model-authored canvas writes.
- `AI_TRACE` metadata for prompts, raw outputs, parsed perception, timings, and
  runtime metrics.

The director config lives at
`public/directors/sketch-lab/drawing-director.json`.

See [../../docs/guides/vision.md](../../docs/guides/vision.md) for vision
workflow guidance.
