# apps/proactive-ui

Proactive UI demo for Cogent Engine's vision pipeline.

The app is a toy game called **Dust Ridge Field Kit**. The user packs gear for a desert expedition while a vision model periodically peeks at the visible UI, explains what the user appears to be doing, and returns strict JSON patches that the app validates and applies to annotated DOM targets. The main showcase is an AI projection layer that can generate custom, context-specific UI from the current kit state.

## Quick start

From the repo root:

```bash
bun install
bun run proactive-ui:dev
```

Open the printed local URL and load the default vision model/projector pair from the start screen. The demo enters automatically after the model is ready. Select a few gear cards, then click `Peek` in the floating developer drawer.

## What this demonstrates

- capturing an app surface as an image with `html-to-image`
- sending screenshot bytes to `engine.chat({ messages, media })`
- constraining model output with a JSON GBNF grammar
- validating model-authored DOM patches against a scanned target contract
- sanitizing generated HTML with `dompurify`
- applying safe DOM mutations while showing the full behind-the-scenes trace
- downscaling the captured UI to a smaller JPEG in `fast` mode for lower-latency vision inference
- rendering one model-authored callout beside the most important highlighted or changed DOM element
- projecting model-authored tradeoff UI that combines coverage, weight, budget, selected gear, and candidate swaps

## Demo flow

The game is a single field-kit board:

- all gear cards are visible and grouped by survival category
- the AI projection layer sits above the board and can become a mission lens, swap analysis, budget/weight visualization, or launch rationale
- the checklist, selected manifest, launch gate, and coach panel stay visible
- the drawer starts open, can be minimized, and remains excluded from screenshot capture

The floating developer drawer is intentionally separate from the main game UI. The model only sees the field-kit surface, not the debugging controls.

## Drop-in pattern

Annotate mutable regions with `data-ai-id`, `data-ai-label`, and `data-ai-ops`:

```html
<aside
  data-ai-id="synthesis-overlay"
  data-ai-label="AI projection layer for custom tradeoff UI"
  data-ai-ops="replaceHtml,appendHtml,addClass,removeClass,setAttribute,scrollIntoView"
></aside>
```

Before each peek, the app scans these annotations into a DOM contract. The model can only patch listed targets with listed operations.

The director constraints live in:

```text
public/directors/field-kit/dom-patch-director.json
```

That config defines the scenario objective, model instructions, patch limits, allowed CSS classes, allowed HTML classes, and allowed attributes.

## Patch response

The model is asked to return only JSON:

```json
{
  "observation": "The user selected navigation gear but hydration is still missing.",
  "intent": "The user is trying to prepare the expedition kit for launch.",
  "patches": [
    {
      "op": "addClass",
      "targetId": "goal-hydration",
      "className": "ai-warning",
      "note": "Hydration is still missing. Add a water item before launching."
    },
    {
      "op": "replaceHtml",
      "targetId": "synthesis-overlay",
      "html": "<div class=\"ai-synth-card ai-synth-warning\"><p class=\"ai-synth-kicker\">Mission lens</p><h3 class=\"ai-synth-title\">Hydration first, budget still safe</h3><div class=\"ai-synth-grid\"><div class=\"ai-synth-metric\"><span class=\"ai-synth-label\">Coverage</span><strong class=\"ai-synth-value\">1 / 6</strong><span class=\"ai-synth-meter\"><span class=\"ai-synth-fill ai-synth-fill-25\"></span></span></div><div class=\"ai-synth-metric\"><span class=\"ai-synth-label\">Budget left</span><strong class=\"ai-synth-value\">$398</strong></div></div><p class=\"ai-synth-why\">Twin water bladders are heavy, but they close the highest desert risk while leaving enough budget for signal and first aid.</p></div>",
      "note": "The projection combines survival risk, budget room, and weight cost instead of showing a static warning."
    }
  ]
}
```

## Safety constraints

The demo intentionally lets the model generate DOM patches, but not arbitrary execution.

- unknown targets are rejected
- operations must be allowed by each target
- CSS classes must be allowlisted
- generated HTML is sanitized
- generated projection UI can only use allowlisted `ai-gen-*` and `ai-synth-*` classes
- scripts, styles, iframes, images, forms, inputs, event handlers, and unsafe attributes are stripped
- patch count and text/HTML lengths are capped
- missing patch notes are replaced with a safe fallback note, and the first accepted note is rendered as an overlay callout instead of being inserted into the patched element
- rejected patches are shown in the trace panel and skipped

Projection cards can use `ai-synth-card`, `ai-synth-grid`, `ai-synth-metric`, `ai-synth-meter`, `ai-synth-fill-*`, `ai-synth-chip`, `ai-synth-swap`, and related classes. Inline styles are not needed or allowed.

## Defaults

- Model: `https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/LFM2.5-VL-450M-F16.gguf`
- Projector: `https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/mmproj-LFM2.5-VL-450m-F16.gguf`

These are the same LiquidAI vision defaults used by `apps/examples`.
