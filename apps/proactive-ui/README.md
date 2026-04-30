# apps/proactive-ui

Proactive UI demo for Cogent Engine's vision pipeline.

The app is a toy game called **Dust Ridge Field Kit**. The user packs gear for a desert expedition while a vision model periodically peeks at the visible UI, explains what the user appears to be doing, and returns strict JSON patches that the app validates and applies to annotated DOM targets.

## Quick start

From the repo root:

```bash
bun install
bun run proactive-ui:dev
```

Open the printed local URL, load the default vision model/projector pair, select a few gear cards, then click `Peek at UI`.

## What this demonstrates

- capturing an app surface as an image with `html-to-image`
- sending screenshot bytes to `engine.chat({ messages, media })`
- constraining model output with a JSON GBNF grammar
- validating model-authored DOM patches against a scanned target contract
- sanitizing generated HTML with `dompurify`
- applying safe DOM mutations while showing the full behind-the-scenes trace

## Drop-in pattern

Annotate mutable regions with `data-ai-id`, `data-ai-label`, and `data-ai-ops`:

```html
<aside
  data-ai-id="coach-panel"
  data-ai-label="Generated proactive coach panel"
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
      "className": "ai-warning"
    },
    {
      "op": "replaceHtml",
      "targetId": "coach-panel",
      "html": "<div class=\"ai-gen-card\"><h3 class=\"ai-gen-title\">Hydration gap</h3><p class=\"ai-gen-note\">Add a water item before launching.</p></div>"
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
- scripts, styles, iframes, images, forms, inputs, event handlers, and unsafe attributes are stripped
- patch count and text/HTML lengths are capped
- rejected patches are shown in the trace panel and skipped

## Defaults

- Model: `https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/LFM2.5-VL-450M-F16.gguf`
- Projector: `https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/mmproj-LFM2.5-VL-450m-F16.gguf`

These are the same LiquidAI vision defaults used by `apps/examples`.
