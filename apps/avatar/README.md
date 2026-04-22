# apps/avatar

A minimal three.js + React example that binds the `cogent-engine` character
harness to a [VRM](https://vrm.dev/) avatar. It streams model output
on-device via WebAssembly, parses inline bracketed cues into scene gestures,
and renders the conversation in a minimal chat UI.

This is an **example**, not a product. It is deliberately small so the
plumbing is readable end-to-end.

---

## Quick start

From the monorepo root:

```bash
bun install
bun run build:wasm        # builds the native runtime once
bun run avatar:dev        # Vite dev server
```

Then open the printed localhost URL, paste a `.gguf` model URL into the
"Model" field, and press **Load**. The first load downloads the model into
`OPFS` (persisted across reloads) and spins up the inference runtime.

The app ships with a starter `character.json` at `public/character.json`
driving a persona named **Aria** and flat actions such as `wave`, `nod`,
`shake_head`, `smile`, and `look_at_you`.

Scripts:

| Command | Description |
| --- | --- |
| `bun run avatar:dev` | Vite dev server with HMR |
| `bun run avatar:build` | Production build into `apps/avatar/dist/` |
| `bun run avatar:preview` | Preview the production build locally |

---

## What you need

- **A `.gguf` model URL** served with permissive CORS. Any small chat model
  works; the starter persona expects an instruct-tuned model (e.g. a
  Qwen-2.5 or Llama-3.2 instruct at 1–3B). The URL is prompted at runtime —
  no build-time config.
- **(Optional) A `.vrm` avatar** at `/avatar.vrm` (or any URL you put in
  `character.json → assets.vrm`). If no VRM is provided or loading fails,
  the scene falls back to a **primitive capsule figure** with named
  `head`, `armL`, `armR` meshes so gestures still play.

---

## `character.json` shape

See `packages/cogent-engine/docs/character-harness.md` for the full schema.
The starter file is a complete, valid example:

```jsonc
{
  "id": "aria",
  "persona": {
    "name": "Aria",
    "description": "A cheerful, curious virtual companion…",
    "dialogExamples": [
      { "user": "hi", "assistant": "[wave] Hi there!" },
      { "user": "what's your name?", "assistant": "[smile] I'm Aria." }
    ]
  },
  "actions": {
    "actions": [
      { "name": "wave", "description": "Wave a greeting." },
      { "name": "smile", "description": "Smile warmly." },
      { "name": "look_at_you", "cue": "look at you", "description": "Briefly turn attention toward the user." }
    ]
  },
  "assets": { "vrm": "/avatar.vrm", "portrait": "/portrait.png" },
  "memory": { "maxTurns": 8 }
}
```

Actions listed here become part of the GBNF grammar handed to the sampler,
so the model literally cannot emit a cue not in the schema. Unknown
actions are logged and ignored by the scene binding, not by the parser.

---

## Architecture at a glance

```
character.json ──► parseCharacterConfig ──► CharacterAgent
                                                │
                                     queuePrompt(grammar, onToken)
                                                │
                                                ▼
                                         CogentEngine
                                                │
                            tokens ◄────────────┘
                                                │
                                  StreamingActionParser
                                                │
                    ┌──────────────── prose / action events ──────────────┐
                    ▼                                                     ▼
              ChatPanel (UI)                                       ActionBus
                                                                        │
                                               ┌───────────────────────┐│
                                               ▼                        ▼
                                        ThreeVRMBinding          (your bindings)
                                               │
                                               ▼
                                       three.js scene / VRM
```

 - `src/App.tsx` — wires engine + agent + bus + chat UI.
- `src/scene/scene.ts` — three.js renderer, lighting, animation loop.
- `src/scene/vrm-loader.ts` — GLTFLoader + `VRMLoaderPlugin` with a
  primitive fallback.
- `src/bindings/three-vrm-binding.ts` — maps `ActionBus` events to VRM
  humanoid bones, expression presets, and lookAt.
- `src/components/AvatarCanvas.tsx` — mounts the scene and wires resize.
- `src/components/ChatPanel.tsx` — streaming chat bubbles with action chips.
- `src/components/ControlsPanel.tsx` — character/model URL inputs and reset.

---

## Known limitations

- **Text-only interaction.** The example intentionally omits speech input,
  speech output, and lipsync while the core chat and action loop is kept
  small and stable.
- **Primitive fallback is ugly on purpose.** It exists to prove that
  gestures drive correctly without a VRM asset; ship a real `.vrm` for any
  demo you'd show someone.
- **Memory is a plain sliding window.** No summarisation, no vector recall.
  Good enough for short sessions; will forget aggressively across long
  ones.

---

## Troubleshooting

- **"character.json HTTP 404"** — the file must be served by Vite; default
  is `/character.json` resolved from `public/`.
- **"Load failed: …"** — the model URL must return a `.gguf` file with
  permissive CORS. HuggingFace resolve URLs work if the repo is public.
- **Actions never fire** — the model may be too small to follow the grammar
  reliably. Try a bigger instruct model, or simplify the action schema.

---

## Further reading

- `packages/cogent-engine/docs/character-harness.md` — full harness API
  and memory model.
- `packages/cogent-engine/docs/grammar-sampling.md` — GBNF transport and
  per-slot sampler invariants.
- `packages/cogent-engine/docs/inference_system_design.md` — the
  engine/runtime architecture the harness sits on top of.
