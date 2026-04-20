# apps/avatar

A minimal three.js + React example that binds the `cogent-engine` character
harness to a [VRM](https://vrm.dev/) avatar. It streams model output
on-device via WebAssembly, parses inline `<action …/>` tags into scene
gestures, and optionally speaks the response through Web Speech with
lipsync.

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
driving a persona named **Aria** and five actions: `wave`, `nod`,
`shake_head`, `set_mood`, `look_at`.

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
- **(Optional) Web Speech synthesis.** Chromium-based browsers work out of
  the box. When unsupported, the TTS checkbox is disabled.

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
    "style": "warm, concise, playful"
  },
  "actions": {
    "actions": [
      { "name": "wave", "description": "Wave a greeting.", "args": [] },
      { "name": "set_mood",
        "description": "Change facial expression.",
        "args": [{ "name": "mood", "type": "enum",
                   "values": ["happy","sad","surprised","angry","neutral"] }] }
    ]
  },
  "assets": { "vrm": "/avatar.vrm", "portrait": "/portrait.png" },
  "memory": { "maxTurns": 8 }
}
```

Actions listed here become part of the GBNF grammar handed to the sampler,
so the model literally cannot emit an action not in the schema. Unknown
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

- `src/App.tsx` — wires engine + agent + bus + lipsync + TTS.
- `src/scene/scene.ts` — three.js renderer, lighting, animation loop.
- `src/scene/vrm-loader.ts` — GLTFLoader + `VRMLoaderPlugin` with a
  primitive fallback.
- `src/bindings/three-vrm-binding.ts` — maps `ActionBus` events to VRM
  humanoid bones, expression presets, and lookAt.
- `src/components/AvatarCanvas.tsx` — mounts the scene, wires resize,
  forwards the `LipsyncDriver` into the binding.
- `src/components/ChatPanel.tsx` — streaming chat bubbles with action chips.
- `src/components/ControlsPanel.tsx` — character/model URL inputs + TTS
  toggle.

---

## Voice + lipsync

The app creates a single `LipsyncDriver` and a single Web Speech TTS
adapter at mount and keeps them alive across character reloads. When the
"Speak responses" checkbox is on and the turn completes:

1. Accumulated prose for the turn is gathered.
2. `lipsync.start()` begins emitting openness samples at 30 Hz.
3. `tts.speak(prose)` resolves when the utterance ends.
4. `lipsync.stop()` closes the mouth.

For VRM avatars the openness signal is applied to the `Aa` expression
preset (widest-mouth viseme). For the primitive fallback it scales the
`head` mesh along Y — cartoon-obvious, by design, because the primitive
avatar is not meant to look realistic.

Sending a new message or resetting memory cancels any in-flight TTS and
stops the driver immediately.

---

## Known limitations

- **Single utterance per turn.** Web Speech has no incremental synthesis
  API; chopping mid-turn produces awkward prosody, so the app speaks the
  full turn's prose at `turn-end`. Swap in a streaming TTS for v2.
- **Pseudo-phoneme lipsync.** The v1 openness signal is a band-limited
  sine + jitter, not derived from real phoneme timings. Mouth shapes are
  plausible but not accurate. Plug real phoneme boundaries into
  `LipsyncDriver` for v2.
- **No per-character voice override.** `CharacterConfig` doesn't yet carry
  a `voice` section, so all characters use the browser default voice. This
  is a deliberate v1 omission, not a bug.
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
- **Avatar loaded but no mouth movement** — confirm the VRM has the `Aa`
  expression preset. Some exporters only ship viseme presets under custom
  names; `ThreeVRMBinding` can be extended to map them.
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
