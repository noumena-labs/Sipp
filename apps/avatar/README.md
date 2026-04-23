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

The app ships with a starter character package at
`public/characters/aria/character.json` driving a persona named **Aria** and
flat actions such as `wave`, `nod`, `shake_head`, `smile`, and `look_at_you`.

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
- **A `.vrm` avatar** at `public/characters/<id>/avatar.vrm`.
- **Mixamo `.fbx` clips** for clip-backed actions at
  `public/characters/<id>/animations/<action-name>.fbx`.

---

## `character.json` shape

See `packages/cogent-engine/docs/character-harness.md` for the full schema.
The starter file is a complete, valid example. It is semantic-only: no avatar
model paths, no portraits, and no animation metadata.

```jsonc
{
  "id": "aria",
  "persona": {
    "name": "Aria",
    "summary": "A cheerful, curious companion who stays close to the immediate moment.",
    "role": "A community coordinator at a neighborhood co-working studio.",
    "backstory": "She grew up in her family's stationery shop and learned to notice moods quickly.",
    "currentLife": {
      "description": "She spends her days keeping a shared studio running smoothly in a bright space full of coffee smells, paper scraps, and little interruptions."
    },
    "personality": {
      "traits": ["warm", "curious", "observant"],
      "description": "She notices small details and can over-read tiny social signals."
    },
    "dialogExamples": [
      { "user": "what can you do?", "assistant": "[glance right] I open up the studio, keep the coffee fresh, and smooth out the little hassles so people can focus here." },
      { "user": "what is this space?", "assistant": "[glance right] Bright shared studio. Coffee in the air, people typing at desks, paper scraps on the counter, and the printer acting temperamental again." },
      { "user": "What does a community coordinator do? Can you help me?", "assistant": "[glance right] Mostly I keep this place running. I make sure the coffee is hot, help people find a desk, and smooth out little bumps in the day." },
      { "user": "what's your name?", "assistant": "[smile] I'm Aria." },
      { "user": "write me a Python script", "assistant": "[shake head] You are asking the wrong girl. I can keep you company while you wrestle with it, though." },
      { "user": "hi", "assistant": "[wave] Hi there. You look like you could use a soft landing." }
    ]
  },
  "actions": {
    "actions": [
      { "name": "wave", "description": "Wave a greeting." },
      { "name": "smile", "description": "Smile warmly." },
      { "name": "look_at_you", "cue": "look at you", "description": "Briefly turn attention toward the user." }
    ]
  },
  "memory": { "maxTurns": 8 }
}
```

## Character package layout

The avatar app resolves renderer assets by folder convention, not by fields in
`character.json`:

```text
public/
  characters/
    aria/
      character.json
      avatar.vrm
      animations/
        idle.fbx
        wave.fbx
        nod.fbx
        shake_head.fbx
```

`idle.fbx` is the default looping base pose for the avatar. `wave`, `nod`, and
`shake_head` are clip-backed in v1 and must each have a matching Mixamo `.fbx`
file. Facial expressions and gaze actions remain code-driven inside the app.

If every action includes a `usageHint`, the system prompt also renders a
compact cue-guidance line. If you omit `usageHint` for any action, that cue
still appears in `Supported cues`, but cue-moment guidance is omitted for the
whole character. This is intentional so guidance is either complete or absent,
never partial.

The first three `dialogExamples` also get mirrored into the system prompt as
always-present anchor examples. Put your highest-value anti-drift cases first:
role enactment, environment grounding, and job/assistance questions.

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
        ChatComposer / TranscriptDrawer                             ActionBus
                                                                         │
                                                ┌───────────────────────┐│
                                                ▼                        ▼
                                         ThreeVRMBinding          (your bindings)
                                                │
                                                ▼
                        app-owned avatar.vrm + Mixamo .fbx clips + three.js scene
```

 - `src/App.tsx` — wires engine + agent + bus + chat UI.
- `src/scene/scene.ts` — three.js renderer, lighting, animation loop.
- `src/scene/vrm-loader.ts` — GLTFLoader + `VRMLoaderPlugin` with
  bounds-based centering for successful VRM loads.
- `src/characters/render-assets.ts` — resolves `avatar.vrm` and Mixamo clips
  from the loaded character folder.
- `src/actions/*` — app-owned action functions, Mixamo retargeting, and
  renderer-specific dispatch.
- `src/bindings/three-vrm-binding.ts` — binds `ActionBus` events onto VRM
  clips, expressions, and lookAt.
- `src/components/AvatarCanvas.tsx` — mounts the scene and wires resize.
- `src/components/ChatComposer.tsx` — prompt composer.
- `src/components/TranscriptDrawer.tsx` — conversation log.
- `src/components/ControlsPanel.tsx` — character/model URL inputs and reset.

---

## Known limitations

- **Text-only interaction.** The example intentionally omits speech input,
  speech output, and lipsync while the core chat and action loop is kept
  small and stable.
- **Mixamo-only body clips in v1.** Clip-backed body gestures currently assume
  Mixamo `.fbx` source files and the upstream `three-vrm` retargeting path.
- **Memory is a plain sliding window.** No summarisation, no vector recall.
  Good enough for short sessions; will forget aggressively across long
  ones.

---

## Troubleshooting

- **"character.json HTTP 404"** — the file must be served by Vite; default
  is `/characters/aria/character.json` resolved from `public/`.
- **"Missing required render asset …"** — `avatar.vrm` or one of the required
  Mixamo `.fbx` clips is missing from the character folder.
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
