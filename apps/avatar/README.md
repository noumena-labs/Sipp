# apps/avatar

Small React + three.js high-fantasy avatar demo showing how to pair
`CogentEngine` with `@noumena-labs/cogentlm-browser/character` and a VRM character.

The app keeps responsibilities split cleanly:

- `character.json` defines persona, actions, and memory
- the app chooses the model URL and owns engine startup
- the app resolves `avatar.vrm` and animation clips by folder convention
- the app renders the fantasy stage, manual action controls, prompt chips, and procedural magic effects

## Quick start

From the repo root:

```bash
bun install
bun run build:wasm
bun run avatar:dev
```

Open the printed local URL and press `Start`. The start screen pre-populates a
small default `.gguf` model URL that can be replaced before entering the demo.

## What the app loads

- Character config: `public/characters/aria/character.json`
- Avatar model: `public/characters/aria/avatar.vrm`
- Idle clip: `public/characters/aria/animations/idle.fbx`
- Action clips: `public/characters/aria/animations/<action>.fbx`
- Default model URL: `https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q4_K_M.gguf`

## Fantasy Demo

Aria is authored as a Dawnblade knight guarding the Starfall Gate. The stage is
an asset-free three.js fantasy ruin with a stone dais, rune rings, crystals,
broken pillars, fog, and drifting magic motes.

The demo exposes three interaction layers:

- chat, driven by the loaded model and `character.json`
- suggested prompt chips for quick user onboarding
- a manual Actions panel that dispatches action events directly through the character event bus

The character config is intentionally internal to this demo. The setup UI only
exposes the model URL.

## Action Coverage

Aria's action schema includes body emotes, facial expressions, gaze targets, and
procedural magic effects. Every configured action is expected to produce a
visible result.

Existing FBX emotes:

- `wave`
- `nod`
- `shake_head`
- `salute`
- `thinking`
- `bashful`
- `excited`
- `happy_blissful`
- `joy_jump`
- `upset_angry`
- `crying`
- `sad_idle`

Code-driven expressions and gaze:

- `smile`
- `look_sad`
- `gasp`
- `look_angry`
- `settle`
- `look_at_you`
- `glance_left`
- `glance_right`
- `look_up`
- `look_down`

Procedural fantasy effects:

- `summon_familiar`
- `cast_starbolt`
- `raise_ward`
- `summon_rune_circle`

## Character flow

`src/App.tsx` uses the public helper:

```ts
const { character, config } = await createCharacterFromConfigUrl({
  configUrl: args.characterUrl,
  engine,
  bus: new CharacterEventBus(),
});
```

The app still owns:

- engine creation
- model download / init
- render asset validation
- scene bindings

## character.json notes

Aria-specific style rules now live in `persona.notes`, not in the engine's
shared prompt renderer. That is the intended authoring pattern for new
characters too.

Use `persona.anchorExamples` for durable steering and `persona.dialogExamples`
for conversational flow examples.

See `packages/cogentlm-browser/docs/character-harness.md` for the full harness API
and config shape.
