# apps/avatar

Small React + three.js example showing how to pair `CogentEngine` with
`cogent-engine/character` and a VRM avatar.

The app keeps responsibilities split cleanly:

- `character.json` defines persona, actions, and memory
- the app chooses the model URL and owns engine startup
- the app resolves `avatar.vrm` and animation clips by folder convention

## Quick start

From the repo root:

```bash
bun install
bun run build:wasm
bun run avatar:dev
```

Open the printed local URL, paste a `.gguf` model URL, and press `Load`.

## What the app loads

- Character config: `public/characters/aria/character.json`
- Avatar model: `public/characters/aria/avatar.vrm`
- Idle clip: `public/characters/aria/animations/idle.fbx`
- Action clips: `public/characters/aria/animations/<action>.fbx`

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

See `packages/cogent-engine/docs/character-harness.md` for the full harness API
and config shape.
