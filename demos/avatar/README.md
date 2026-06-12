# Avatar Demo

`demos/avatar` is a React and three.js character demo showing how to pair
`SippClient` with the browser package character helpers and a VRM character.

## Run

```bash
cargo xtask run demos serve avatar
```

Open the printed local URL and press `Start`. The start screen pre-populates a
small default `.gguf` model URL that can be replaced before entering the demo.

## What It Demonstrates

- Character config loading from `public/characters/aria/character.json`.
- Local model startup owned by the app.
- VRM and FBX animation asset resolution by folder convention.
- Manual action controls and prompt chips.
- Character event bus integration with scene effects.

The character config is demo-owned. New character authoring puts durable
steering in `persona.anchorExamples`, conversational examples in
`persona.dialogExamples`, and style notes in `persona.notes`.

See `lib/web/src/character` for the character runtime APIs and
[../../docs/examples-demos.md](../../docs/examples-demos.md) for the demo
index.
