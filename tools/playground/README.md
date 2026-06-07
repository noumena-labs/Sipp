# Browser Playground

`tools/playground` is the browser runtime playground for the CogentLM browser
package. It exposes runtime diagnostics, browser environment details, backend
observability, report export, and repeatable measurement runs.

## Run

From the repository root:

```bash
cargo xtask run tools serve playground
```

For a production build:

```bash
cargo xtask run tools build playground
```

For the automated playground runtime smoke:

```bash
cargo xtask test smoke suite playground-browser
```

## Automation API

The page exposes a stable automation API:

- `window.__cogentPlayground.getEnvironment()`
- `window.__cogentPlayground.getRuntimeObservability()`
- `window.__cogentPlayground.getBackendObservability()`
- `window.__cogentPlayground.getRuntimeSmoke()`
- `window.__cogentPlayground.runRuntimeSmoke()`
- `window.__cogentPlayground.getLastReport()`

This API is used by the automated browser playground smoke runner.

See [../../docs/examples-demos.md](../../docs/examples-demos.md) for where the
playground fits alongside examples and demos.
