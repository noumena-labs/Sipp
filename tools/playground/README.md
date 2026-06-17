# Browser Playground

`tools/playground` is the browser runtime playground for the Sipp browser
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

- `window.__sippPlayground.getEnvironment()`
- `window.__sippPlayground.getRuntimeObservability()`
- `window.__sippPlayground.getBackendObservability()`
- `window.__sippPlayground.getLastReport()`

This API is used by the automated browser playground smoke runner.

See [../../docs/en/examples-demos.md](../../docs/en/examples-demos.md) for where the
playground fits alongside examples and demos.
