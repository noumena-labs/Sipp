# Browser Caching

Browser-local inference caches model data in browser storage so repeated loads
avoid full network downloads when the runtime supports that path. CogentLM
browser examples and demos use this path for GGUF model loading.

## Responsibilities

The browser package owns runtime integration and cache mechanics. Applications
still own:

- The model URL or file selection UI.
- Progress display and cancellation behavior.
- Storage-clearing controls when users need to reclaim space.
- Fallback behavior when browser storage is unavailable.

## Practical Guidance

- Prefer model URLs that support range requests for large assets.
- Keep default demo models small enough for first-run onboarding.
- Treat browser storage as user-controlled and best-effort.
- Close `CogentClient` instances when a page, worker, or component no longer
  needs local runtime resources.

Use the browser examples for minimal flows and the playground for runtime
diagnostics.
