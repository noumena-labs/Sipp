# Browser Caching

Browser-local inference caches model data in browser storage so repeated loads
avoid full network downloads when the runtime supports that path. Sipp
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
- Close `SippClient` instances when a page, worker, or component no longer
  needs local runtime resources.

## Interrupted Downloads

Remote model downloads keep successfully written OPFS bytes when the network
stalls or a retryable request fails. A later attempt resumes with `Range` and
`If-Range` when the server exposes an ETag or last-modified validator. If the
server rejects or ignores the range, Sipp discards the partial file and retries
the request from the beginning. Partial downloads survive page reloads and are
removed when they become invalid or remain unused for seven days.

Set the per-chunk stall deadline when adding a remote model. The default is 30
seconds:

```ts
const model = await client.models.add(['/models/model.gguf'], {
  stallTimeoutMs: 30_000,
});
```

Applications can observe a range fallback without intercepting console output:

```ts
const unsubscribe = client.subscribeEvents((event) => {
  if (event.type === 'fallback-warning' && event.kind === 'transfer') {
    reportDownloadFallback(event.detail);
  }
});

// Stop observing when the owning view or worker is disposed.
unsubscribe();
```

Use the browser examples for minimal flows and the playground for runtime
diagnostics.
