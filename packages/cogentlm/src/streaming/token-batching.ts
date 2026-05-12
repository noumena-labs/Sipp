// Helpers for consuming streamed tokens without forcing per-token DOM
// reflow on the main thread.
//
// Background: in worker mode the engine writes tokens to a SAB ring at
// native decode rate.  The main thread polls the ring on each animation
// frame and fires `onToken` for every drained record.  If the user's
// `onToken` does GPU-touching work per call (innerText, canvas paint,
// etc.) it competes with the worker's WebGPU compute on the shared
// physical GPU, inflating native `decode_ms` and dropping reported TPS.
//
// The fix is to keep `onToken` itself cheap (CPU-only mutation of an
// accumulator) and rate-limit the rendering callback to at most once per
// animation frame.  These helpers wrap that pattern so users don't have
// to re-implement the rAF coalescing.

/** Payload delivered to a frame-batched render callback. */
export interface BatchedTokens {
  /** Full text accumulated so far, including this batch. */
  accumulated: string;
  /** Concatenation of tokens that arrived since the last flush. */
  delta: string;
}

export interface BatchTokensByFrameOptions {
  /**
   * Throttle renders to at most one per `intervalMs` milliseconds, using
   * `setTimeout` as the scheduler.  Useful when 60 Hz rAF rendering is
   * still pulling enough GPU compositor work to dent native decode TPS —
   * dropping to ~30 Hz (33 ms) or ~20 Hz (50 ms) cuts compositor frame
   * count proportionally and is visually indistinguishable for token
   * streaming.  Mutually exclusive with `schedule`.
   */
  intervalMs?: number;
  /**
   * Custom scheduler.  Defaults to `requestAnimationFrame` in browsers
   * and a 16 ms `setTimeout` otherwise.  Override for tests or to plug
   * in a non-rAF scheduler.  Must invoke `cb` exactly once when the
   * next tick is due and return a cancellation handle.
   */
  schedule?: (cb: () => void) => unknown;
  /** Matching cancellation for the custom scheduler. */
  cancel?: (handle: unknown) => void;
}

/**
 * Wraps a render callback with frame-rate coalescing for use as an
 * `onToken` handler.  Returns an object with:
 *
 * - `onToken(token)`: the SDK-facing handler.  Cheap; only appends.
 * - `flush()`: forces an immediate render of whatever is buffered.  Call
 *   this when the chat promise resolves so the tail tokens land on
 *   screen even if no animation frame fired between the last token and
 *   completion.
 *
 * The render callback receives the FULL accumulated text plus the
 * latest delta.  Use the accumulated form for `innerText` / `textContent`
 * to keep the DOM mutation O(1) per frame instead of O(N²).
 *
 * Example:
 *
 *     const stream = batchTokensByFrame(({ accumulated }) => {
 *       outputRef.current!.textContent = accumulated;
 *     });
 *     await engine.chat(messages, { onToken: stream.onToken });
 *     stream.flush();
 */
export function batchTokensByFrame(
  render: (batch: BatchedTokens) => void,
  options: BatchTokensByFrameOptions = {}
): {
  onToken: (token: string) => void;
  flush: () => void;
} {
  // Selection order: explicit `schedule` wins; otherwise `intervalMs` picks
  // a setTimeout-based throttle; otherwise rAF; otherwise a 16ms fallback.
  const intervalMs = options.intervalMs;
  const useInterval =
    options.schedule === undefined &&
    intervalMs !== undefined &&
    Number.isFinite(intervalMs) &&
    intervalMs >= 0;
  const useRAF =
    options.schedule === undefined &&
    !useInterval &&
    typeof requestAnimationFrame === 'function';
  const schedule =
    options.schedule ??
    (useInterval
      ? (cb: () => void) => setTimeout(cb, intervalMs as number)
      : useRAF
        ? (cb: () => void) => requestAnimationFrame(cb)
        : (cb: () => void) => setTimeout(cb, 16));
  const cancel =
    options.cancel ??
    (useRAF
      ? (handle: unknown) => cancelAnimationFrame(handle as number)
      : (handle: unknown) =>
          clearTimeout(handle as ReturnType<typeof setTimeout>));

  let accumulated = '';
  let pendingDelta = '';
  let scheduled: unknown = null;

  const flush = (): void => {
    if (scheduled != null) {
      cancel(scheduled);
      scheduled = null;
    }
    if (pendingDelta.length === 0) {
      return;
    }
    const delta = pendingDelta;
    pendingDelta = '';
    render({ accumulated, delta });
  };

  const onToken = (token: string): void => {
    if (token.length === 0) {
      return;
    }
    pendingDelta += token;
    accumulated += token;
    if (scheduled != null) {
      return;
    }
    scheduled = schedule(() => {
      scheduled = null;
      const delta = pendingDelta;
      if (delta.length === 0) {
        return;
      }
      pendingDelta = '';
      render({ accumulated, delta });
    });
  };

  return { onToken, flush };
}
