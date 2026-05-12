import type { EngineExecutionMode } from '../core/inference-types.js';

export interface TransportObservability {
  executionMode: EngineExecutionMode;
  workerBacked: boolean;
  enabled: boolean;
  activeTokenTransport?: 'none' | 'runtime-events' | 'streaming-buffer' | 'direct-callback';
  runtimeEventDrainCount?: number;
  runtimeEventTokenCount?: number;
  runtimeEventTerminalCount?: number;
  runtimeEventTextBytes?: number;
  // Cumulative ms spent inside bridge.runInferenceLoop() calls (wasm-resident
  // wall time observed from JS) and the matching invocation count.
  schedulerProgressCount?: number;
  schedulerProgressMs?: number;
  // Cumulative ms spent in drainRuntimeEvents (JS-side queue drain, distinct
  // from the streaming-buffer drain hook).
  runtimeEventDrainMs?: number;
  // Cumulative ms spent invoking user `onToken` thunks (DirectCallback path
  // or RuntimeEvents replay).  Tracks bridge-jitter contribution.
  tokenCallbackCount?: number;
  tokenCallbackMs?: number;
  // Cumulative ms spent in scheduler-pump idle waits (not currently used;
  // retained for future yield primitives).
  schedulerYieldCount?: number;
  schedulerYieldMs?: number;
  // Cumulative ms spent inside the `_ce_yield_drain` hook (SAB ring writes
  // from the native streaming scratch buffer).  This time also shows up on
  // the native side as part of `yieldWaitMs`, so it is the "what JS did
  // during the yield" attribution.
  streamingDrainCount?: number;
  streamingDrainMs?: number;
}
