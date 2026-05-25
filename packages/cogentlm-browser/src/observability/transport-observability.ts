import type { EngineExecutionMode } from '../core/inference-types.js';

// JS-side counters describing how tokens are crossing the worker/main
// boundary.  SAB is the fast path; callback/postMessage batches are the
// portable fallback.
export interface TransportObservability {
  executionMode: EngineExecutionMode;
  workerBacked: boolean;
  enabled: boolean;
  // 'none' when no streaming consumer is attached (engine in NONE mode).
  activeTokenTransport?: 'none' | 'streaming-buffer' | 'callback';
  // Cumulative ms spent inside `_ce_yield_drain` (SAB ring writes from the
  // native streaming scratch buffer).
  streamingDrainCount?: number;
  streamingDrainMs?: number;
}
