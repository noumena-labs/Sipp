import type { EngineExecutionMode } from '../core/inference-types.js';

// JS-side counters describing how tokens are crossing the worker/main
// boundary.  Streaming is exclusively SAB-ring-based.
export interface TransportObservability {
  executionMode: EngineExecutionMode;
  workerBacked: boolean;
  enabled: boolean;
  // 'none' when no streaming consumer is attached (engine in NONE mode).
  activeTokenTransport?: 'none' | 'streaming-buffer';
  // Cumulative ms spent inside `_ce_yield_drain` (SAB ring writes from the
  // native streaming scratch buffer).
  streamingDrainCount?: number;
  streamingDrainMs?: number;
}
