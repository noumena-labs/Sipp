import type { EngineExecutionMode } from '../core/inference-types.js';

export interface TransportObservability {
  executionMode: EngineExecutionMode;
  workerBacked: boolean;
  enabled: boolean;
  bufferedTokenLimit: number;
  flushIntervalMs: number;
  flushCount: number;
  coalescedTokenCount: number;
  maxObservedBufferedTokenCount: number;
  activeTokenTransport?: 'none' | 'runtime-events';
  runtimeEventDrainCount?: number;
  runtimeEventTokenCount?: number;
  runtimeEventTerminalCount?: number;
  runtimeEventTextBytes?: number;
  schedulerProgressCount?: number;
  schedulerProgressMs?: number;
  runtimeEventDrainMs?: number;
  tokenCallbackCount?: number;
  tokenCallbackMs?: number;
  pumpStepCount?: number;
  pumpStepMs?: number;
  schedulerYieldCount?: number;
  schedulerYieldMs?: number;
}
