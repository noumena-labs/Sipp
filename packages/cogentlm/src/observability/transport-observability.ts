import type { EngineExecutionMode } from '../core/inference-types.js';

export interface TransportObservability {
  executionMode: EngineExecutionMode;
  workerBacked: boolean;
  enabled: boolean;
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
  schedulerYieldCount?: number;
  schedulerYieldMs?: number;
}
