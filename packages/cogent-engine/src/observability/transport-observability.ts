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
  tokenTransportPreference?: 'auto' | 'runtime-events';
  activeTokenTransport?: 'none' | 'callbacks' | 'runtime-events';
  tokenCallbackRegistrationCount?: number;
  nativeCallbackTokenCount?: number;
  runtimeEventDrainCount?: number;
  runtimeEventTokenCount?: number;
  runtimeEventTerminalCount?: number;
  runtimeEventTextBytes?: number;
}
