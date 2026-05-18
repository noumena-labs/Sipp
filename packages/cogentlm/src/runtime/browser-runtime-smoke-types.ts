import type { BackendObservability } from '../types.js';

export interface BrowserGgufIngestSmokeResult {
  available: boolean;
  layoutForLargeFile: 'single-file' | 'split-gguf' | null;
  plannedShardCount: number | null;
  streamedShardCount: number;
  streamedBytes: number;
  error: string | null;
}

export interface BrowserRustEngineSmokeResult {
  available: boolean;
  abiVersion: number;
  engineId: number | null;
  error: string | null;
}

export interface BrowserRuntimeSmokeResult {
  rustEngine: BrowserRustEngineSmokeResult;
  ggufIngest: BrowserGgufIngestSmokeResult;
  backend: BackendObservability | null;
  webgpuReady: boolean;
}
