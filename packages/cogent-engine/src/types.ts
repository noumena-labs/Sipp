export type FlashAttentionMode = 'auto' | 'enabled' | 'disabled';
export type PromptFormatMode = 'auto-chat' | 'raw';

export interface InferenceInitConfig {
  nCtx?: number;
  nBatch?: number;
  nUbatch?: number;
  nSeqMax?: number;
  nThreads?: number;
  nThreadsBatch?: number;
  nGpuLayers?: number;
  flashAttention?: FlashAttentionMode;
  kvUnified?: boolean;
  maxCachedSessions?: number;
  retainedPrefixTokens?: number;
}

export interface PromptGenerationOptions {
  nTokens?: number;
  promptFormat?: PromptFormatMode;
}

export interface PromptStreamOptions extends PromptGenerationOptions {
  onToken?: (token: string) => void;
}

export interface PromptPerformanceStats {
  totalMs: number;
  promptEvalMs: number;
  decodeEvalMs: number;
  sampleMs: number;
  inputTokenCount: number;
  promptEvalTokens: number;
  decodeEvalCount: number;
  sampleCount: number;
  outputTokenCount: number;
}
