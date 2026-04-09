export interface EngineModuleOptions {
  locateFile?: (path: string, prefix?: string) => string;
  [key: string]: unknown;
}

export interface CogentConfig {
  moduleUrl?: string;
  wasmUrl?: string;
  moduleOptions?: EngineModuleOptions;
  maxModelBytes?: number;
  trustedOrigins?: string[];
  executionMode?: 'auto' | 'worker' | 'main-thread';
  workerUrl?: string;
  workerMaxBufferedTokens?: number;
  workerTokenFlushIntervalMs?: number;
  persistentModelCache?: {
    enabled?: boolean;
  };
}
