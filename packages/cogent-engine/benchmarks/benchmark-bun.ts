import { existsSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { CogentEngine, getBundledBunRuntimeUrls } from '../dist/esm/index.js';
import type {
  BackendObservability,
  FlashAttentionMode,
  InferenceInitConfig,
  PromptFormatMode,
  RuntimeObservabilityMetrics,
  SchedulerPolicyMode,
} from '../src/types.js';

// Benchmark metric glossary:
// - TTFT: time to first token. Measured from request start until the first streamed token callback.
// - TPOT: time per output token after the first token. For one request:
//         (E2EL - TTFT) / (output_tokens - 1).
// - ITL: inter-token latency. The actual token-to-token gaps observed between streamed token callbacks.
// - E2EL: end-to-end latency. Request start until the final streamed token has been received.
// - Request throughput: successful requests / benchmark duration.
// - Output token throughput: generated output tokens / benchmark duration.
// - Total token throughput: (logical input tokens + output tokens) / benchmark duration.
// - Logical input tokens: full prompt token count for the request after prompt formatting.
// - Effective prompt-eval tokens: tokens the model actually evaluated during prefill.
//   This can be lower than logical input tokens when KV/prefix reuse is hit.
// - promptEvalMs: native llama.cpp prefill time.
// - decodeEvalMs: native llama.cpp decode time for generated tokens.
// - sampleMs: native sampler-chain time spent choosing output tokens from logits.

type BenchmarkPresetName = 'default' | 'single';
type PromptBucket = 'short' | 'medium' | 'long' | 'custom';
type OutputBucket = 'short' | 'medium' | 'long';

interface BenchmarkOptions {
  modelPath: string;
  preset: BenchmarkPresetName;
  prompt?: string;
  tokensOverride?: number;
  warmupRuns: number;
  measuredRuns: number;
  cancelChurnRuns: number;
  cancelChurnTokens: number;
  jsonPath?: string;
  artifactLabel?: string;
  quantizationLabel?: string;
  promptFormat: PromptFormatMode;
  initConfig: InferenceInitConfig;
}

interface BenchmarkSummary {
  meanMs: number;
  medianMs: number;
  p99Ms: number;
  minMs: number;
  maxMs: number;
}

interface BenchmarkRun {
  label: string;
  contextKey: string;
  // E2EL for a single request.
  wallMs: number;
  // TTFT for a single request.
  ttftMs: number | null;
  // TPOT for a single request.
  tpotMs: number | null;
  // All observed inter-token gaps for this request.
  itlMsValues: number[];
  // Logical input size after prompt formatting/tokenization.
  inputTokenCount: number | null;
  // Native effective prefill work reported by llama.cpp perf counters.
  promptEvalTokenCount: number | null;
  // Generated output token count for this request.
  outputTokenCount: number | null;
  outputLength: number;
  outputPreview: string;
  runtimeObservability: RuntimeObservabilityMetrics | null;
}

interface ServingBenchmarkSummary {
  successfulRequests: number;
  benchmarkDurationMs: number;
  totalInputTokens: number;
  totalGeneratedTokens: number;
  // Aggregate metrics aligned with TensorRT-LLM style reporting.
  requestThroughputRps: number | null;
  outputTokenThroughputTps: number | null;
  totalTokenThroughputTps: number | null;
  ttftMs: BenchmarkSummary | null;
  tpotMs: BenchmarkSummary | null;
  itlMs: BenchmarkSummary | null;
  e2elMs: BenchmarkSummary;
}

interface RuntimeBenchmarkSummary {
  // Runtime-side diagnostics. These are useful for root-cause analysis but are not
  // the main serving metrics because they exclude JS/Wasm/host overhead.
  avgLogicalInputTokenCount: number | null;
  avgPromptEvalTokens: number | null;
  avgTotalMs: number | null;
  avgPromptEvalMs: number | null;
  avgDecodeEvalMs: number | null;
  avgSampleMs: number | null;
  avgOutputTokenCount: number | null;
  avgQueueDelayMs: number | null;
  avgTailItlMs: number | null;
  avgBatchParticipationCount: number | null;
  avgDecodeFirstTickCount: number | null;
  avgChunkedPrefillTickCount: number | null;
  avgMixedWorkloadTickCount: number | null;
  avgLcpReuseTokens: number | null;
  avgPrefixCacheRestoreTokens: number | null;
  avgPrefixCacheHitCount: number | null;
  avgPrefixCacheStoreCount: number | null;
  promptEvalTokensPerSecond: number | null;
  outputTokensPerSecond: number | null;
}

interface DerivedBenchmarkSummary {
  avgHostOverheadMs: number | null;
  avgPromptReuseTokens: number | null;
  avgPromptReuseRatio: number | null;
}

interface BenchmarkGroupSummary {
  serving: ServingBenchmarkSummary;
  runtime: RuntimeBenchmarkSummary;
  derived: DerivedBenchmarkSummary;
}

interface MemorySnapshotSummary {
  before: RuntimeMemoryUsage;
  after: RuntimeMemoryUsage;
  delta: RuntimeMemoryUsage;
}

interface BenchmarkGroupResult {
  id: 'coldPrompt' | 'hotFreshContext' | 'hotReuseContext';
  label: string;
  warmupRuns: number;
  measuredRuns: number;
  benchmarkDurationMs: number;
  summary: BenchmarkGroupSummary;
  memory: MemorySnapshotSummary | null;
  runs: BenchmarkRun[];
}

interface ScenarioRuntimeMetrics {
  initEngineMs: number;
}

interface BenchmarkScenarioDefinition {
  id: string;
  label: string;
  prompt: string;
  promptBucket: PromptBucket;
  promptChars: number;
  promptWords: number;
  outputTokenLimit: number;
  outputBucket: OutputBucket;
  promptFormat: PromptFormatMode;
  contextBucket: 'single-request';
  concurrency: 1;
}

interface BenchmarkScenarioResult {
  definition: BenchmarkScenarioDefinition;
  runtime: ScenarioRuntimeMetrics;
  coldPrompt: BenchmarkGroupResult;
  hotFreshContext: BenchmarkGroupResult;
  hotReuseContext: BenchmarkGroupResult;
}

interface MixedLoadBenchmarkDefinition {
  id: 'mixed-lilo-vs-siso';
  label: string;
  background: BenchmarkScenarioDefinition;
  foreground: BenchmarkScenarioDefinition;
  concurrency: 2;
}

interface MixedLoadBenchmarkResult {
  definition: MixedLoadBenchmarkDefinition;
  runtime: ScenarioRuntimeMetrics;
  background: BenchmarkGroupResult;
  foreground: BenchmarkGroupResult;
}

interface RuntimeMemoryUsage {
  rssBytes: number;
  heapUsedBytes: number;
  externalBytes: number;
  arrayBuffersBytes: number;
}

interface QueueCancelChurnResult {
  iterations: number;
  tokenLimit: number;
  benchmarkDurationMs: number;
  enqueueLatencyMs: BenchmarkSummary | null;
  cancelLatencyMs: BenchmarkSummary | null;
  cancelledCount: number;
  memory: MemorySnapshotSummary;
  smokePrompt: {
    wallMs: number | null;
    outputLength: number;
    runtimeObservabilityAvailable: boolean;
    failed: boolean;
    errorMessage: string | null;
  };
  warnings: string[];
}

interface BenchmarkReport {
  schemaVersion: 'cogent.benchmark.bun.v8';
  generatedAt: string;
  benchmark: {
    script: string;
    preset: BenchmarkPresetName;
    promptFormat: PromptFormatMode;
    warmupRuns: number;
    measuredRuns: number;
    cancelChurnRuns: number;
    cancelChurnTokens: number;
    scenarioCount: number;
  };
  environment: {
    runtimeKind: 'bun';
    bunVersion: string;
    nodeCompatVersion: string;
    platform: string;
    arch: string;
    userAgent: string | null;
    hasNavigatorGpu: boolean;
    adapterAvailable: false;
    adapterLabel: null;
  };
  model: {
    modelPath: string;
    fileName: string;
    artifactLabel: string;
    quantizationLabel: string | null;
    modelBytes: number;
  };
  backend: BenchmarkBackendProfile;
  runtime: {
    initConfig: InferenceInitConfig;
    readModelMs: number;
    initModuleMs: number;
    loadModelIntoMemfsMs: number;
    initEngineSummary: {
      initEngineMs: BenchmarkSummary;
    };
  };
  memory: RuntimeMemoryUsage;
  scenarios: BenchmarkScenarioResult[];
  mixedLoad: MixedLoadBenchmarkResult | null;
  queueCancelChurn: QueueCancelChurnResult | null;
  warnings: string[];
  limitations: string[];
}

type RequestedExecutionMode = 'cpu-only' | 'gpu-offload';
type InferredExecutionBackend = 'cpu' | 'webgpu' | 'unknown';
type RuntimeBackendStatus =
  | 'not-compiled'
  | 'compiled-not-registered'
  | 'registered-no-devices'
  | 'webgpu-ready'
  | 'unknown';

interface BenchmarkBackendProfile {
  // Requested mode comes from benchmark initConfig. Inferred backend is conservative:
  // it reports the execution path the runtime environment makes plausible today,
  // not an exact layer-by-layer placement report inside llama.cpp.
  requestedExecutionMode: RequestedExecutionMode;
  requestedGpuLayers: number | null;
  inferredExecutionBackend: InferredExecutionBackend;
  runtimeBackendStatus: RuntimeBackendStatus;
  gpuOffloadSupported: boolean | null;
  availableBackends: string[];
  backendRegistries: BackendObservability['availableBackends'];
  runtimeDeviceCount: number;
  runtimeAcceleratorDeviceCount: number;
  runtimeDeviceLabels: string[];
  runtimeDevices: BackendObservability['devices'];
  hostAdapter: {
    apiAvailable: boolean;
    adapterAvailable: boolean;
    adapterLabel: string | null;
  };
  notes: string[];
}

const SHORT_PROMPT = 'Write one sentence about measuring inference performance.';
const LONG_PROMPT = [
  'You are evaluating a browser-hosted inference runtime built with TypeScript, WebAssembly, and llama.cpp.',
  'Describe how you would benchmark cold start, module initialization, model load, engine initialization, prompt evaluation throughput, decode throughput, reused-context performance, and TTFT.',
  'Keep the answer concise but explain why prompt length and output length should be swept separately.',
].join(' ');

const DEFAULT_SHORT_OUTPUT_TOKENS = 16;
const DEFAULT_LONG_OUTPUT_TOKENS = 128;
const DEFAULT_WARMUP_RUNS = 1;
const DEFAULT_MEASURED_RUNS = 3;
const DEFAULT_CANCEL_CHURN_RUNS = 250;
const DEFAULT_CANCEL_CHURN_TOKENS = 16;
const CANCEL_CHURN_MEMORY_WARNING_BYTES = 8 * 1024 * 1024;
const OUTPUT_PREVIEW_LIMIT = 120;
const QUANTIZATION_SUFFIX_PATTERN =
  /(?:[-_.])((?:IQ\d+(?:_[A-Z0-9]+)*)|(?:Q\d+(?:_[A-Z0-9]+)*)|(?:BF16|F16|F32|FP16|FP32))$/i;

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.resolve(scriptDir, '..');
const repoRoot = path.resolve(packageRoot, '..', '..');

const DEFAULT_PRESET_CASES = [
  {
    id: 'siso',
    label: 'Short Input / Short Output',
    prompt: SHORT_PROMPT,
    outputTokenLimit: DEFAULT_SHORT_OUTPUT_TOKENS,
  },
  {
    id: 'silo',
    label: 'Short Input / Long Output',
    prompt: SHORT_PROMPT,
    outputTokenLimit: DEFAULT_LONG_OUTPUT_TOKENS,
  },
  {
    id: 'liso',
    label: 'Long Input / Short Output',
    prompt: LONG_PROMPT,
    outputTokenLimit: DEFAULT_SHORT_OUTPUT_TOKENS,
  },
  {
    id: 'lilo',
    label: 'Long Input / Long Output',
    prompt: LONG_PROMPT,
    outputTokenLimit: DEFAULT_LONG_OUTPUT_TOKENS,
  },
] as const;

function nowMs(): number {
  return performance.now();
}

function round(value: number): number {
  return Number(value.toFixed(3));
}

function formatBytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
}

function parseOptionalPositiveInt(flagName: string, rawValue: string | undefined): number | undefined {
  if (rawValue == null) {
    return undefined;
  }

  const value = Number.parseInt(rawValue, 10);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Expected a positive integer for ${flagName}, got "${rawValue}".`);
  }
  return value;
}

function parseOptionalNonNegativeInt(flagName: string, rawValue: string | undefined): number | undefined {
  if (rawValue == null) {
    return undefined;
  }

  const value = Number.parseInt(rawValue, 10);
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`Expected a non-negative integer for ${flagName}, got "${rawValue}".`);
  }
  return value;
}

function parsePositiveInt(flagName: string, rawValue: string | undefined, fallback: number): number {
  return parseOptionalPositiveInt(flagName, rawValue) ?? fallback;
}

function parseNonNegativeInt(flagName: string, rawValue: string | undefined, fallback: number): number {
  return parseOptionalNonNegativeInt(flagName, rawValue) ?? fallback;
}

function parsePreset(rawPreset: string | undefined): BenchmarkPresetName {
  const preset = (rawPreset ?? 'default').trim();
  if (preset === 'default' || preset === 'single') {
    return preset;
  }
  throw new Error(`Unsupported preset "${preset}". Use "default" or "single".`);
}

function parsePromptFormat(rawValue: string | undefined): PromptFormatMode {
  const value = (rawValue ?? 'auto-chat').trim();
  if (value === 'auto-chat' || value === 'raw') {
    return value;
  }
  throw new Error(`Unsupported prompt format "${value}". Use "auto-chat" or "raw".`);
}

function parseFlashAttention(rawValue: string | undefined): FlashAttentionMode | undefined {
  if (rawValue == null) {
    return undefined;
  }
  const value = rawValue.trim();
  if (value === 'auto' || value === 'enabled' || value === 'disabled') {
    return value;
  }
  throw new Error(`Unsupported flash attention mode "${value}". Use "auto", "enabled", or "disabled".`);
}

function parseSchedulerPolicy(rawValue: string | undefined): SchedulerPolicyMode | undefined {
  if (rawValue == null) {
    return undefined;
  }
  const value = rawValue.trim();
  if (value === 'latency-first' || value === 'balanced' || value === 'throughput-first') {
    return value;
  }
  throw new Error(
    `Unsupported scheduler policy "${value}". Use "latency-first", "balanced", or "throughput-first".`
  );
}

function parseOptionalBoolean(flagName: string, rawValue: string | undefined): boolean | undefined {
  if (rawValue == null) {
    return undefined;
  }
  if (rawValue === 'true') {
    return true;
  }
  if (rawValue === 'false') {
    return false;
  }
  throw new Error(`Expected "true" or "false" for ${flagName}, got "${rawValue}".`);
}

function resolveModelPath(rawPath: string | undefined): string {
  const candidates = rawPath
    ? [rawPath]
    : [
        process.env.COGENT_BENCH_MODEL,
        path.resolve(process.cwd(), 'Qwen3.5-0.8B-Q4_0.gguf'),
        path.resolve(repoRoot, 'Qwen3.5-0.8B-Q4_0.gguf'),
      ];

  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }

    const resolved = path.isAbsolute(candidate) ? candidate : path.resolve(process.cwd(), candidate);
    if (existsSync(resolved)) {
      return resolved;
    }
  }

  throw new Error('No GGUF model found. Pass --model <path> or set COGENT_BENCH_MODEL.');
}

function parseArgs(argv: string[]): BenchmarkOptions {
  const options = new Map<string, string>();

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith('--')) {
      throw new Error(`Unexpected positional argument "${arg}". Use --help for usage.`);
    }

    const key = arg.slice(2);
    if (key === 'help') {
      printHelp();
      process.exit(0);
    }

    const value = argv[i + 1];
    if (!value || value.startsWith('--')) {
      throw new Error(`Missing value for --${key}.`);
    }

    options.set(key, value);
    i++;
  }

  const preset = parsePreset(options.get('preset'));
  const prompt = options.get('prompt');
  if (preset !== 'single' && prompt) {
    throw new Error('--prompt can only be used with --preset single.');
  }

  return {
    modelPath: resolveModelPath(options.get('model')),
    preset,
    prompt,
    tokensOverride: parseOptionalPositiveInt('--tokens', options.get('tokens')),
    warmupRuns: parseNonNegativeInt('--warmup', options.get('warmup'), DEFAULT_WARMUP_RUNS),
    measuredRuns: parsePositiveInt('--runs', options.get('runs'), DEFAULT_MEASURED_RUNS),
    cancelChurnRuns: parseNonNegativeInt(
      '--cancel-churn-runs',
      options.get('cancel-churn-runs'),
      DEFAULT_CANCEL_CHURN_RUNS
    ),
    cancelChurnTokens: parsePositiveInt(
      '--cancel-churn-tokens',
      options.get('cancel-churn-tokens'),
      DEFAULT_CANCEL_CHURN_TOKENS
    ),
    jsonPath: options.get('json'),
    artifactLabel: options.get('artifact-label'),
    quantizationLabel: options.get('quantization'),
    promptFormat: parsePromptFormat(options.get('prompt-format')),
    initConfig: {
      nCtx: parseOptionalPositiveInt('--ctx', options.get('ctx')),
      nBatch: parseOptionalPositiveInt('--batch', options.get('batch')),
      nUbatch: parseOptionalPositiveInt('--ubatch', options.get('ubatch')),
      nSeqMax: parseOptionalPositiveInt('--seq-max', options.get('seq-max')),
      nThreads: parseOptionalPositiveInt('--threads', options.get('threads')),
      nThreadsBatch: parseOptionalPositiveInt('--threads-batch', options.get('threads-batch')),
      nGpuLayers: parseOptionalNonNegativeInt('--gpu-layers', options.get('gpu-layers')),
      flashAttention: parseFlashAttention(options.get('flash-attention')),
      kvUnified: parseOptionalBoolean('--kv-unified', options.get('kv-unified')),
      maxCachedSessions: parseOptionalPositiveInt('--max-cached-sessions', options.get('max-cached-sessions')),
      retainedPrefixTokens: parseOptionalNonNegativeInt(
        '--retained-prefix-tokens',
        options.get('retained-prefix-tokens')
      ),
      prefillChunkSize: parseOptionalNonNegativeInt(
        '--prefill-chunk-size',
        options.get('prefill-chunk-size')
      ),
      schedulerPolicy: parseSchedulerPolicy(options.get('scheduler-policy')),
      decodeTokenReserve: parseOptionalNonNegativeInt(
        '--decode-token-reserve',
        options.get('decode-token-reserve')
      ),
      adaptivePrefillChunking: parseOptionalBoolean(
        '--adaptive-prefill-chunking',
        options.get('adaptive-prefill-chunking')
      ),
    },
  };
}

function printHelp(): void {
  console.log(`Usage: bun ./benchmarks/benchmark-bun.ts [options]

Options:
  --model <path>                   Path to a GGUF model file
  --preset <name>                  Benchmark preset: default | single (default: default)
  --prompt <text>                  Prompt text for --preset single
  --tokens <n>                     Max generation tokens per run or preset override
  --prompt-format <mode>           auto-chat | raw (default: auto-chat)
  --warmup <n>                     Warmup runs per benchmark group (default: ${DEFAULT_WARMUP_RUNS})
  --runs <n>                       Measured runs per benchmark group (default: ${DEFAULT_MEASURED_RUNS})
  --cancel-churn-runs <n>          Queue/cancel churn iterations after scenarios (default: ${DEFAULT_CANCEL_CHURN_RUNS})
  --cancel-churn-tokens <n>        Token limit used for churn requests and smoke prompt (default: ${DEFAULT_CANCEL_CHURN_TOKENS})
  --ctx <n>                        Optional llama context size
  --batch <n>                      Optional llama logical batch size
  --ubatch <n>                     Optional llama physical batch size
  --seq-max <n>                    Optional llama max sequence count
  --threads <n>                    Optional generation thread count
  --threads-batch <n>              Optional batch thread count
  --gpu-layers <n>                 Optional GPU layer count
  --flash-attention <mode>         auto | enabled | disabled
  --kv-unified <true|false>        Optional KV unified buffer setting
  --max-cached-sessions <n>        Optional session cache limit
  --retained-prefix-tokens <n>     Optional retained prefix tokens during KV trimming
  --prefill-chunk-size <n>         Optional per-slot prefill chunk size for the scheduler
  --scheduler-policy <mode>        latency-first | balanced | throughput-first
  --decode-token-reserve <n>       Optional decode token reservation per scheduler tick
  --adaptive-prefill-chunking      Optional adaptive prefill chunk sizing (true | false)
  --artifact-label <text>          Optional artifact label override
  --quantization <text>            Optional quantization label override
  --json <path>                    Optional JSON output path
  --help                           Show this message

Presets:
  default  Four benchmark quadrants: SISO, SILO, LISO, LILO.
  single   One prompt with cold, hot fresh-context, and hot reused-context groups.
`);
}

async function measureAsync<T>(fn: () => Promise<T> | T): Promise<{ ms: number; value: T }> {
  const start = nowMs();
  const value = await fn();
  return { ms: round(nowMs() - start), value };
}

function summarize(values: number[]): BenchmarkSummary {
  const sorted = [...values].sort((left, right) => left - right);
  const total = sorted.reduce((acc, value) => acc + value, 0);
  const percentileIndex = Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.99) - 1);

  return {
    meanMs: round(total / sorted.length),
    medianMs: round(sorted[Math.floor(sorted.length / 2)]),
    p99Ms: round(sorted[percentileIndex]),
    minMs: round(sorted[0]),
    maxMs: round(sorted[sorted.length - 1]),
  };
}

function summarizeOptional(values: Array<number | null>): BenchmarkSummary | null {
  const filtered = values.filter((value): value is number => value != null && Number.isFinite(value));
  return filtered.length === 0 ? null : summarize(filtered);
}

function averageRuntimeObservabilityMetric(
  observabilityRuns: Array<RuntimeObservabilityMetrics | null>,
  metric: (metrics: RuntimeObservabilityMetrics) => number
): number | null {
  const values = observabilityRuns
    .filter((metrics): metrics is RuntimeObservabilityMetrics => metrics !== null)
    .map(metric)
    .filter((value) => Number.isFinite(value) && value >= 0);

  if (values.length === 0) {
    return null;
  }

  const total = values.reduce((acc, value) => acc + value, 0);
  return round(total / values.length);
}

function summarizeThroughput(
  observabilityRuns: Array<RuntimeObservabilityMetrics | null>,
  metric: (metrics: RuntimeObservabilityMetrics) => number | null
): number | null {
  const values = observabilityRuns
    .filter((metrics): metrics is RuntimeObservabilityMetrics => metrics !== null)
    .map(metric)
    .filter((value): value is number => value != null && value > 0);

  if (values.length === 0) {
    return null;
  }

  const total = values.reduce((acc, value) => acc + value, 0);
  return round(total / values.length);
}

function promptTokensPerSecond(metrics: RuntimeObservabilityMetrics): number | null {
  // Effective prefill throughput based on llama.cpp perf counters, not end-to-end wall time.
  if (metrics.promptEvalMs <= 0 || metrics.promptEvalTokens <= 0) {
    return null;
  }
  return (metrics.promptEvalTokens * 1000) / metrics.promptEvalMs;
}

function decodeTokensPerSecond(metrics: RuntimeObservabilityMetrics): number | null {
  // Effective decode throughput based on llama.cpp perf counters, not end-to-end wall time.
  if (metrics.decodeEvalMs <= 0 || metrics.outputTokenCount <= 0) {
    return null;
  }
  return (metrics.outputTokenCount * 1000) / metrics.decodeEvalMs;
}

function captureMemoryUsage(): RuntimeMemoryUsage {
  const usage = process.memoryUsage();
  return {
    rssBytes: usage.rss,
    heapUsedBytes: usage.heapUsed,
    externalBytes: usage.external,
    arrayBuffersBytes: usage.arrayBuffers,
  };
}

function diffMemoryUsage(
  before: RuntimeMemoryUsage,
  after: RuntimeMemoryUsage
): RuntimeMemoryUsage {
  return {
    rssBytes: after.rssBytes - before.rssBytes,
    heapUsedBytes: after.heapUsedBytes - before.heapUsedBytes,
    externalBytes: after.externalBytes - before.externalBytes,
    arrayBuffersBytes: after.arrayBuffersBytes - before.arrayBuffersBytes,
  };
}

function averageBenchmarkRunMetric(
  runs: BenchmarkRun[],
  metric: (run: BenchmarkRun) => number | null
): number | null {
  const values = runs
    .map(metric)
    .filter((value): value is number => value != null && Number.isFinite(value));
  if (values.length === 0) {
    return null;
  }
  const total = values.reduce((acc, value) => acc + value, 0);
  return round(total / values.length);
}

function formatRuntimeDeviceLabel(device: BackendObservability['devices'][number]): string {
  const detail = device.description || device.name || device.backendName || device.type;
  return `${device.backendName}:${detail}`;
}

function inferRequestedExecutionMode(initConfig: InferenceInitConfig): RequestedExecutionMode {
  return initConfig.nGpuLayers === 0 ? 'cpu-only' : 'gpu-offload';
}

function inferRuntimeBackendStatus(runtimeBackend: BackendObservability | null): RuntimeBackendStatus {
  if (runtimeBackend == null) {
    return 'unknown';
  }
  if (!runtimeBackend.webgpuCompiled) {
    return 'not-compiled';
  }
  if (!runtimeBackend.webgpuRegistered) {
    return 'compiled-not-registered';
  }
  if (runtimeBackend.webgpuDeviceCount <= 0) {
    return 'registered-no-devices';
  }
  return 'webgpu-ready';
}

function inferExecutionBackend(
  requestedExecutionMode: RequestedExecutionMode,
  runtimeBackend: BackendObservability | null
): InferredExecutionBackend {
  if (runtimeBackend == null) {
    return 'unknown';
  }
  if (
    requestedExecutionMode === 'gpu-offload' &&
    runtimeBackend.webgpuRegistered &&
    runtimeBackend.webgpuDeviceCount > 0 &&
    runtimeBackend.gpuOffloadSupported
  ) {
    return 'webgpu';
  }
  return 'cpu';
}

function buildBenchmarkBackendProfile(
  runtimeBackend: BackendObservability | null,
  initConfig: InferenceInitConfig,
  hostAdapter: {
    apiAvailable: boolean;
    adapterAvailable: boolean;
    adapterLabel: string | null;
  }
): BenchmarkBackendProfile {
  const requestedExecutionMode = inferRequestedExecutionMode(initConfig);
  const runtimeDevices = runtimeBackend?.devices ?? [];
  const acceleratorDevices = runtimeDevices.filter((device) => device.type !== 'cpu');
  const notes: string[] = [];

  if (requestedExecutionMode === 'cpu-only') {
    notes.push('GPU offload was disabled explicitly by initConfig.nGpuLayers = 0.');
  } else if (!runtimeBackend?.webgpuCompiled) {
    notes.push('The package build did not include ggml-webgpu.');
  } else if (!runtimeBackend.webgpuRegistered) {
    notes.push('ggml-webgpu was compiled, but the runtime did not register a usable WebGPU backend.');
  } else if (runtimeBackend.webgpuDeviceCount <= 0) {
    notes.push('ggml-webgpu was registered, but it reported no runtime devices.');
  }

  if (runtimeDevices.length === 0) {
    notes.push('No ggml runtime devices were reported by the engine.');
  }

  return {
    requestedExecutionMode,
    requestedGpuLayers: initConfig.nGpuLayers ?? null,
    inferredExecutionBackend: inferExecutionBackend(requestedExecutionMode, runtimeBackend),
    runtimeBackendStatus: inferRuntimeBackendStatus(runtimeBackend),
    gpuOffloadSupported: runtimeBackend?.gpuOffloadSupported ?? null,
    availableBackends: runtimeBackend?.availableBackends.map((backend) => backend.name) ?? [],
    backendRegistries: runtimeBackend?.availableBackends ?? [],
    runtimeDeviceCount: runtimeDevices.length,
    runtimeAcceleratorDeviceCount: acceleratorDevices.length,
    runtimeDeviceLabels: runtimeDevices.map(formatRuntimeDeviceLabel),
    runtimeDevices,
    hostAdapter,
    notes,
  };
}

async function initializeScenarioEngine(
  runtimeUrls: ReturnType<typeof getBundledBunRuntimeUrls>,
  modelBytes: Uint8Array,
  fileName: string
): Promise<{
  engine: CogentEngine;
  modelPath: string;
  initModuleMs: number;
  loadModelIntoMemfsMs: number;
}> {
  const engine = new CogentEngine(runtimeUrls);

  try {
    // Load the Wasm module and copy the model into MEMFS once for the whole benchmark.
    // Recreating the module per scenario makes peak memory depend on host GC timing.
    const initModule = await measureAsync(() => engine.initModule());
    const loadModel = await measureAsync(() => engine.loadModelFromBuffer(modelBytes, fileName));

    return {
      engine,
      modelPath: loadModel.value,
      initModuleMs: initModule.ms,
      loadModelIntoMemfsMs: loadModel.ms,
    };
  } catch (error) {
    engine.close();
    throw error;
  }
}

async function reinitializeScenarioEngine(
  engine: CogentEngine,
  modelBytes: Uint8Array,
  fileName: string,
  initConfig: InferenceInitConfig
): Promise<ScenarioRuntimeMetrics> {
  // The native runtime releases loaded model files after init, so reinitialization
  // needs to restore the model into MEMFS before rebuilding engine state.
  const modelPath = engine.loadModelFromBuffer(modelBytes, fileName);
  const initEngine = await measureAsync(() => engine.initEngine(modelPath, initConfig));
  return {
    initEngineMs: initEngine.ms,
  };
}

function classifyPromptBucket(prompt: string): PromptBucket {
  const trimmedLength = prompt.trim().length;
  if (trimmedLength <= 80) {
    return 'short';
  }
  if (trimmedLength <= 240) {
    return 'medium';
  }
  if (trimmedLength > 240) {
    return 'long';
  }
  return 'custom';
}

function classifyOutputBucket(outputTokenLimit: number): OutputBucket {
  if (outputTokenLimit <= 16) {
    return 'short';
  }
  if (outputTokenLimit <= 64) {
    return 'medium';
  }
  return 'long';
}

function countWords(input: string): number {
  const trimmed = input.trim();
  if (!trimmed) {
    return 0;
  }
  return trimmed.split(/\s+/).length;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function inferQuantizationLabel(fileName: string): string | null {
  const stem = path.basename(fileName, path.extname(fileName));
  const match = stem.match(QUANTIZATION_SUFFIX_PATTERN);
  return match ? match[1].toUpperCase() : null;
}

function deriveArtifactLabel(fileName: string, quantizationLabel: string | null): string {
  const stem = path.basename(fileName, path.extname(fileName));
  if (!quantizationLabel) {
    return stem;
  }

  const suffixPattern = new RegExp(`[-_.]${escapeRegExp(quantizationLabel)}$`, 'i');
  const stripped = stem.replace(suffixPattern, '');
  return stripped || stem;
}

function buildScenarios(options: BenchmarkOptions): BenchmarkScenarioDefinition[] {
  if (options.preset === 'single') {
    const prompt = options.prompt ?? SHORT_PROMPT;
    const outputTokenLimit = options.tokensOverride ?? DEFAULT_SHORT_OUTPUT_TOKENS;

    return [
      {
        id: 'single',
        label: 'Single Prompt',
        prompt,
        promptBucket: classifyPromptBucket(prompt),
        promptChars: prompt.length,
        promptWords: countWords(prompt),
        outputTokenLimit,
        outputBucket: classifyOutputBucket(outputTokenLimit),
        promptFormat: options.promptFormat,
        contextBucket: 'single-request',
        concurrency: 1,
      },
    ];
  }

  return DEFAULT_PRESET_CASES.map((scenario) => {
    const outputTokenLimit = options.tokensOverride ?? scenario.outputTokenLimit;
    return {
      id: scenario.id,
      label: scenario.label,
      prompt: scenario.prompt,
      promptBucket: classifyPromptBucket(scenario.prompt),
      promptChars: scenario.prompt.length,
      promptWords: countWords(scenario.prompt),
      outputTokenLimit,
      outputBucket: classifyOutputBucket(outputTokenLimit),
      promptFormat: options.promptFormat,
      contextBucket: 'single-request',
      concurrency: 1,
    };
  });
}

function buildMixedLoadDefinition(promptFormat: PromptFormatMode): MixedLoadBenchmarkDefinition {
  return {
    id: 'mixed-lilo-vs-siso',
    label: 'Mixed Load: LILO Background vs SISO Foreground',
    background: {
      id: 'mixed-background-lilo',
      label: 'Background Long Input / Long Output',
      prompt: LONG_PROMPT,
      promptBucket: classifyPromptBucket(LONG_PROMPT),
      promptChars: LONG_PROMPT.length,
      promptWords: countWords(LONG_PROMPT),
      outputTokenLimit: DEFAULT_LONG_OUTPUT_TOKENS,
      outputBucket: classifyOutputBucket(DEFAULT_LONG_OUTPUT_TOKENS),
      promptFormat,
      contextBucket: 'single-request',
      concurrency: 1,
    },
    foreground: {
      id: 'mixed-foreground-siso',
      label: 'Foreground Short Input / Short Output',
      prompt: SHORT_PROMPT,
      promptBucket: classifyPromptBucket(SHORT_PROMPT),
      promptChars: SHORT_PROMPT.length,
      promptWords: countWords(SHORT_PROMPT),
      outputTokenLimit: DEFAULT_SHORT_OUTPUT_TOKENS,
      outputBucket: classifyOutputBucket(DEFAULT_SHORT_OUTPUT_TOKENS),
      promptFormat,
      contextBucket: 'single-request',
      concurrency: 1,
    },
    concurrency: 2,
  };
}

function buildPhase4BenchmarkConfig(initConfig: InferenceInitConfig): InferenceInitConfig {
  return {
    ...initConfig,
    nSeqMax: Math.max(initConfig.nSeqMax ?? 1, 2),
    maxCachedSessions: Math.max(initConfig.maxCachedSessions ?? 8, 2),
    enableRuntimeObservability: true,
    enableBackendProfiling: true,
  };
}

async function runPromptBenchmark(
  engine: CogentEngine,
  labelPrefix: string,
  prompt: string,
  promptFormat: PromptFormatMode,
  tokens: number,
  warmupRuns: number,
  measuredRuns: number,
  contextKeyFactory: (index: number) => string
) : Promise<{ benchmarkDurationMs: number; memory: MemorySnapshotSummary; runs: BenchmarkRun[] }> {
  const memoryBefore = captureMemoryUsage();
  for (let i = 0; i < warmupRuns; i++) {
    await engine.submitPrompt(contextKeyFactory(i), prompt, {
      nTokens: tokens,
      promptFormat,
    });
  }

  const runs: BenchmarkRun[] = [];
  const benchmarkStart = nowMs();
  for (let i = 0; i < measuredRuns; i++) {
    const label = `${labelPrefix}-${i + 1}`;
    const contextKey = contextKeyFactory(i + warmupRuns);
    const start = nowMs();
    let ttftMs: number | null = null;
    const tokenEventTimes: number[] = [];

    const output = await engine.submitPrompt(contextKey, prompt, {
      nTokens: tokens,
      promptFormat,
      onToken: () => {
        const elapsedMs = round(nowMs() - start);
        tokenEventTimes.push(elapsedMs);
        if (ttftMs == null) {
          ttftMs = elapsedMs;
        }
      },
    });

    const wallMs = round(nowMs() - start);
    const runtimeObservability = engine.getRuntimeObservability();
    if (output.length === 0 && runtimeObservability == null) {
      throw new Error(
        `Prompt run "${label}" returned empty output and no runtime observability payload. The runtime likely failed to create or execute the request context.`
      );
    }

    // Prefer the native output token counter when available, but fall back to the
    // observed stream callback count so the benchmark still works if perf payloads
    // are unavailable or partially missing.
    const outputTokenCount =
      runtimeObservability?.outputTokenCount ?? tokenEventTimes.length;
    const itlMsValues: number[] = [];
    for (let tokenIndex = 1; tokenIndex < tokenEventTimes.length; tokenIndex++) {
      itlMsValues.push(round(tokenEventTimes[tokenIndex] - tokenEventTimes[tokenIndex - 1]));
    }

    // TensorRT-LLM-style TPOT: average time per output token after the first token.
    const tpotMs =
      ttftMs != null && outputTokenCount > 1
        ? round((wallMs - ttftMs) / (outputTokenCount - 1))
        : null;

    runs.push({
      label,
      contextKey,
      wallMs,
      ttftMs,
      tpotMs,
      itlMsValues,
      inputTokenCount: runtimeObservability?.inputTokenCount ?? null,
      promptEvalTokenCount: runtimeObservability?.promptEvalTokens ?? null,
      outputTokenCount,
      outputLength: output.length,
      outputPreview: output.slice(0, OUTPUT_PREVIEW_LIMIT).replace(/\s+/g, ' ').trim(),
      runtimeObservability,
    });
  }

  const memoryAfter = captureMemoryUsage();
  return {
    benchmarkDurationMs: round(nowMs() - benchmarkStart),
    memory: {
      before: memoryBefore,
      after: memoryAfter,
      delta: diffMemoryUsage(memoryBefore, memoryAfter),
    },
    runs,
  };
}

async function runQueuedMixedLoadPair(
  engine: CogentEngine,
  labelPrefix: string,
  background: BenchmarkScenarioDefinition,
  foreground: BenchmarkScenarioDefinition,
  runIndex: number
): Promise<{ backgroundRun: BenchmarkRun; foregroundRun: BenchmarkRun }> {
  const backgroundLabel = `${labelPrefix}-background-${runIndex + 1}`;
  const foregroundLabel = `${labelPrefix}-foreground-${runIndex + 1}`;
  const backgroundContextKey = `${background.id}-mixed-${runIndex}`;
  const foregroundContextKey = `${foreground.id}-mixed-${runIndex}`;

  const backgroundStart = nowMs();
  let backgroundTtftMs: number | null = null;
  const backgroundTokenEventTimes: number[] = [];
  const backgroundRequestId = await engine.queuePrompt(backgroundContextKey, background.prompt, {
    nTokens: background.outputTokenLimit,
    promptFormat: background.promptFormat,
    onToken: () => {
      const elapsedMs = round(nowMs() - backgroundStart);
      backgroundTokenEventTimes.push(elapsedMs);
      if (backgroundTtftMs == null) {
        backgroundTtftMs = elapsedMs;
      }
    },
  });

  const foregroundStart = nowMs();
  let foregroundTtftMs: number | null = null;
  const foregroundTokenEventTimes: number[] = [];
  const foregroundRequestId = await engine.queuePrompt(foregroundContextKey, foreground.prompt, {
    nTokens: foreground.outputTokenLimit,
    promptFormat: foreground.promptFormat,
    onToken: () => {
      const elapsedMs = round(nowMs() - foregroundStart);
      foregroundTokenEventTimes.push(elapsedMs);
      if (foregroundTtftMs == null) {
        foregroundTtftMs = elapsedMs;
      }
    },
  });

  const foregroundResponse = await engine.runQueuedRequest(foregroundRequestId);
  const foregroundWallMs = round(nowMs() - foregroundStart);
  const backgroundResponse = await engine.runQueuedRequest(backgroundRequestId);
  const backgroundWallMs = round(nowMs() - backgroundStart);

  const toRun = (
    label: string,
    contextKey: string,
    wallMs: number,
    ttftMs: number | null,
    tokenEventTimes: number[],
    response: Awaited<ReturnType<CogentEngine['runQueuedRequest']>>
  ): BenchmarkRun => {
    const runtimeObservability = response.runtimeObservability ?? null;
    const outputTokenCount =
      runtimeObservability?.outputTokenCount ?? tokenEventTimes.length;
    const itlMsValues: number[] = [];
    for (let tokenIndex = 1; tokenIndex < tokenEventTimes.length; tokenIndex += 1) {
      itlMsValues.push(round(tokenEventTimes[tokenIndex] - tokenEventTimes[tokenIndex - 1]));
    }

    const effectiveTtftMs = ttftMs ?? runtimeObservability?.ttftMs ?? null;
    const tpotMs =
      effectiveTtftMs != null && outputTokenCount > 1
        ? round((wallMs - effectiveTtftMs) / (outputTokenCount - 1))
        : null;

    return {
      label,
      contextKey,
      wallMs,
      ttftMs: effectiveTtftMs,
      tpotMs,
      itlMsValues,
      inputTokenCount: runtimeObservability?.inputTokenCount ?? null,
      promptEvalTokenCount: runtimeObservability?.promptEvalTokens ?? null,
      outputTokenCount,
      outputLength: response.outputText.length,
      outputPreview: response.outputText
        .slice(0, OUTPUT_PREVIEW_LIMIT)
        .replace(/\s+/g, ' ')
        .trim(),
      runtimeObservability,
    };
  };

  return {
    backgroundRun: toRun(
      backgroundLabel,
      backgroundContextKey,
      backgroundWallMs,
      backgroundTtftMs,
      backgroundTokenEventTimes,
      backgroundResponse
    ),
    foregroundRun: toRun(
      foregroundLabel,
      foregroundContextKey,
      foregroundWallMs,
      foregroundTtftMs,
      foregroundTokenEventTimes,
      foregroundResponse
    ),
  };
}

async function runMixedLoadBenchmark(
  engine: CogentEngine,
  definition: MixedLoadBenchmarkDefinition,
  runtime: ScenarioRuntimeMetrics,
  warmupRuns: number,
  measuredRuns: number
): Promise<MixedLoadBenchmarkResult> {
  for (let i = 0; i < warmupRuns; i += 1) {
    const warmup = await runQueuedMixedLoadPair(
      engine,
      `${definition.id}-warmup`,
      definition.background,
      definition.foreground,
      i
    );
    void warmup;
  }

  const foregroundRuns: BenchmarkRun[] = [];
  const backgroundRuns: BenchmarkRun[] = [];
  const benchmarkStart = nowMs();

  for (let i = 0; i < measuredRuns; i += 1) {
    const pair = await runQueuedMixedLoadPair(
      engine,
      definition.id,
      definition.background,
      definition.foreground,
      i
    );
    backgroundRuns.push(pair.backgroundRun);
    foregroundRuns.push(pair.foregroundRun);
  }

  const benchmarkDurationMs = round(nowMs() - benchmarkStart);
  return {
    definition,
    runtime,
    background: createGroupResult(
      'hotFreshContext',
      `${definition.background.label} Under Mixed Load`,
      warmupRuns,
      measuredRuns,
      benchmarkDurationMs,
      backgroundRuns
    ),
    foreground: createGroupResult(
      'hotFreshContext',
      `${definition.foreground.label} Under Mixed Load`,
      warmupRuns,
      measuredRuns,
      benchmarkDurationMs,
      foregroundRuns
    ),
  };
}

async function runQueueCancelChurn(
  engine: CogentEngine,
  promptFormat: PromptFormatMode,
  iterations: number,
  tokenLimit: number
): Promise<QueueCancelChurnResult | null> {
  if (iterations <= 0) {
    return null;
  }

  const memoryBefore = captureMemoryUsage();
  const enqueueLatenciesMs: number[] = [];
  const cancelLatenciesMs: number[] = [];
  let cancelledCount = 0;
  const benchmarkStart = nowMs();

  for (let index = 0; index < iterations; index += 1) {
    const enqueue = await measureAsync(() =>
      engine.queuePrompt(`cancel-churn-${index}`, SHORT_PROMPT, {
        nTokens: tokenLimit,
        promptFormat,
        onToken: () => {},
      })
    );
    enqueueLatenciesMs.push(enqueue.ms);

    const cancel = await measureAsync(() =>
      engine.cancelQueuedRequest(enqueue.value)
    );
    cancelLatenciesMs.push(cancel.ms);
    if (cancel.value) {
      cancelledCount++;
    }
  }

  let smokeWallMs: number | null = null;
  let smokeOutputLength = 0;
  let smokeRuntimeObservabilityAvailable = false;
  let smokeErrorMessage: string | null = null;
  let smokeFailed = false;

  const smokeStart = nowMs();
  try {
    const output = await engine.submitPrompt('cancel-churn-smoke', SHORT_PROMPT, {
      nTokens: tokenLimit,
      promptFormat,
    });
    smokeWallMs = round(nowMs() - smokeStart);
    smokeOutputLength = output.length;
    smokeRuntimeObservabilityAvailable = engine.getRuntimeObservability() != null;
  } catch (error) {
    smokeWallMs = round(nowMs() - smokeStart);
    smokeFailed = true;
    smokeErrorMessage = error instanceof Error ? error.message : String(error);
  }

  const memoryAfter = captureMemoryUsage();
  const memory = {
    before: memoryBefore,
    after: memoryAfter,
    delta: diffMemoryUsage(memoryBefore, memoryAfter),
  };

  const warnings: string[] = [];
  if (!smokeRuntimeObservabilityAvailable) {
    warnings.push('queueCancelChurn: smoke-after-churn prompt returned no runtime observability payload.');
  }
  if (smokeFailed) {
    warnings.push(
      `queueCancelChurn: smoke-after-churn prompt failed${smokeErrorMessage ? `: ${smokeErrorMessage}` : '.'}`
    );
  }
  if (memory.delta.rssBytes > CANCEL_CHURN_MEMORY_WARNING_BYTES) {
    warnings.push(
      `queueCancelChurn: rss grew by ${formatBytes(memory.delta.rssBytes)} after churn.`
    );
  }

  return {
    iterations,
    tokenLimit,
    benchmarkDurationMs: round(nowMs() - benchmarkStart),
    enqueueLatencyMs: summarizeOptional(enqueueLatenciesMs),
    cancelLatencyMs: summarizeOptional(cancelLatenciesMs),
    cancelledCount,
    memory,
    smokePrompt: {
      wallMs: smokeWallMs,
      outputLength: smokeOutputLength,
      runtimeObservabilityAvailable: smokeRuntimeObservabilityAvailable,
      failed: smokeFailed,
      errorMessage: smokeErrorMessage,
    },
    warnings,
  };
}

function collectGroupWarnings(group: BenchmarkGroupResult, label: string): string[] {
  const warnings: string[] = [];
  const missingObservabilityCount = group.runs.filter(
    (run) => run.runtimeObservability == null
  ).length;
  if (missingObservabilityCount > 0) {
    warnings.push(
      `${label}: ${missingObservabilityCount}/${group.runs.length} runs were missing runtime observability payloads.`
    );
  }

  return warnings;
}

function collectBenchmarkWarnings(
  scenarioResults: BenchmarkScenarioResult[],
  mixedLoadResult: MixedLoadBenchmarkResult | null,
  queueCancelChurn: QueueCancelChurnResult | null
): string[] {
  const warnings: string[] = [];

  for (const scenario of scenarioResults) {
    warnings.push(
      ...collectGroupWarnings(scenario.coldPrompt, `${scenario.definition.id}/coldPrompt`),
      ...collectGroupWarnings(
        scenario.hotFreshContext,
        `${scenario.definition.id}/hotFreshContext`
      ),
      ...collectGroupWarnings(
        scenario.hotReuseContext,
        `${scenario.definition.id}/hotReuseContext`
      )
    );
  }

  if (mixedLoadResult != null) {
    warnings.push(
      ...collectGroupWarnings(mixedLoadResult.foreground, `${mixedLoadResult.definition.id}/foreground`),
      ...collectGroupWarnings(mixedLoadResult.background, `${mixedLoadResult.definition.id}/background`)
    );
  }

  if (queueCancelChurn != null) {
    warnings.push(...queueCancelChurn.warnings);
  }

  return warnings;
}

function summarizeGroup(
  runs: BenchmarkRun[],
  benchmarkDurationMs: number
): BenchmarkGroupSummary {
  const observabilityRuns = runs.map((run) => run.runtimeObservability);
  const totalInputTokens = runs.reduce((acc, run) => acc + (run.inputTokenCount ?? 0), 0);
  const totalGeneratedTokens = runs.reduce((acc, run) => acc + (run.outputTokenCount ?? 0), 0);
  const totalItls = runs.flatMap((run) => run.itlMsValues);
  // Throughput metrics use the actual measured wall-clock group duration, not the
  // sum of request latencies. This matches standard serving benchmark reporting.
  const benchmarkDurationSeconds = benchmarkDurationMs > 0 ? benchmarkDurationMs / 1000 : 0;

  return {
    serving: {
      successfulRequests: runs.length,
      benchmarkDurationMs,
      totalInputTokens,
      totalGeneratedTokens,
      requestThroughputRps:
        benchmarkDurationSeconds > 0 ? round(runs.length / benchmarkDurationSeconds) : null,
      outputTokenThroughputTps:
        benchmarkDurationSeconds > 0 ? round(totalGeneratedTokens / benchmarkDurationSeconds) : null,
      totalTokenThroughputTps:
        benchmarkDurationSeconds > 0
          ? round((totalInputTokens + totalGeneratedTokens) / benchmarkDurationSeconds)
          : null,
      ttftMs: summarizeOptional(runs.map((run) => run.ttftMs)),
      tpotMs: summarizeOptional(runs.map((run) => run.tpotMs)),
      itlMs: summarizeOptional(totalItls),
      e2elMs: summarize(runs.map((run) => run.wallMs)),
    },
    runtime: {
      avgLogicalInputTokenCount: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.inputTokenCount
      ),
      avgPromptEvalTokens: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.promptEvalTokens
      ),
      avgTotalMs: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.totalMs
      ),
      avgPromptEvalMs: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.promptEvalMs
      ),
      avgDecodeEvalMs: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.decodeEvalMs
      ),
      avgSampleMs: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.sampleMs
      ),
      avgOutputTokenCount: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.outputTokenCount
      ),
      avgQueueDelayMs: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.queueDelayMs
      ),
      avgTailItlMs: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.tailItlMs
      ),
      avgBatchParticipationCount: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.batchParticipationCount
      ),
      avgDecodeFirstTickCount: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.decodeFirstTickCount
      ),
      avgChunkedPrefillTickCount: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.chunkedPrefillTickCount
      ),
      avgMixedWorkloadTickCount: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.mixedWorkloadTickCount
      ),
      avgLcpReuseTokens: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.lcpReuseTokens
      ),
      avgPrefixCacheRestoreTokens: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.prefixCacheRestoreTokens
      ),
      avgPrefixCacheHitCount: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.prefixCacheHitCount
      ),
      avgPrefixCacheStoreCount: averageRuntimeObservabilityMetric(
        observabilityRuns,
        (metrics) => metrics.prefixCacheStoreCount
      ),
      promptEvalTokensPerSecond: summarizeThroughput(
        observabilityRuns,
        promptTokensPerSecond
      ),
      outputTokensPerSecond: summarizeThroughput(
        observabilityRuns,
        decodeTokensPerSecond
      ),
    },
    derived: {
      avgHostOverheadMs: averageBenchmarkRunMetric(runs, (run) => {
        const runtimeTotalMs = run.runtimeObservability?.totalMs ?? null;
        if (runtimeTotalMs == null || runtimeTotalMs < 0) {
          return null;
        }
        return run.wallMs - runtimeTotalMs;
      }),
      avgPromptReuseTokens: averageBenchmarkRunMetric(runs, (run) => {
        if (run.inputTokenCount == null || run.promptEvalTokenCount == null) {
          return null;
        }
        return Math.max(0, run.inputTokenCount - run.promptEvalTokenCount);
      }),
      avgPromptReuseRatio: averageBenchmarkRunMetric(runs, (run) => {
        if (
          run.inputTokenCount == null ||
          run.promptEvalTokenCount == null ||
          run.inputTokenCount <= 0
        ) {
          return null;
        }
        return Math.max(0, run.inputTokenCount - run.promptEvalTokenCount) / run.inputTokenCount;
      }),
    },
  };
}

function createGroupResult(
  id: BenchmarkGroupResult['id'],
  label: string,
  warmupRuns: number,
  measuredRuns: number,
  benchmarkDurationMs: number,
  runs: BenchmarkRun[],
  memory: MemorySnapshotSummary | null = null
): BenchmarkGroupResult {
  return {
    id,
    label,
    warmupRuns,
    measuredRuns,
    benchmarkDurationMs,
    summary: summarizeGroup(runs, benchmarkDurationMs),
    memory,
    runs,
  };
}

async function runScenarioBenchmark(
  engine: CogentEngine,
  scenario: BenchmarkScenarioDefinition,
  runtime: ScenarioRuntimeMetrics,
  warmupRuns: number,
  measuredRuns: number
): Promise<BenchmarkScenarioResult> {
  const coldPrompt = await runPromptBenchmark(
    engine,
    `${scenario.id}-cold`,
    scenario.prompt,
    scenario.promptFormat,
    scenario.outputTokenLimit,
    0,
    1,
    () => `${scenario.id}-cold`
  );

  const hotFreshContext = await runPromptBenchmark(
    engine,
    `${scenario.id}-hot-fresh`,
    scenario.prompt,
    scenario.promptFormat,
    scenario.outputTokenLimit,
    warmupRuns,
    measuredRuns,
    (index) => `${scenario.id}-fresh-${index}`
  );

  const hotReuseContext = await runPromptBenchmark(
    engine,
    `${scenario.id}-hot-reuse`,
    scenario.prompt,
    scenario.promptFormat,
    scenario.outputTokenLimit,
    warmupRuns,
    measuredRuns,
    () => `${scenario.id}-reuse`
  );

  return {
    definition: scenario,
    runtime,
    coldPrompt: createGroupResult(
      'coldPrompt',
      'Cold Prompt',
      0,
      1,
      coldPrompt.benchmarkDurationMs,
      coldPrompt.runs,
      coldPrompt.memory
    ),
    hotFreshContext: createGroupResult(
      'hotFreshContext',
      'Hot Prompt: Fresh Context',
      warmupRuns,
      measuredRuns,
      hotFreshContext.benchmarkDurationMs,
      hotFreshContext.runs,
      hotFreshContext.memory
    ),
    hotReuseContext: createGroupResult(
      'hotReuseContext',
      'Hot Prompt: Reused Context',
      warmupRuns,
      measuredRuns,
      hotReuseContext.benchmarkDurationMs,
      hotReuseContext.runs,
      hotReuseContext.memory
    ),
  };
}

function printGroupResult(group: BenchmarkGroupResult): void {
  const { serving, runtime, derived } = group.summary;

  console.log(`\n  ${group.label}`);
  console.log(`    Successful requests:             ${serving.successfulRequests}`);
  console.log(`    Benchmark duration (s):          ${round(serving.benchmarkDurationMs / 1000)}`);
  console.log(`    Total input tokens:              ${serving.totalInputTokens}`);
  console.log(`    Total generated tokens:          ${serving.totalGeneratedTokens}`);
  console.log(
    `    Request throughput (req/s):      ${serving.requestThroughputRps == null ? 'n/a' : serving.requestThroughputRps}`
  );
  console.log(
    `    Output token throughput (tok/s): ${serving.outputTokenThroughputTps == null ? 'n/a' : serving.outputTokenThroughputTps}`
  );
  console.log(
    `    Total token throughput (tok/s):  ${serving.totalTokenThroughputTps == null ? 'n/a' : serving.totalTokenThroughputTps}`
  );
  if (serving.ttftMs) {
    console.log(`    Mean TTFT (ms):                  ${serving.ttftMs.meanMs}`);
    console.log(`    Median TTFT (ms):                ${serving.ttftMs.medianMs}`);
    console.log(`    P99 TTFT (ms):                   ${serving.ttftMs.p99Ms}`);
  }
  if (serving.tpotMs) {
    console.log(`    Mean TPOT (ms):                  ${serving.tpotMs.meanMs}`);
    console.log(`    Median TPOT (ms):                ${serving.tpotMs.medianMs}`);
    console.log(`    P99 TPOT (ms):                   ${serving.tpotMs.p99Ms}`);
  }
  if (serving.itlMs) {
    console.log(`    Mean ITL (ms):                   ${serving.itlMs.meanMs}`);
    console.log(`    Median ITL (ms):                 ${serving.itlMs.medianMs}`);
    console.log(`    P99 ITL (ms):                    ${serving.itlMs.p99Ms}`);
  }
  console.log(`    Mean E2EL (ms):                  ${serving.e2elMs.meanMs}`);
  console.log(`    Median E2EL (ms):                ${serving.e2elMs.medianMs}`);
  console.log(`    P99 E2EL (ms):                   ${serving.e2elMs.p99Ms}`);

  if (
    runtime.avgTotalMs != null &&
    runtime.avgPromptEvalMs != null &&
    runtime.avgDecodeEvalMs != null &&
    runtime.avgSampleMs != null &&
    runtime.avgLogicalInputTokenCount != null &&
    runtime.avgPromptEvalTokens != null &&
    runtime.avgOutputTokenCount != null
  ) {
    console.log(
      `    Runtime counters:                total_ms=${runtime.avgTotalMs} prompt_eval_ms=${runtime.avgPromptEvalMs} decode_eval_ms=${runtime.avgDecodeEvalMs} sample_ms=${runtime.avgSampleMs} input_tokens=${runtime.avgLogicalInputTokenCount} effective_prompt_tokens=${runtime.avgPromptEvalTokens} output_tokens=${runtime.avgOutputTokenCount}`
    );
  }
  if (runtime.promptEvalTokensPerSecond != null) {
    console.log(`    Prompt eval tok/s:               ${runtime.promptEvalTokensPerSecond}`);
  }
  if (runtime.outputTokensPerSecond != null) {
    console.log(`    Decode tok/s:                    ${runtime.outputTokensPerSecond}`);
  }
  if (derived.avgHostOverheadMs != null) {
    console.log(`    Avg host overhead (ms):          ${derived.avgHostOverheadMs}`);
  }
  if (derived.avgPromptReuseTokens != null) {
    console.log(`    Avg prompt reuse tokens:         ${derived.avgPromptReuseTokens}`);
  }
  if (derived.avgPromptReuseRatio != null) {
    console.log(`    Avg prompt reuse ratio:          ${round(derived.avgPromptReuseRatio * 100)}%`);
  }
  if (runtime.avgQueueDelayMs != null) {
    console.log(`    Avg queue delay (ms):            ${runtime.avgQueueDelayMs}`);
  }
  if (runtime.avgTailItlMs != null) {
    console.log(`    Avg tail ITL (ms):               ${runtime.avgTailItlMs}`);
  }
  if (runtime.avgBatchParticipationCount != null) {
    console.log(`    Avg batch participations:        ${runtime.avgBatchParticipationCount}`);
  }
  if (runtime.avgDecodeFirstTickCount != null) {
    console.log(`    Avg decode-first ticks:          ${runtime.avgDecodeFirstTickCount}`);
  }
  if (runtime.avgChunkedPrefillTickCount != null) {
    console.log(`    Avg chunked-prefill ticks:       ${runtime.avgChunkedPrefillTickCount}`);
  }
  if (runtime.avgMixedWorkloadTickCount != null) {
    console.log(`    Avg mixed-workload ticks:        ${runtime.avgMixedWorkloadTickCount}`);
  }
  if (runtime.avgLcpReuseTokens != null) {
    console.log(`    Avg LCP reuse tokens:            ${runtime.avgLcpReuseTokens}`);
  }
  if (runtime.avgPrefixCacheRestoreTokens != null) {
    console.log(
      `    Avg prefix-cache restore tokens: ${runtime.avgPrefixCacheRestoreTokens}`
    );
  }
  if (runtime.avgPrefixCacheHitCount != null) {
    console.log(`    Avg prefix-cache hits:           ${runtime.avgPrefixCacheHitCount}`);
  }
  if (runtime.avgPrefixCacheStoreCount != null) {
    console.log(`    Avg prefix-cache stores:         ${runtime.avgPrefixCacheStoreCount}`);
  }
  if (group.memory != null) {
    console.log(
      `    Memory delta:                    rss=${formatBytes(group.memory.delta.rssBytes)} heap=${formatBytes(group.memory.delta.heapUsedBytes)} external=${formatBytes(group.memory.delta.externalBytes)}`
    );
  }
}

function printScenarioResult(result: BenchmarkScenarioResult): void {
  const definition = result.definition;
  console.log(`\nScenario: ${definition.label}`);
  console.log(`  init engine=${result.runtime.initEngineMs} ms`);
  console.log(
    `  prompt bucket=${definition.promptBucket} chars=${definition.promptChars} words=${definition.promptWords} format=${definition.promptFormat}`
  );
  console.log(
    `  output bucket=${definition.outputBucket} token_limit=${definition.outputTokenLimit} concurrency=${definition.concurrency}`
  );

  printGroupResult(result.coldPrompt);
  printGroupResult(result.hotFreshContext);
  printGroupResult(result.hotReuseContext);
}

function printMixedLoadResult(result: MixedLoadBenchmarkResult): void {
  console.log(`\nMixed Load: ${result.definition.label}`);
  console.log(`  init engine=${result.runtime.initEngineMs} ms`);
  console.log(
    `  background=${result.definition.background.label} | foreground=${result.definition.foreground.label} | concurrency=${result.definition.concurrency}`
  );
  printGroupResult(result.foreground);
  printGroupResult(result.background);
}

function printQueueCancelChurnResult(result: QueueCancelChurnResult): void {
  console.log('\nQueue/Cancel Churn');
  console.log(`  iterations   ${result.iterations}`);
  console.log(`  token limit  ${result.tokenLimit}`);
  console.log(`  cancelled    ${result.cancelledCount}`);
  console.log(`  duration (s) ${round(result.benchmarkDurationMs / 1000)}`);
  if (result.enqueueLatencyMs != null) {
    console.log(`  enqueue mean ${result.enqueueLatencyMs.meanMs} ms`);
    console.log(`  enqueue p99  ${result.enqueueLatencyMs.p99Ms} ms`);
  }
  if (result.cancelLatencyMs != null) {
    console.log(`  cancel mean  ${result.cancelLatencyMs.meanMs} ms`);
    console.log(`  cancel p99   ${result.cancelLatencyMs.p99Ms} ms`);
  }
  console.log(
    `  memory delta rss=${formatBytes(result.memory.delta.rssBytes)} heap=${formatBytes(result.memory.delta.heapUsedBytes)} external=${formatBytes(result.memory.delta.externalBytes)}`
  );
  console.log(
    `  smoke prompt failed=${result.smokePrompt.failed} output_length=${result.smokePrompt.outputLength} runtime_observability=${result.smokePrompt.runtimeObservabilityAvailable}`
  );
  if (result.smokePrompt.errorMessage != null) {
    console.log(`  smoke error  ${result.smokePrompt.errorMessage}`);
  }
}

function printWarnings(warnings: string[]): void {
  if (warnings.length === 0) {
    return;
  }
  console.log('\nWarnings');
  for (const warning of warnings) {
    console.log(`  - ${warning}`);
  }
}

function printBackendProfile(backend: BenchmarkBackendProfile): void {
  console.log('\nBackend');
  console.log(`  requested execution  ${backend.requestedExecutionMode}`);
  console.log(`  requested gpu layers ${backend.requestedGpuLayers == null ? 'default' : backend.requestedGpuLayers}`);
  console.log(`  inferred backend     ${backend.inferredExecutionBackend}`);
  console.log(`  runtime status       ${backend.runtimeBackendStatus}`);
  console.log(`  gpu offload support  ${backend.gpuOffloadSupported == null ? 'unknown' : backend.gpuOffloadSupported}`);
  console.log(`  registered backends  ${backend.availableBackends.length === 0 ? 'none' : backend.availableBackends.join(', ')}`);
  console.log(`  runtime devices      ${backend.runtimeDeviceLabels.length === 0 ? 'none' : backend.runtimeDeviceLabels.join(' | ')}`);
  if (backend.notes.length > 0) {
    console.log(`  notes                ${backend.notes.join(' | ')}`);
  }
}

async function writeJsonReport(jsonPath: string, report: BenchmarkReport): Promise<void> {
  const resolvedPath = path.isAbsolute(jsonPath) ? jsonPath : path.resolve(process.cwd(), jsonPath);
  await mkdir(path.dirname(resolvedPath), { recursive: true });
  await Bun.write(resolvedPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(`\nSaved JSON report to ${resolvedPath}`);
}

function ensureBundledBunRuntimeUrls(): ReturnType<typeof getBundledBunRuntimeUrls> {
  const runtimeUrls = getBundledBunRuntimeUrls();
  const modulePath = fileURLToPath(runtimeUrls.moduleUrl);
  const wasmPath = fileURLToPath(runtimeUrls.wasmUrl);

  if (!existsSync(modulePath) || !existsSync(wasmPath)) {
    throw new Error(
      `Missing Bun runtime artifacts. Run "bun run build:wasm:bun" first.\nExpected:\n- ${modulePath}\n- ${wasmPath}`
    );
  }

  return runtimeUrls;
}

async function main(): Promise<void> {
  const options = parseArgs(Bun.argv.slice(2));
  const scenarios = buildScenarios(options);
  const mixedLoadDefinition = buildMixedLoadDefinition(options.promptFormat);
  const effectiveInitConfig = buildPhase4BenchmarkConfig(options.initConfig);
  const runtimeUrls = ensureBundledBunRuntimeUrls();
  const fileName = path.basename(options.modelPath);
  const quantizationLabel = options.quantizationLabel ?? inferQuantizationLabel(fileName);
  const artifactLabel = options.artifactLabel ?? deriveArtifactLabel(fileName, quantizationLabel);

  console.log('Bun inference benchmark');
  console.log(`  preset      ${options.preset}`);
  console.log(`  scenarios    ${scenarios.length}`);
  console.log(`  model       ${options.modelPath}`);
  console.log(`  artifact    ${artifactLabel}`);
  console.log(`  quant       ${quantizationLabel ?? 'unknown'}`);
  console.log(`  format      ${options.promptFormat}`);
  console.log(`  warmup      ${options.warmupRuns}`);
  console.log(`  runs        ${options.measuredRuns}`);
  console.log(`  churn       ${options.cancelChurnRuns}`);
  console.log(`  churn tok   ${options.cancelChurnTokens}`);
  console.log(`  prefill     ${effectiveInitConfig.prefillChunkSize ?? 0}`);
  console.log(`  policy      ${effectiveInitConfig.schedulerPolicy ?? 'balanced'}`);
  console.log(`  reserve     ${effectiveInitConfig.decodeTokenReserve ?? 1}`);

  const requestsPerScenario = 1 + 2 * (options.warmupRuns + options.measuredRuns);
  const mixedLoadRequests = 2 * (options.warmupRuns + options.measuredRuns);
  const totalPlannedRequests = scenarios.length * requestsPerScenario + mixedLoadRequests;
  const totalPlannedOutputTokens = scenarios.reduce(
    (acc, scenario) => acc + scenario.outputTokenLimit * requestsPerScenario,
    0
  ) + (DEFAULT_LONG_OUTPUT_TOKENS + DEFAULT_SHORT_OUTPUT_TOKENS) * (options.warmupRuns + options.measuredRuns);
  console.log(`  requests    ${totalPlannedRequests}`);
  console.log(`  max tokens  ${totalPlannedOutputTokens}`);
  if (totalPlannedOutputTokens >= 2000) {
    console.log('  note        long benchmark; this can take several minutes on a local Wasm/CPU path');
  }

  const readModel = await measureAsync(async () => {
    const file = Bun.file(options.modelPath);
    const bytes = new Uint8Array(await file.arrayBuffer());
    return {
      bytes,
      size: bytes.byteLength,
    };
  });

  const modelBytes = readModel.value.bytes;
  const startup = await initializeScenarioEngine(runtimeUrls, modelBytes, fileName);
  const maybeNavigator =
    typeof navigator !== 'undefined'
      ? (navigator as { gpu?: unknown; userAgent?: string })
      : null;
  const runtimeInitConfig =
    maybeNavigator?.gpu == null
      ? {
          ...effectiveInitConfig,
          nGpuLayers: 0,
        }
      : effectiveInitConfig;
  const runtimeBackend = await startup.engine.getBackendObservability();
  const backendProfile = buildBenchmarkBackendProfile(runtimeBackend, runtimeInitConfig, {
    apiAvailable: false,
    adapterAvailable: false,
    adapterLabel: null,
  });

  console.log('\nRuntime');
  console.log(`  model read ms    ${readModel.ms}`);
  console.log(`  init module ms   ${startup.initModuleMs}`);
  console.log(`  memfs load ms    ${startup.loadModelIntoMemfsMs}`);
  console.log(`  model size       ${formatBytes(readModel.value.size)}`);
  printBackendProfile(backendProfile);

  const scenarioResults: BenchmarkScenarioResult[] = [];
  let mixedLoadResult: MixedLoadBenchmarkResult | null = null;
  let queueCancelChurnResult: QueueCancelChurnResult | null = null;
  try {
    for (const scenario of scenarios) {
      const runtime = await reinitializeScenarioEngine(
        startup.engine,
        modelBytes,
        fileName,
        runtimeInitConfig
      );
      const scenarioResult = await runScenarioBenchmark(
        startup.engine,
        scenario,
        runtime,
        options.warmupRuns,
        options.measuredRuns
      );
      printScenarioResult(scenarioResult);
      scenarioResults.push(scenarioResult);
    }

    const churnRuntime = await reinitializeScenarioEngine(
      startup.engine,
      modelBytes,
      fileName,
      runtimeInitConfig
    );
    void churnRuntime;
    queueCancelChurnResult = await runQueueCancelChurn(
      startup.engine,
      options.promptFormat,
      options.cancelChurnRuns,
      options.cancelChurnTokens
    );
    if (queueCancelChurnResult != null) {
      printQueueCancelChurnResult(queueCancelChurnResult);
    }

    if (maybeNavigator?.gpu != null) {
      const mixedRuntime = await reinitializeScenarioEngine(
        startup.engine,
        modelBytes,
        fileName,
        runtimeInitConfig
      );
      mixedLoadResult = await runMixedLoadBenchmark(
        startup.engine,
        mixedLoadDefinition,
        mixedRuntime,
        options.warmupRuns,
        options.measuredRuns
      );
      printMixedLoadResult(mixedLoadResult);
    }
  } finally {
    startup.engine.close();
  }

  const warnings = collectBenchmarkWarnings(
    scenarioResults,
    mixedLoadResult,
    queueCancelChurnResult
  );
  printWarnings(warnings);

  const report: BenchmarkReport = {
    schemaVersion: 'cogent.benchmark.bun.v8',
    generatedAt: new Date().toISOString(),
    benchmark: {
      script: 'packages/cogent-engine/benchmarks/benchmark-bun.ts',
      preset: options.preset,
      promptFormat: options.promptFormat,
      warmupRuns: options.warmupRuns,
      measuredRuns: options.measuredRuns,
      cancelChurnRuns: options.cancelChurnRuns,
      cancelChurnTokens: options.cancelChurnTokens,
      scenarioCount: scenarioResults.length,
    },
    environment: {
      runtimeKind: 'bun',
      bunVersion: Bun.version,
      nodeCompatVersion: process.version,
      platform: process.platform,
      arch: process.arch,
      userAgent: maybeNavigator?.userAgent ?? null,
      hasNavigatorGpu: Boolean(maybeNavigator?.gpu),
      adapterAvailable: false,
      adapterLabel: null,
    },
    model: {
      modelPath: options.modelPath,
      fileName,
      artifactLabel,
      quantizationLabel,
      modelBytes: readModel.value.size,
    },
    backend: backendProfile,
    runtime: {
      initConfig: runtimeInitConfig,
      readModelMs: readModel.ms,
      initModuleMs: startup.initModuleMs,
      loadModelIntoMemfsMs: startup.loadModelIntoMemfsMs,
      initEngineSummary: {
        initEngineMs: summarize(
          [
            ...scenarioResults.map((scenario) => scenario.runtime.initEngineMs),
            ...(mixedLoadResult == null ? [] : [mixedLoadResult.runtime.initEngineMs]),
          ]
        ),
      },
    },
    memory: captureMemoryUsage(),
    scenarios: scenarioResults,
    mixedLoad: mixedLoadResult,
    queueCancelChurn: queueCancelChurnResult,
    warnings,
    limitations: [
      'This Bun track is authoritative for Wasm host/runtime overhead, not browser WebGPU kernel behavior.',
      'The backend section reports runtime backend availability and requested execution mode, not a browser-selected WebGPU adapter.',
      'TTFT is measured from the first streamed token callback exposed by the runtime.',
      'TPOT is computed per request as (E2EL - TTFT) / (output tokens - 1), while ITL is computed from token-to-token callback intervals.',
      'Logical input tokens and effective prompt-eval tokens are reported separately so context reuse does not distort headline throughput metrics.',
      'Serial scenario groups remain concurrency=1 baselines; the mixedLoad section is the Phase 4 concurrency=2 fairness check.',
      'The Emscripten module and MEMFS model are loaded once per benchmark run; each scenario reinitializes only the native inference engine.',
      ...(maybeNavigator?.gpu == null
        ? [
            'Bun on this machine does not expose navigator.gpu, so the benchmark forces nGpuLayers=0 and runs CPU/Wasm only.',
            'The mixedLoad Phase 4 benchmark is skipped on this Bun track because it is intended to validate the real browser scheduler path.',
          ]
        : []),
    ],
  };

  if (options.jsonPath) {
    await writeJsonReport(options.jsonPath, report);
  }
}

await main();
