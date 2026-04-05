import { existsSync } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { CogentEngine, getBundledRuntimeUrls } from '../dist/esm/index.js';
import type {
  BackendInfo,
  FlashAttentionMode,
  InferenceInitConfig,
  PromptFormatMode,
  PromptPerformanceStats,
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
  perf: PromptPerformanceStats | null;
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
  promptEvalTokensPerSecond: number | null;
  outputTokensPerSecond: number | null;
}

interface BenchmarkGroupSummary {
  serving: ServingBenchmarkSummary;
  runtime: RuntimeBenchmarkSummary;
}

interface BenchmarkGroupResult {
  id: 'coldPrompt' | 'hotFreshContext' | 'hotReuseContext';
  label: string;
  warmupRuns: number;
  measuredRuns: number;
  benchmarkDurationMs: number;
  summary: BenchmarkGroupSummary;
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

interface RuntimeMemoryUsage {
  rssBytes: number;
  heapUsedBytes: number;
  externalBytes: number;
  arrayBuffersBytes: number;
}

interface BenchmarkReport {
  schemaVersion: 'cogent.benchmark.bun.v5';
  generatedAt: string;
  benchmark: {
    script: string;
    preset: BenchmarkPresetName;
    promptFormat: PromptFormatMode;
    warmupRuns: number;
    measuredRuns: number;
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
  backendRegistries: BackendInfo['availableBackends'];
  runtimeDeviceCount: number;
  runtimeAcceleratorDeviceCount: number;
  runtimeDeviceLabels: string[];
  runtimeDevices: BackendInfo['devices'];
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

function averagePerfMetric(
  perfRuns: Array<PromptPerformanceStats | null>,
  metric: (perf: PromptPerformanceStats) => number
): number | null {
  const values = perfRuns
    .filter((perf): perf is PromptPerformanceStats => perf !== null)
    .map(metric)
    .filter((value) => Number.isFinite(value) && value >= 0);

  if (values.length === 0) {
    return null;
  }

  const total = values.reduce((acc, value) => acc + value, 0);
  return round(total / values.length);
}

function summarizeThroughput(
  perfRuns: Array<PromptPerformanceStats | null>,
  metric: (perf: PromptPerformanceStats) => number | null
): number | null {
  const values = perfRuns
    .filter((perf): perf is PromptPerformanceStats => perf !== null)
    .map(metric)
    .filter((value): value is number => value != null && value > 0);

  if (values.length === 0) {
    return null;
  }

  const total = values.reduce((acc, value) => acc + value, 0);
  return round(total / values.length);
}

function promptTokensPerSecond(perf: PromptPerformanceStats): number | null {
  // Effective prefill throughput based on llama.cpp perf counters, not end-to-end wall time.
  if (perf.promptEvalMs <= 0 || perf.promptEvalTokens <= 0) {
    return null;
  }
  return (perf.promptEvalTokens * 1000) / perf.promptEvalMs;
}

function decodeTokensPerSecond(perf: PromptPerformanceStats): number | null {
  // Effective decode throughput based on llama.cpp perf counters, not end-to-end wall time.
  if (perf.decodeEvalMs <= 0 || perf.outputTokenCount <= 0) {
    return null;
  }
  return (perf.outputTokenCount * 1000) / perf.decodeEvalMs;
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

function formatRuntimeDeviceLabel(device: BackendInfo['devices'][number]): string {
  const detail = device.description || device.name || device.backendName || device.type;
  return `${device.backendName}:${detail}`;
}

function inferRequestedExecutionMode(initConfig: InferenceInitConfig): RequestedExecutionMode {
  return initConfig.nGpuLayers === 0 ? 'cpu-only' : 'gpu-offload';
}

function inferRuntimeBackendStatus(runtimeBackend: BackendInfo | null): RuntimeBackendStatus {
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
  runtimeBackend: BackendInfo | null
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
  runtimeBackend: BackendInfo | null,
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
  runtimeUrls: ReturnType<typeof getBundledRuntimeUrls>,
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
  modelPath: string,
  initConfig: InferenceInitConfig
): Promise<ScenarioRuntimeMetrics> {
  // Scenario isolation comes from rebuilding the native inference runtime, not from
  // recreating the JS/Wasm module and re-copying the model on every case.
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

async function runPromptBenchmark(
  engine: CogentEngine,
  labelPrefix: string,
  prompt: string,
  promptFormat: PromptFormatMode,
  tokens: number,
  warmupRuns: number,
  measuredRuns: number,
  contextKeyFactory: (index: number) => string
): Promise<{ benchmarkDurationMs: number; runs: BenchmarkRun[] }> {
  for (let i = 0; i < warmupRuns; i++) {
    await engine.streamPrompt(contextKeyFactory(i), prompt, {
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

    const output = await engine.streamPrompt(contextKey, prompt, {
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
    const perf = engine.getLastPromptPerformance();
    if (output.length === 0 && perf == null) {
      throw new Error(
        `Prompt run "${label}" returned empty output and no perf payload. The runtime likely failed to create or execute the request context.`
      );
    }

    // Prefer the native output token counter when available, but fall back to the
    // observed stream callback count so the benchmark still works if perf payloads
    // are unavailable or partially missing.
    const outputTokenCount = perf?.outputTokenCount ?? tokenEventTimes.length;
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
      inputTokenCount: perf?.inputTokenCount ?? null,
      promptEvalTokenCount: perf?.promptEvalTokens ?? null,
      outputTokenCount,
      outputLength: output.length,
      outputPreview: output.slice(0, OUTPUT_PREVIEW_LIMIT).replace(/\s+/g, ' ').trim(),
      perf,
    });
  }

  return {
    benchmarkDurationMs: round(nowMs() - benchmarkStart),
    runs,
  };
}

function summarizeGroup(
  runs: BenchmarkRun[],
  benchmarkDurationMs: number
): BenchmarkGroupSummary {
  const perfRuns = runs.map((run) => run.perf);
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
      avgLogicalInputTokenCount: averagePerfMetric(perfRuns, (perf) => perf.inputTokenCount),
      avgPromptEvalTokens: averagePerfMetric(perfRuns, (perf) => perf.promptEvalTokens),
      avgTotalMs: averagePerfMetric(perfRuns, (perf) => perf.totalMs),
      avgPromptEvalMs: averagePerfMetric(perfRuns, (perf) => perf.promptEvalMs),
      avgDecodeEvalMs: averagePerfMetric(perfRuns, (perf) => perf.decodeEvalMs),
      avgSampleMs: averagePerfMetric(perfRuns, (perf) => perf.sampleMs),
      avgOutputTokenCount: averagePerfMetric(perfRuns, (perf) => perf.outputTokenCount),
      promptEvalTokensPerSecond: summarizeThroughput(perfRuns, promptTokensPerSecond),
      outputTokensPerSecond: summarizeThroughput(perfRuns, decodeTokensPerSecond),
    },
  };
}

function createGroupResult(
  id: BenchmarkGroupResult['id'],
  label: string,
  warmupRuns: number,
  measuredRuns: number,
  benchmarkDurationMs: number,
  runs: BenchmarkRun[]
): BenchmarkGroupResult {
  return {
    id,
    label,
    warmupRuns,
    measuredRuns,
    benchmarkDurationMs,
    summary: summarizeGroup(runs, benchmarkDurationMs),
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
      coldPrompt.runs
    ),
    hotFreshContext: createGroupResult(
      'hotFreshContext',
      'Hot Prompt: Fresh Context',
      warmupRuns,
      measuredRuns,
      hotFreshContext.benchmarkDurationMs,
      hotFreshContext.runs
    ),
    hotReuseContext: createGroupResult(
      'hotReuseContext',
      'Hot Prompt: Reused Context',
      warmupRuns,
      measuredRuns,
      hotReuseContext.benchmarkDurationMs,
      hotReuseContext.runs
    ),
  };
}

function printGroupResult(group: BenchmarkGroupResult): void {
  const { serving, runtime } = group.summary;

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

async function main(): Promise<void> {
  const options = parseArgs(Bun.argv.slice(2));
  const scenarios = buildScenarios(options);
  const runtimeUrls = getBundledRuntimeUrls();
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

  const requestsPerScenario = 1 + 2 * (options.warmupRuns + options.measuredRuns);
  const totalPlannedRequests = scenarios.length * requestsPerScenario;
  const totalPlannedOutputTokens = scenarios.reduce(
    (acc, scenario) => acc + scenario.outputTokenLimit * requestsPerScenario,
    0
  );
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

  let modelBytes = readModel.value.bytes;
  const startup = await initializeScenarioEngine(runtimeUrls, modelBytes, fileName);
  modelBytes = new Uint8Array(0);
  const runtimeBackend = await startup.engine.getBackendInfo();
  const backendProfile = buildBenchmarkBackendProfile(runtimeBackend, options.initConfig, {
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
  try {
    for (const scenario of scenarios) {
      const runtime = await reinitializeScenarioEngine(
        startup.engine,
        startup.modelPath,
        options.initConfig
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
  } finally {
    startup.engine.close();
  }

  const maybeNavigator =
    typeof navigator !== 'undefined'
      ? (navigator as { gpu?: unknown; userAgent?: string })
      : null;

  const report: BenchmarkReport = {
    schemaVersion: 'cogent.benchmark.bun.v5',
    generatedAt: new Date().toISOString(),
    benchmark: {
      script: 'packages/cogent-engine/benchmarks/benchmark-bun.ts',
      preset: options.preset,
      promptFormat: options.promptFormat,
      warmupRuns: options.warmupRuns,
      measuredRuns: options.measuredRuns,
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
      initConfig: options.initConfig,
      readModelMs: readModel.ms,
      initModuleMs: startup.initModuleMs,
      loadModelIntoMemfsMs: startup.loadModelIntoMemfsMs,
      initEngineSummary: {
        initEngineMs: summarize(scenarioResults.map((scenario) => scenario.runtime.initEngineMs)),
      },
    },
    memory: captureMemoryUsage(),
    scenarios: scenarioResults,
    limitations: [
      'This Bun track is authoritative for Wasm host/runtime overhead, not browser WebGPU kernel behavior.',
      'The backend section reports runtime backend availability and requested execution mode, not a browser-selected WebGPU adapter.',
      'TTFT is measured from the first streamed token callback exposed by the runtime.',
      'TPOT is computed per request as (E2EL - TTFT) / (output tokens - 1), while ITL is computed from token-to-token callback intervals.',
      'Logical input tokens and effective prompt-eval tokens are reported separately so context reuse does not distort headline throughput metrics.',
      'Concurrency is fixed at 1 until the slot scheduler phases are implemented.',
      'The Emscripten module and MEMFS model are loaded once per benchmark run; each scenario reinitializes only the native inference engine.',
    ],
  };

  if (options.jsonPath) {
    await writeJsonReport(options.jsonPath, report);
  }
}

await main();
