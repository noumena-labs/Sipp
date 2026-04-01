import { mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { CogentEngine, getBundledRuntimeUrls } from '../dist/esm/index.js';

interface PromptPerformanceStats {
  totalMs: number;
  promptEvalMs: number;
  decodeEvalMs: number;
  sampleMs: number;
  promptEvalTokens: number;
  decodeEvalCount: number;
  sampleCount: number;
  outputTokenCount: number;
}

type BenchmarkPresetName = 'single' | 'default';
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
}

interface BenchmarkSummary {
  minMs: number;
  medianMs: number;
  meanMs: number;
  p95Ms: number;
  maxMs: number;
}

interface DerivedRunMetrics {
  ttftMs: number | null;
  promptTokensPerSecond: number | null;
  decodeTokensPerSecond: number | null;
}

interface BenchmarkRun {
  label: string;
  contextKey: string;
  wallMs: number;
  outputLength: number;
  outputPreview: string;
  perf: PromptPerformanceStats | null;
  derived: DerivedRunMetrics;
}

interface BenchmarkGroupSummary {
  wall: BenchmarkSummary;
  avgTotalMs: number | null;
  avgPromptEvalMs: number | null;
  avgDecodeEvalMs: number | null;
  avgSampleMs: number | null;
  avgPromptEvalTokens: number | null;
  avgOutputTokenCount: number | null;
  promptTokensPerSecond: number | null;
  decodeTokensPerSecond: number | null;
}

interface BenchmarkGroupResult {
  id: 'coldPrompt' | 'hotFreshContext' | 'hotReuseContext';
  label: string;
  warmupRuns: number;
  measuredRuns: number;
  summary: BenchmarkGroupSummary;
  runs: BenchmarkRun[];
}

interface ScenarioRuntimeMetrics {
  initModuleMs: number;
  loadModelIntoMemfsMs: number;
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
  schemaVersion: 'cogent.benchmark.bun.v1';
  generatedAt: string;
  benchmark: {
    script: string;
    preset: BenchmarkPresetName;
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
  runtime: {
    readModelMs: number;
    scenarioInitSummary: {
      initModuleMs: BenchmarkSummary;
      loadModelIntoMemfsMs: BenchmarkSummary;
      initEngineMs: BenchmarkSummary;
    };
  };
  memory: RuntimeMemoryUsage;
  scenarios: BenchmarkScenarioResult[];
  limitations: string[];
}

const DEFAULT_PROMPT = 'Write one sentence about measuring inference performance.';
const MEDIUM_PROMPT =
  'Summarize a browser-hosted LLM runtime benchmark plan. Mention cold start, warm prompt latency, prompt evaluation throughput, decode throughput, and why reused-context measurement matters.';
const LONG_PROMPT = [
  'You are evaluating a browser-hosted inference runtime built with TypeScript, WebAssembly, and llama.cpp.',
  'Describe how you would benchmark cold start, module initialization, model load, engine initialization, prompt evaluation throughput, decode throughput, and reused-context performance.',
  'Keep the answer concise but cover why prompt length and output length should be swept separately.',
].join(' ');

const DEFAULT_TOKENS = 16;
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
    id: 'short-short',
    label: 'Short Prompt / Short Output',
    prompt: DEFAULT_PROMPT,
    outputTokenLimit: 16,
  },
  {
    id: 'medium-short',
    label: 'Medium Prompt / Short Output',
    prompt: MEDIUM_PROMPT,
    outputTokenLimit: 16,
  },
  {
    id: 'long-short',
    label: 'Long Prompt / Short Output',
    prompt: LONG_PROMPT,
    outputTokenLimit: 16,
  },
  {
    id: 'medium-medium',
    label: 'Medium Prompt / Medium Output',
    prompt: MEDIUM_PROMPT,
    outputTokenLimit: 64,
  },
] as const;

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

function parsePositiveInt(flagName: string, rawValue: string | undefined, fallback: number): number {
  return parseOptionalPositiveInt(flagName, rawValue) ?? fallback;
}

function parsePreset(rawPreset: string | undefined): BenchmarkPresetName {
  const preset = (rawPreset ?? 'default').trim();
  if (preset === 'single' || preset === 'default') {
    return preset;
  }
  throw new Error(`Unsupported preset "${preset}". Use "default" or "single".`);
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
    warmupRuns: parsePositiveInt('--warmup', options.get('warmup'), DEFAULT_WARMUP_RUNS),
    measuredRuns: parsePositiveInt('--runs', options.get('runs'), DEFAULT_MEASURED_RUNS),
    jsonPath: options.get('json'),
    artifactLabel: options.get('artifact-label'),
    quantizationLabel: options.get('quantization'),
  };
}

function printHelp(): void {
  console.log(`Usage: bun ./benchmarks/benchmark-bun.ts [options]

Options:
  --model <path>            Path to a GGUF model file
  --preset <name>           Benchmark preset: default | single (default: default)
  --prompt <text>           Prompt text for --preset single
  --tokens <n>              Max generation tokens per run or preset override
  --warmup <n>              Warmup runs per benchmark group (default: ${DEFAULT_WARMUP_RUNS})
  --runs <n>                Measured runs per benchmark group (default: ${DEFAULT_MEASURED_RUNS})
  --artifact-label <text>   Optional artifact label override
  --quantization <text>     Optional quantization label override
  --json <path>             Optional JSON output path
  --help                    Show this message

Presets:
  default  Standard matrix: short/medium/long prompt buckets plus medium-output sweep.
  single   One prompt with cold, hot fresh-context, and hot reused-context groups.
`);
}

function nowMs(): number {
  return performance.now();
}

async function measureAsync<T>(label: string, fn: () => Promise<T> | T): Promise<{ label: string; ms: number; value: T }> {
  const start = nowMs();
  const value = await fn();
  return { label, ms: nowMs() - start, value };
}

function formatBytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
}

function round(value: number): number {
  return Number(value.toFixed(3));
}

function summarize(values: number[]): BenchmarkSummary {
  const sorted = [...values].sort((left, right) => left - right);
  const total = sorted.reduce((acc, value) => acc + value, 0);
  const percentileIndex = Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1);

  return {
    minMs: round(sorted[0]),
    medianMs: round(sorted[Math.floor(sorted.length / 2)]),
    meanMs: round(total / sorted.length),
    p95Ms: round(sorted[percentileIndex]),
    maxMs: round(sorted[sorted.length - 1]),
  };
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
  if (perf.promptEvalMs <= 0 || perf.promptEvalTokens <= 0) {
    return null;
  }
  return (perf.promptEvalTokens * 1000) / perf.promptEvalMs;
}

function decodeTokensPerSecond(perf: PromptPerformanceStats): number | null {
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

async function initializeScenarioEngine(
  runtimeUrls: ReturnType<typeof getBundledRuntimeUrls>,
  modelBytes: Uint8Array,
  fileName: string
): Promise<{ engine: CogentEngine; runtime: ScenarioRuntimeMetrics }> {
  const engine = new CogentEngine(runtimeUrls);

  try {
    const initModule = await measureAsync('initModule', () => engine.initModule());
    const loadModel = await measureAsync('loadModelFromBuffer', () =>
      engine.loadModelFromBuffer(modelBytes, fileName)
    );
    const initEngine = await measureAsync('initEngine', () => engine.initEngine(loadModel.value));

    return {
      engine,
      runtime: {
        initModuleMs: round(initModule.ms),
        loadModelIntoMemfsMs: round(loadModel.ms),
        initEngineMs: round(initEngine.ms),
      },
    };
  } catch (error) {
    engine.close();
    throw error;
  }
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
    const prompt = options.prompt ?? DEFAULT_PROMPT;
    const outputTokenLimit = options.tokensOverride ?? DEFAULT_TOKENS;

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
      contextBucket: 'single-request',
      concurrency: 1,
    };
  });
}

function deriveRunMetrics(perf: PromptPerformanceStats | null): DerivedRunMetrics {
  if (!perf) {
    return {
      ttftMs: null,
      promptTokensPerSecond: null,
      decodeTokensPerSecond: null,
    };
  }

  return {
    ttftMs: null,
    promptTokensPerSecond: promptTokensPerSecond(perf),
    decodeTokensPerSecond: decodeTokensPerSecond(perf),
  };
}

async function runPromptBenchmark(
  engine: CogentEngine,
  labelPrefix: string,
  prompt: string,
  tokens: number,
  warmupRuns: number,
  measuredRuns: number,
  contextKeyFactory: (index: number) => string
): Promise<BenchmarkRun[]> {
  for (let i = 0; i < warmupRuns; i++) {
    await engine.prompt(contextKeyFactory(i), prompt, tokens);
  }

  const runs: BenchmarkRun[] = [];
  for (let i = 0; i < measuredRuns; i++) {
    const label = `${labelPrefix}-${i + 1}`;
    const contextKey = contextKeyFactory(i + warmupRuns);
    const start = nowMs();
    const output = await engine.prompt(contextKey, prompt, tokens);
    const wallMs = nowMs() - start;
    const perf = engine.getLastPromptPerformance();
    if (output.length === 0 && perf == null) {
      throw new Error(
        `Prompt run "${label}" returned empty output and no perf payload. The runtime likely failed to create or execute the request context.`
      );
    }

    runs.push({
      label,
      contextKey,
      wallMs: round(wallMs),
      outputLength: output.length,
      outputPreview: output.slice(0, OUTPUT_PREVIEW_LIMIT).replace(/\s+/g, ' ').trim(),
      perf,
      derived: deriveRunMetrics(perf),
    });
  }

  return runs;
}

function summarizeGroup(runs: BenchmarkRun[]): BenchmarkGroupSummary {
  const perfRuns = runs.map((run) => run.perf);

  return {
    wall: summarize(runs.map((run) => run.wallMs)),
    avgTotalMs: averagePerfMetric(perfRuns, (perf) => perf.totalMs),
    avgPromptEvalMs: averagePerfMetric(perfRuns, (perf) => perf.promptEvalMs),
    avgDecodeEvalMs: averagePerfMetric(perfRuns, (perf) => perf.decodeEvalMs),
    avgSampleMs: averagePerfMetric(perfRuns, (perf) => perf.sampleMs),
    avgPromptEvalTokens: averagePerfMetric(perfRuns, (perf) => perf.promptEvalTokens),
    avgOutputTokenCount: averagePerfMetric(perfRuns, (perf) => perf.outputTokenCount),
    promptTokensPerSecond: summarizeThroughput(perfRuns, promptTokensPerSecond),
    decodeTokensPerSecond: summarizeThroughput(perfRuns, decodeTokensPerSecond),
  };
}

function createGroupResult(
  id: BenchmarkGroupResult['id'],
  label: string,
  warmupRuns: number,
  measuredRuns: number,
  runs: BenchmarkRun[]
): BenchmarkGroupResult {
  return {
    id,
    label,
    warmupRuns,
    measuredRuns,
    summary: summarizeGroup(runs),
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
    scenario.outputTokenLimit,
    0,
    1,
    () => `${scenario.id}-cold`
  );

  const hotFreshContext = await runPromptBenchmark(
    engine,
    `${scenario.id}-hot-fresh`,
    scenario.prompt,
    scenario.outputTokenLimit,
    warmupRuns,
    measuredRuns,
    (index) => `${scenario.id}-fresh-${index}`
  );

  const hotReuseContext = await runPromptBenchmark(
    engine,
    `${scenario.id}-hot-reuse`,
    scenario.prompt,
    scenario.outputTokenLimit,
    warmupRuns,
    measuredRuns,
    () => `${scenario.id}-reuse`
  );

  return {
    definition: scenario,
    runtime,
    coldPrompt: createGroupResult('coldPrompt', 'Cold Prompt', 0, 1, coldPrompt),
    hotFreshContext: createGroupResult(
      'hotFreshContext',
      'Hot Prompt: Fresh Context',
      warmupRuns,
      measuredRuns,
      hotFreshContext
    ),
    hotReuseContext: createGroupResult(
      'hotReuseContext',
      'Hot Prompt: Reused Context',
      warmupRuns,
      measuredRuns,
      hotReuseContext
    ),
  };
}

function printGroupResult(group: BenchmarkGroupResult): void {
  const summary = group.summary;

  console.log(`\n  ${group.label}`);
  console.log(
    `    wall ms          min=${summary.wall.minMs} median=${summary.wall.medianMs} mean=${summary.wall.meanMs} p95=${summary.wall.p95Ms} max=${summary.wall.maxMs}`
  );

  if (summary.promptTokensPerSecond != null) {
    console.log(`    prompt tok/s     avg=${summary.promptTokensPerSecond}`);
  }

  if (summary.decodeTokensPerSecond != null) {
    console.log(`    decode tok/s     avg=${summary.decodeTokensPerSecond}`);
  }

  if (
    summary.avgTotalMs != null &&
    summary.avgPromptEvalMs != null &&
    summary.avgDecodeEvalMs != null &&
    summary.avgSampleMs != null &&
    summary.avgOutputTokenCount != null
  ) {
    console.log(
      `    native perf      total_ms=${summary.avgTotalMs} prompt_eval_ms=${summary.avgPromptEvalMs} decode_eval_ms=${summary.avgDecodeEvalMs} sample_ms=${summary.avgSampleMs} output_tokens=${summary.avgOutputTokenCount}`
    );
  }
}

function printScenarioResult(result: BenchmarkScenarioResult): void {
  const definition = result.definition;
  console.log(`\nScenario: ${definition.label}`);
  console.log(
    `  init module=${result.runtime.initModuleMs} ms load model=${result.runtime.loadModelIntoMemfsMs} ms init engine=${result.runtime.initEngineMs} ms`
  );
  console.log(
    `  prompt bucket=${definition.promptBucket} chars=${definition.promptChars} words=${definition.promptWords}`
  );
  console.log(
    `  output bucket=${definition.outputBucket} token_limit=${definition.outputTokenLimit} concurrency=${definition.concurrency}`
  );

  printGroupResult(result.coldPrompt);
  printGroupResult(result.hotFreshContext);
  printGroupResult(result.hotReuseContext);
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
  console.log(`  preset     ${options.preset}`);
  console.log(`  scenarios   ${scenarios.length}`);
  console.log(`  model      ${options.modelPath}`);
  console.log(`  artifact   ${artifactLabel}`);
  console.log(`  quant      ${quantizationLabel ?? 'unknown'}`);
  console.log(`  warmup     ${options.warmupRuns}`);
  console.log(`  runs       ${options.measuredRuns}`);

  let report: BenchmarkReport | null = null;

  const readModel = await measureAsync('readModel', async () => {
    const file = Bun.file(options.modelPath);
    const bytes = new Uint8Array(await file.arrayBuffer());
    return {
      bytes,
      size: bytes.byteLength,
    };
  });

  const modelBytes = readModel.value.bytes;
  console.log(`\nRuntime`);
  console.log(`  model read ms    ${round(readModel.ms)}`);
  console.log(`  model size       ${formatBytes(readModel.value.size)}`);

  const scenarioResults: BenchmarkScenarioResult[] = [];
  for (const scenario of scenarios) {
    const { engine, runtime } = await initializeScenarioEngine(
      runtimeUrls,
      modelBytes,
      fileName
    );

    try {
      const scenarioResult = await runScenarioBenchmark(
        engine,
        scenario,
        runtime,
        options.warmupRuns,
        options.measuredRuns
      );
      printScenarioResult(scenarioResult);
      scenarioResults.push(scenarioResult);
    } finally {
      engine.close();
    }
  }

  const maybeNavigator =
    typeof navigator !== 'undefined'
      ? (navigator as { gpu?: unknown; userAgent?: string })
      : null;

  report = {
    schemaVersion: 'cogent.benchmark.bun.v1',
    generatedAt: new Date().toISOString(),
    benchmark: {
      script: 'packages/cogent-engine/benchmarks/benchmark-bun.ts',
      preset: options.preset,
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
    runtime: {
      readModelMs: round(readModel.ms),
      scenarioInitSummary: {
        initModuleMs: summarize(scenarioResults.map((scenario) => scenario.runtime.initModuleMs)),
        loadModelIntoMemfsMs: summarize(
          scenarioResults.map((scenario) => scenario.runtime.loadModelIntoMemfsMs)
        ),
        initEngineMs: summarize(scenarioResults.map((scenario) => scenario.runtime.initEngineMs)),
      },
    },
    memory: captureMemoryUsage(),
    scenarios: scenarioResults,
    limitations: [
      'This Bun track is authoritative for Wasm host/runtime overhead, not browser WebGPU kernel behavior.',
      'TTFT is not measured separately in the current non-streaming API and is reported as null.',
      'Concurrency is fixed at 1 until the slot scheduler phases are implemented.',
      'Each scenario uses a fresh engine instance so context allocation from one case does not poison the next case.',
    ],
  };

  if (options.jsonPath) {
    await writeJsonReport(options.jsonPath, report);
  }

  if (!report) {
    throw new Error('Benchmark did not produce a report.');
  }
}

await main();
