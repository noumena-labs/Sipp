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

interface BenchmarkRun {
  label: string;
  wallMs: number;
  outputLength: number;
  outputPreview: string;
  perf: PromptPerformanceStats | null;
}

interface BenchmarkSummary {
  minMs: number;
  medianMs: number;
  meanMs: number;
  p95Ms: number;
  maxMs: number;
}

interface BenchmarkOptions {
  modelPath: string;
  prompt: string;
  tokens: number;
  warmupRuns: number;
  measuredRuns: number;
  jsonPath?: string;
}

const DEFAULT_PROMPT = 'Write one sentence about measuring inference performance.';
const DEFAULT_TOKENS = 16;
const DEFAULT_WARMUP_RUNS = 1;
const DEFAULT_MEASURED_RUNS = 3;

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.resolve(scriptDir, '..');
const repoRoot = path.resolve(packageRoot, '..', '..');

function parseNumberFlag(flagName: string, rawValue: string | undefined, fallback: number): number {
  if (rawValue == null) {
    return fallback;
  }

  const value = Number.parseInt(rawValue, 10);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Expected a positive integer for ${flagName}, got "${rawValue}".`);
  }
  return value;
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

  throw new Error(
    'No GGUF model found. Pass --model <path> or set COGENT_BENCH_MODEL.'
  );
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

  return {
    modelPath: resolveModelPath(options.get('model')),
    prompt: options.get('prompt') ?? DEFAULT_PROMPT,
    tokens: parseNumberFlag('--tokens', options.get('tokens'), DEFAULT_TOKENS),
    warmupRuns: parseNumberFlag('--warmup', options.get('warmup'), DEFAULT_WARMUP_RUNS),
    measuredRuns: parseNumberFlag('--runs', options.get('runs'), DEFAULT_MEASURED_RUNS),
    jsonPath: options.get('json'),
  };
}

function printHelp(): void {
  console.log(`Usage: bun ./benchmarks/benchmark-bun.ts [options]

Options:
  --model <path>    Path to a GGUF model file
  --prompt <text>   Prompt text to benchmark
  --tokens <n>      Max generation tokens per run (default: ${DEFAULT_TOKENS})
  --warmup <n>      Warmup runs per benchmark group (default: ${DEFAULT_WARMUP_RUNS})
  --runs <n>        Measured runs per benchmark group (default: ${DEFAULT_MEASURED_RUNS})
  --json <path>     Optional JSON output path
  --help            Show this message
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

function summarizeThroughput(perfRuns: Array<PromptPerformanceStats | null>): number | null {
  const values = perfRuns
    .filter((perf): perf is PromptPerformanceStats => perf !== null)
    .map((perf) => {
      if (perf.decodeEvalMs <= 0 || perf.outputTokenCount <= 0) {
        return 0;
      }
      return (perf.outputTokenCount * 1000) / perf.decodeEvalMs;
    })
    .filter((value) => value > 0);

  if (values.length === 0) {
    return null;
  }

  const total = values.reduce((acc, value) => acc + value, 0);
  return round(total / values.length);
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

function captureMemoryUsage() {
  const usage = process.memoryUsage();
  return {
    rssBytes: usage.rss,
    heapUsedBytes: usage.heapUsed,
    externalBytes: usage.external,
    arrayBuffersBytes: usage.arrayBuffers,
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

    runs.push({
      label,
      wallMs: round(wallMs),
      outputLength: output.length,
      outputPreview: output.slice(0, 120).replace(/\s+/g, ' ').trim(),
      perf,
    });
  }

  return runs;
}

function printRunGroup(title: string, runs: BenchmarkRun[]): void {
  const wallSummary = summarize(runs.map((run) => run.wallMs));
  const perfRuns = runs.map((run) => run.perf);
  const avgTokensPerSecond = summarizeThroughput(perfRuns);
  const avgPromptEvalMs = averagePerfMetric(perfRuns, (perf) => perf.promptEvalMs);
  const avgDecodeEvalMs = averagePerfMetric(perfRuns, (perf) => perf.decodeEvalMs);
  const avgSampleMs = averagePerfMetric(perfRuns, (perf) => perf.sampleMs);
  const avgOutputTokenCount = averagePerfMetric(perfRuns, (perf) => perf.outputTokenCount);

  console.log(`\n${title}`);
  console.log(`  wall ms      min=${wallSummary.minMs} median=${wallSummary.medianMs} mean=${wallSummary.meanMs} p95=${wallSummary.p95Ms} max=${wallSummary.maxMs}`);
  if (avgTokensPerSecond != null) {
    console.log(`  decode tok/s avg=${avgTokensPerSecond}`);
  }
  if (avgPromptEvalMs != null && avgDecodeEvalMs != null && avgSampleMs != null && avgOutputTokenCount != null) {
    console.log(
      `  native perf  prompt_eval_ms=${avgPromptEvalMs} decode_eval_ms=${avgDecodeEvalMs} sample_ms=${avgSampleMs} output_tokens=${avgOutputTokenCount}`
    );
  }
}

async function main(): Promise<void> {
  const options = parseArgs(Bun.argv.slice(2));
  const runtime = getBundledRuntimeUrls();
  const engine = new CogentEngine(runtime);

  const report: Record<string, unknown> = {
    environment: {
      bunVersion: Bun.version,
      platform: process.platform,
      arch: process.arch,
    },
    config: {
      modelPath: options.modelPath,
      prompt: options.prompt,
      tokens: options.tokens,
      warmupRuns: options.warmupRuns,
      measuredRuns: options.measuredRuns,
    },
  };

  console.log('Bun inference benchmark');
  console.log(`  model   ${options.modelPath}`);
  console.log(`  tokens  ${options.tokens}`);
  console.log(`  warmup  ${options.warmupRuns}`);
  console.log(`  runs    ${options.measuredRuns}`);

  try {
    const readModel = await measureAsync('readModel', async () => {
      const file = Bun.file(options.modelPath);
      const bytes = new Uint8Array(await file.arrayBuffer());
      return {
        bytes,
        size: bytes.byteLength,
      };
    });

    let modelBytes = readModel.value.bytes;
    console.log(`\nRuntime`);
    console.log(`  model read ms   ${round(readModel.ms)}`);
    console.log(`  model size      ${formatBytes(readModel.value.size)}`);

    const initModule = await measureAsync('initModule', () => engine.initModule());
    console.log(`  module init ms  ${round(initModule.ms)}`);

    const loadModel = await measureAsync('loadModelFromBuffer', () =>
      engine.loadModelFromBuffer(modelBytes, path.basename(options.modelPath))
    );
    console.log(`  MEMFS load ms   ${round(loadModel.ms)}`);

    modelBytes = new Uint8Array();

    const initEngine = await measureAsync('initEngine', () => engine.initEngine(loadModel.value));
    console.log(`  engine init ms  ${round(initEngine.ms)}`);

    const coldPrompt = await runPromptBenchmark(
      engine,
      'cold',
      options.prompt,
      options.tokens,
      0,
      1,
      () => 'bench-cold'
    );
    printRunGroup('Cold Prompt', coldPrompt);

    const hotFresh = await runPromptBenchmark(
      engine,
      'hot-fresh',
      options.prompt,
      options.tokens,
      options.warmupRuns,
      options.measuredRuns,
      (index) => `bench-fresh-${index}`
    );
    printRunGroup('Hot Prompt: Fresh Context', hotFresh);

    const hotReuse = await runPromptBenchmark(
      engine,
      'hot-reuse',
      options.prompt,
      options.tokens,
      options.warmupRuns,
      options.measuredRuns,
      () => 'bench-reuse'
    );
    printRunGroup('Hot Prompt: Reused Context', hotReuse);

    report.runtime = {
      readModelMs: round(readModel.ms),
      modelBytes: readModel.value.size,
      initModuleMs: round(initModule.ms),
      loadModelIntoMemfsMs: round(loadModel.ms),
      initEngineMs: round(initEngine.ms),
    };
    report.memory = captureMemoryUsage();
    report.coldPrompt = coldPrompt;
    report.hotFreshContext = hotFresh;
    report.hotReuseContext = hotReuse;

    if (options.jsonPath) {
      const jsonPath = path.isAbsolute(options.jsonPath)
        ? options.jsonPath
        : path.resolve(process.cwd(), options.jsonPath);
      await Bun.write(jsonPath, `${JSON.stringify(report, null, 2)}\n`);
      console.log(`\nSaved JSON report to ${jsonPath}`);
    }
  } finally {
    engine.close();
  }
}

await main();
