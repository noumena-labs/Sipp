import { CogentEngine, getBundledRuntimeUrls } from 'cogent-engine';
import './style.css';

// Browser benchmark metric glossary:
// - TTFT: request start until the first streamed token callback.
// - TPOT: average time per generated token after the first token.
// - ITL: token-to-token gaps measured from streamed token callbacks.
// - E2EL: full request latency until the final streamed token is received.
// - Request/output/total throughput: aggregate serving metrics over the measured group.
// - Logical input tokens vs effective prompt-eval tokens:
//   logical input size is the full request prompt size, while effective prompt-eval
//   work is what llama.cpp actually had to prefill after any context reuse.

const SHORT_PROMPT = 'Write one sentence about measuring inference performance.';
const LONG_PROMPT = [
  'You are evaluating a browser-hosted inference runtime built with TypeScript, WebAssembly, and llama.cpp.',
  'Describe how you would benchmark cold start, module initialization, model load, engine initialization, prompt evaluation throughput, decode throughput, reused-context performance, and TTFT.',
  'Keep the answer concise but explain why prompt length and output length should be swept separately.',
].join(' ');

const DEFAULT_SHORT_OUTPUT_TOKENS = 16;
const DEFAULT_LONG_OUTPUT_TOKENS = 128;
const DEFAULT_BENCHMARK_SCENARIOS = [
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
];

const DEFAULT_MIXED_LOAD_DEFINITION = {
  id: 'mixed-lilo-vs-siso',
  label: 'Mixed Load: LILO Background vs SISO Foreground',
  background: {
    id: 'mixed-background-lilo',
    label: 'Background Long Input / Long Output',
    prompt: LONG_PROMPT,
    outputTokenLimit: DEFAULT_LONG_OUTPUT_TOKENS,
  },
  foreground: {
    id: 'mixed-foreground-siso',
    label: 'Foreground Short Input / Short Output',
    prompt: SHORT_PROMPT,
    outputTokenLimit: DEFAULT_SHORT_OUTPUT_TOKENS,
  },
  concurrency: 2,
};

const app = document.querySelector('#app');
app.innerHTML = `
  <div class="shell">
    <header class="hero">
      <div class="eyebrow">Browser Benchmark</div>
      <h1>CogentEngine Benchmark App</h1>
      <p>
        Browser-hosted benchmark harness for the real WebGPU inference path.
        This app is benchmark-only: no decorative Three.js scene, no fallback rendering layer,
        and no demo-specific behavior outside runtime validation and reporting.
      </p>
    </header>

    <div class="layout">
      <div class="column">
        <section class="section">
          <div class="section-header">
            <h2>Environment</h2>
          </div>
          <div id="environment" class="metric-grid">
            <div class="metric-card">
              <span class="metric-label">Browser</span>
              <span class="metric-value">collecting...</span>
            </div>
          </div>
        </section>

        <section class="section">
          <div class="section-header">
            <h2>Model Source</h2>
          </div>
          <div class="field-grid">
            <div class="row">
              <label for="modelUrl">Model URL</label>
              <input id="modelUrl" placeholder="https://.../model.gguf" />
            </div>
            <div class="row">
              <label for="modelFile">Local GGUF File</label>
              <input id="modelFile" type="file" accept=".gguf" />
            </div>
          </div>
          <div class="button-row">
            <button id="initRuntimeBtn" type="button">Init Runtime</button>
            <button id="loadModelBtn" type="button">Load Model + Init Engine</button>
          </div>
        </section>

        <section class="section">
          <div class="section-header">
            <h2>Prompt</h2>
          </div>
          <div class="field-grid">
            <div class="row">
              <label for="promptText">Prompt Text</label>
              <textarea id="promptText">Describe how to benchmark browser-hosted inference.</textarea>
            </div>
          </div>
          <div class="field-grid field-grid-compact">
            <div class="row">
              <label for="tokenCount">Max Tokens</label>
              <input id="tokenCount" type="number" min="1" max="512" value="64" />
            </div>
            <div class="row">
              <label for="warmupRuns">Warmup Runs</label>
              <input id="warmupRuns" type="number" min="0" max="10" value="1" />
            </div>
            <div class="row">
              <label for="benchmarkRuns">Measured Runs</label>
              <input id="benchmarkRuns" type="number" min="1" max="10" value="3" />
            </div>
          </div>
          <div class="field-grid field-grid-compact">
            <div class="row">
              <label for="prefillChunkSize">Prefill Chunk</label>
              <input id="prefillChunkSize" type="number" min="0" max="512" value="0" />
            </div>
            <div class="row">
              <label for="schedulerPolicy">Scheduler Policy</label>
              <select id="schedulerPolicy">
                <option value="latency-first">latency-first</option>
                <option value="balanced" selected>balanced</option>
                <option value="throughput-first">throughput-first</option>
              </select>
            </div>
            <div class="row">
              <label for="decodeTokenReserve">Decode Reserve</label>
              <input id="decodeTokenReserve" type="number" min="0" max="64" value="1" />
            </div>
          </div>
          <div class="button-row">
            <button id="runPromptBtn" type="button">Run Single Inference</button>
            <button id="runBenchmarkBtn" type="button">Run Full Browser Benchmark</button>
          </div>
        </section>

        <p id="status" class="status">Status: idle</p>
      </div>

      <div class="column">
        <section class="section">
          <div class="section-header">
            <h2>Response</h2>
          </div>
          <div id="responseMeta" class="metric-grid">
            <div class="metric-card">
              <span class="metric-label">Last Run</span>
              <span class="metric-value">No inference yet</span>
            </div>
          </div>
          <div id="response" class="response"></div>
        </section>

        <section class="section">
          <div class="section-header">
            <h2>Benchmark Report</h2>
            <button id="downloadReportBtn" class="secondary-button" type="button" disabled>
              Download JSON
            </button>
          </div>
          <div id="benchmarkResults" class="benchmark-results">
            <p class="hint">
              Run the browser benchmark to execute the standard four-case matrix:
              SISO, SILO, LISO, and LILO, each with cold, hot-fresh, and hot-reuse measurements,
              plus one mixed-load fairness run with a LILO background request and a SISO foreground request.
            </p>
          </div>
        </section>
      </div>
    </div>
  </div>
`;

function createEngine() {
  return new CogentEngine(getBundledRuntimeUrls());
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function round(value) {
  return Number(value.toFixed(3));
}

function formatMs(value) {
  return `${round(value)} ms`;
}

function formatMiB(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
}

function formatBytes(bytes) {
  if (bytes == null || !Number.isFinite(bytes) || bytes < 0) {
    return 'n/a';
  }
  if (bytes >= 1024 * 1024) {
    return formatMiB(bytes);
  }
  if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(2)} KiB`;
  }
  return `${bytes} B`;
}

function countWords(text) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function classifyPromptBucket(prompt) {
  const wordCount = countWords(prompt);
  if (wordCount <= 16) {
    return 'short';
  }
  if (wordCount <= 64) {
    return 'medium';
  }
  return 'long';
}

function classifyOutputBucket(tokenCount) {
  if (tokenCount <= 32) {
    return 'short';
  }
  if (tokenCount <= 96) {
    return 'medium';
  }
  return 'long';
}

function buildBenchmarkScenarios() {
  return DEFAULT_BENCHMARK_SCENARIOS.map((scenario) => ({
    id: scenario.id,
    label: scenario.label,
    prompt: scenario.prompt,
    promptBucket: classifyPromptBucket(scenario.prompt),
    promptChars: scenario.prompt.length,
    promptWords: countWords(scenario.prompt),
    outputTokenLimit: scenario.outputTokenLimit,
    outputBucket: classifyOutputBucket(scenario.outputTokenLimit),
  }));
}

function buildMixedLoadDefinition() {
  return {
    id: DEFAULT_MIXED_LOAD_DEFINITION.id,
    label: DEFAULT_MIXED_LOAD_DEFINITION.label,
    background: {
      ...DEFAULT_MIXED_LOAD_DEFINITION.background,
      promptBucket: classifyPromptBucket(DEFAULT_MIXED_LOAD_DEFINITION.background.prompt),
      promptChars: DEFAULT_MIXED_LOAD_DEFINITION.background.prompt.length,
      promptWords: countWords(DEFAULT_MIXED_LOAD_DEFINITION.background.prompt),
      outputBucket: classifyOutputBucket(DEFAULT_MIXED_LOAD_DEFINITION.background.outputTokenLimit),
      promptFormat: 'auto-chat',
      contextBucket: 'single-request',
      concurrency: 1,
    },
    foreground: {
      ...DEFAULT_MIXED_LOAD_DEFINITION.foreground,
      promptBucket: classifyPromptBucket(DEFAULT_MIXED_LOAD_DEFINITION.foreground.prompt),
      promptChars: DEFAULT_MIXED_LOAD_DEFINITION.foreground.prompt.length,
      promptWords: countWords(DEFAULT_MIXED_LOAD_DEFINITION.foreground.prompt),
      outputBucket: classifyOutputBucket(DEFAULT_MIXED_LOAD_DEFINITION.foreground.outputTokenLimit),
      promptFormat: 'auto-chat',
      contextBucket: 'single-request',
      concurrency: 1,
    },
    concurrency: DEFAULT_MIXED_LOAD_DEFINITION.concurrency,
  };
}

function buildPhase4BenchmarkInitConfig(initConfig = {}) {
  return {
    ...initConfig,
    nSeqMax: Math.max(initConfig.nSeqMax ?? 1, 2),
    maxCachedSessions: Math.max(initConfig.maxCachedSessions ?? 8, 2),
  };
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function metricCard(label, value, tone = 'default') {
  return `
    <div class="metric-card ${tone !== 'default' ? `metric-card-${tone}` : ''}">
      <span class="metric-label">${escapeHtml(label)}</span>
      <span class="metric-value">${escapeHtml(value)}</span>
    </div>
  `;
}

function summarize(values) {
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

function summarizeOptional(values) {
  const filtered = values.filter((value) => value != null && Number.isFinite(value));
  return filtered.length === 0 ? null : summarize(filtered);
}

function averagePerfMetric(perfRuns, metric) {
  const values = perfRuns
    .filter((perf) => perf !== null)
    .map(metric)
    .filter((value) => Number.isFinite(value) && value >= 0);

  if (values.length === 0) {
    return null;
  }

  const total = values.reduce((acc, value) => acc + value, 0);
  return round(total / values.length);
}

function summarizePromptThroughput(perfRuns) {
  const values = perfRuns
    .filter((perf) => perf !== null)
    .map((perf) => {
      if (perf.promptEvalMs <= 0 || perf.promptEvalTokens <= 0) {
        return 0;
      }
      return (perf.promptEvalTokens * 1000) / perf.promptEvalMs;
    })
    .filter((value) => value > 0);

  if (values.length === 0) {
    return null;
  }

  const total = values.reduce((acc, value) => acc + value, 0);
  return round(total / values.length);
}

function summarizeDecodeThroughput(perfRuns) {
  const values = perfRuns
    .filter((perf) => perf !== null)
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

function maxNullable(values) {
  const filtered = values.filter((value) => value != null && Number.isFinite(value));
  if (filtered.length === 0) {
    return null;
  }
  return Math.max(...filtered);
}

async function captureBrowserMemorySnapshot(label) {
  const snapshot = {
    label,
    capturedAt: new Date().toISOString(),
    source: 'unavailable',
    usedJsHeapBytes: null,
    totalJsHeapBytes: null,
    jsHeapLimitBytes: null,
    userAgentBytes: null,
    error: null,
  };

  if (typeof performance !== 'undefined' && performance.memory) {
    snapshot.source = 'performance.memory';
    snapshot.usedJsHeapBytes = performance.memory.usedJSHeapSize ?? null;
    snapshot.totalJsHeapBytes = performance.memory.totalJSHeapSize ?? null;
    snapshot.jsHeapLimitBytes = performance.memory.jsHeapSizeLimit ?? null;
  }

  if (typeof performance !== 'undefined' && typeof performance.measureUserAgentSpecificMemory === 'function') {
    try {
      const uaMemory = await performance.measureUserAgentSpecificMemory();
      snapshot.userAgentBytes = uaMemory.bytes ?? null;
      snapshot.source =
        snapshot.source === 'performance.memory'
          ? 'performance.memory + measureUserAgentSpecificMemory'
          : 'measureUserAgentSpecificMemory';
    } catch (error) {
      snapshot.error = errorMessage(error);
    }
  }

  return snapshot;
}

function summarizeMemorySnapshots(memorySnapshots) {
  return {
    snapshotCount: memorySnapshots.length,
    maxUsedJsHeapBytes: maxNullable(memorySnapshots.map((snapshot) => snapshot.usedJsHeapBytes)),
    maxTotalJsHeapBytes: maxNullable(memorySnapshots.map((snapshot) => snapshot.totalJsHeapBytes)),
    maxUserAgentBytes: maxNullable(memorySnapshots.map((snapshot) => snapshot.userAgentBytes)),
    finalSnapshot: memorySnapshots.length > 0 ? memorySnapshots[memorySnapshots.length - 1] : null,
  };
}

function summarizeRunGroup(runs, benchmarkDurationMs) {
  const perfRuns = runs.map((run) => run.perf);
  const totalInputTokens = runs.reduce((acc, run) => acc + (run.inputTokenCount ?? 0), 0);
  const totalGeneratedTokens = runs.reduce((acc, run) => acc + (run.outputTokenCount ?? 0), 0);
  const allItls = runs.flatMap((run) => run.itlMsValues);
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
      itlMs: summarizeOptional(allItls),
      e2elMs: summarize(runs.map((run) => run.wallMs)),
    },
    runtime: {
      avgLogicalInputTokenCount: averagePerfMetric(perfRuns, (perf) => perf.inputTokenCount),
      avgPromptEvalTokens: averagePerfMetric(perfRuns, (perf) => perf.promptEvalTokens),
      avgPromptEvalMs: averagePerfMetric(perfRuns, (perf) => perf.promptEvalMs),
      avgDecodeEvalMs: averagePerfMetric(perfRuns, (perf) => perf.decodeEvalMs),
      avgSampleMs: averagePerfMetric(perfRuns, (perf) => perf.sampleMs),
      avgOutputTokenCount: averagePerfMetric(perfRuns, (perf) => perf.outputTokenCount),
      avgQueueDelayMs: averagePerfMetric(perfRuns, (perf) => perf.queueDelayMs),
      avgTailItlMs: averagePerfMetric(perfRuns, (perf) => perf.tailItlMs),
      avgSchedulerTickCount: averagePerfMetric(perfRuns, (perf) => perf.schedulerTickCount),
      avgBatchParticipationCount: averagePerfMetric(perfRuns, (perf) => perf.batchParticipationCount),
      avgDecodeFirstTickCount: averagePerfMetric(perfRuns, (perf) => perf.decodeFirstTickCount),
      avgChunkedPrefillTickCount: averagePerfMetric(perfRuns, (perf) => perf.chunkedPrefillTickCount),
      avgMixedWorkloadTickCount: averagePerfMetric(perfRuns, (perf) => perf.mixedWorkloadTickCount),
      promptTokensPerSecond: summarizePromptThroughput(perfRuns),
      decodeTokensPerSecond: summarizeDecodeThroughput(perfRuns),
    },
  };
}

function describeRuntimeBackend(info) {
  if (!info) {
    return 'runtime not initialized';
  }
  if (info.apiAvailable === false) {
    return 'backend-info API unavailable';
  }
  if (!info.webgpuCompiled) {
    return 'CPU-only build';
  }
  if (!info.webgpuRegistered) {
    return 'WebGPU backend unavailable at runtime';
  }
  return `WebGPU backend ready (${info.webgpuDeviceCount} device${info.webgpuDeviceCount === 1 ? '' : 's'})`;
}

function describeRuntimeDevices(info) {
  if (!info || !Array.isArray(info.devices) || info.devices.length === 0) {
    return 'none';
  }
  return info.devices
    .map((device) => `${device.backendName || device.type}:${device.description || device.name || device.type}`)
    .join(' | ');
}

function inferRequestedExecutionMode(runtimeBackend) {
  if (!runtimeBackend) {
    return 'unknown';
  }
  return runtimeBackend.webgpuRegistered ? 'gpu-offload' : 'cpu-only';
}

function inferRuntimeBackendStatus(runtimeBackend) {
  if (!runtimeBackend) {
    return 'unknown';
  }
  if (!runtimeBackend.webgpuCompiled) {
    return 'not-compiled';
  }
  if (!runtimeBackend.webgpuRegistered) {
    return 'compiled-not-registered';
  }
  if ((runtimeBackend.webgpuDeviceCount ?? 0) <= 0) {
    return 'registered-no-devices';
  }
  return 'webgpu-ready';
}

function inferExecutionBackend(environment, runtimeBackend) {
  if (!runtimeBackend) {
    return 'unknown';
  }
  if (
    environment?.adapterAvailable &&
    runtimeBackend.webgpuRegistered &&
    runtimeBackend.webgpuDeviceCount > 0 &&
    runtimeBackend.gpuOffloadSupported
  ) {
    return 'webgpu';
  }
  return 'cpu';
}

function buildBenchmarkBackendProfile(environment, runtimeBackend) {
  const runtimeDevices = Array.isArray(runtimeBackend?.devices) ? runtimeBackend.devices : [];
  const acceleratorDevices = runtimeDevices.filter((device) => device.type !== 'cpu');
  const notes = [];

  if (!environment?.hasNavigatorGpu) {
    notes.push('navigator.gpu is unavailable in this browser session.');
  } else if (!environment.adapterAvailable) {
    notes.push('navigator.gpu is present, but requestAdapter() did not produce a usable adapter.');
  }

  if (!runtimeBackend?.webgpuCompiled) {
    notes.push('The package build did not include ggml-webgpu.');
  } else if (!runtimeBackend.webgpuRegistered) {
    notes.push('ggml-webgpu was compiled, but the runtime did not register a usable WebGPU backend.');
  } else if ((runtimeBackend.webgpuDeviceCount ?? 0) <= 0) {
    notes.push('ggml-webgpu was registered, but it reported no runtime devices.');
  }

  return {
    requestedExecutionMode: inferRequestedExecutionMode(runtimeBackend),
    requestedGpuLayers: null,
    inferredExecutionBackend: inferExecutionBackend(environment, runtimeBackend),
    runtimeBackendStatus: inferRuntimeBackendStatus(runtimeBackend),
    gpuOffloadSupported: runtimeBackend?.gpuOffloadSupported ?? null,
    availableBackends: runtimeBackend?.availableBackends?.map((backend) => backend.name) ?? [],
    backendRegistries: runtimeBackend?.availableBackends ?? [],
    runtimeDeviceCount: runtimeDevices.length,
    runtimeAcceleratorDeviceCount: acceleratorDevices.length,
    runtimeDeviceLabels: runtimeDevices.map((device) => `${device.backendName || device.type}:${device.description || device.name || device.type}`),
    runtimeDevices,
    hostAdapter: {
      apiAvailable: Boolean(environment?.hasNavigatorGpu),
      adapterAvailable: Boolean(environment?.adapterAvailable),
      adapterLabel: environment?.adapterLabel ?? null,
      adapterVendor: environment?.adapterVendor ?? null,
      adapterArchitecture: environment?.adapterArchitecture ?? null,
      adapterDescription: environment?.adapterDescription ?? null,
    },
    notes,
  };
}

function benchmarkSection(title, group) {
  const { serving, runtime } = group.summary;
  const metrics = [
    metricCard('Req/s', serving.requestThroughputRps == null ? 'n/a' : String(serving.requestThroughputRps)),
    metricCard('Output tok/s', serving.outputTokenThroughputTps == null ? 'n/a' : String(serving.outputTokenThroughputTps)),
    metricCard('Total tok/s', serving.totalTokenThroughputTps == null ? 'n/a' : String(serving.totalTokenThroughputTps)),
    metricCard('Mean TTFT', serving.ttftMs == null ? 'n/a' : `${serving.ttftMs.meanMs} ms`),
    metricCard('Mean TPOT', serving.tpotMs == null ? 'n/a' : `${serving.tpotMs.meanMs} ms`),
    metricCard('Mean ITL', serving.itlMs == null ? 'n/a' : `${serving.itlMs.meanMs} ms`),
    metricCard('Mean E2EL', `${serving.e2elMs.meanMs} ms`),
    metricCard('Queue Delay', runtime.avgQueueDelayMs == null ? 'n/a' : `${runtime.avgQueueDelayMs} ms`),
    metricCard('Tail ITL', runtime.avgTailItlMs == null ? 'n/a' : `${runtime.avgTailItlMs} ms`),
    metricCard('Prompt Eval tok/s', runtime.promptTokensPerSecond == null ? 'n/a' : String(runtime.promptTokensPerSecond)),
  ].join('');

  const preview = group.runs[0]?.outputPreview?.trim() || '(empty response)';

  return `
    <article class="result-card">
      <h3>${escapeHtml(title)}</h3>
      <div class="metric-grid">${metrics}</div>
      <p class="result-detail">
        requests=${serving.successfulRequests}
        duration=${round(serving.benchmarkDurationMs / 1000)} s
        input_tokens=${serving.totalInputTokens}
        output_tokens=${serving.totalGeneratedTokens}
      </p>
      <p class="result-detail">
        e2el median=${serving.e2elMs.medianMs} ms
        p99=${serving.e2elMs.p99Ms} ms
        min=${serving.e2elMs.minMs} ms
        max=${serving.e2elMs.maxMs} ms
      </p>
      ${serving.ttftMs == null ? '' : `<p class="result-detail">ttft median=${serving.ttftMs.medianMs} ms p99=${serving.ttftMs.p99Ms} ms</p>`}
      ${serving.tpotMs == null ? '' : `<p class="result-detail">tpot median=${serving.tpotMs.medianMs} ms p99=${serving.tpotMs.p99Ms} ms</p>`}
      ${serving.itlMs == null ? '' : `<p class="result-detail">itl median=${serving.itlMs.medianMs} ms p99=${serving.itlMs.p99Ms} ms</p>`}
      <p class="result-detail">
        scheduler_ticks=${runtime.avgSchedulerTickCount == null ? 'n/a' : runtime.avgSchedulerTickCount}
        batch_participation=${runtime.avgBatchParticipationCount == null ? 'n/a' : runtime.avgBatchParticipationCount}
        decode_first_ticks=${runtime.avgDecodeFirstTickCount == null ? 'n/a' : runtime.avgDecodeFirstTickCount}
        chunked_prefill_ticks=${runtime.avgChunkedPrefillTickCount == null ? 'n/a' : runtime.avgChunkedPrefillTickCount}
        mixed_workload_ticks=${runtime.avgMixedWorkloadTickCount == null ? 'n/a' : runtime.avgMixedWorkloadTickCount}
      </p>
      <p class="result-preview">${escapeHtml(preview)}</p>
    </article>
  `;
}

function scenarioSection(result) {
  const definition = result.definition;

  return `
    <section class="result-card">
      <h3>${escapeHtml(definition.label)}</h3>
      <div class="metric-grid">
        ${metricCard('Scenario', definition.id.toUpperCase())}
        ${metricCard('Prompt Bucket', definition.promptBucket)}
        ${metricCard('Output Bucket', definition.outputBucket)}
        ${metricCard('Prompt Chars', String(definition.promptChars))}
        ${metricCard('Prompt Words', String(definition.promptWords))}
        ${metricCard('Max Tokens', String(definition.outputTokenLimit))}
        ${metricCard('Engine Init', formatMs(result.runtime.initEngineMs))}
      </div>
      <div class="result-stack">
        ${benchmarkSection('Cold Prompt', result.coldPrompt)}
        ${benchmarkSection('Hot Prompt: Fresh Context', result.hotFreshContext)}
        ${benchmarkSection('Hot Prompt: Reused Context', result.hotReuseContext)}
      </div>
    </section>
  `;
}

function mixedLoadSection(result) {
  if (!result) {
    return '';
  }

  if (result.unsupported) {
    return `
      <section class="result-card">
        <h3>${escapeHtml(result.definition.label)}</h3>
        <div class="metric-grid">
          ${metricCard('Background', result.definition.background.label)}
          ${metricCard('Foreground', result.definition.foreground.label)}
          ${metricCard('Concurrency', String(result.definition.concurrency))}
          ${metricCard('Status', 'Skipped', 'warn')}
        </div>
        <p class="result-detail">
          ${escapeHtml(
            result.reason ||
              'Mixed-load benchmark was skipped because the loaded engine instance does not support queued requests.'
          )}
        </p>
      </section>
    `;
  }

  return `
    <section class="result-card">
      <h3>${escapeHtml(result.definition.label)}</h3>
      <div class="metric-grid">
        ${metricCard('Background', result.definition.background.label)}
        ${metricCard('Foreground', result.definition.foreground.label)}
        ${metricCard('Concurrency', String(result.definition.concurrency))}
        ${metricCard('Engine Init', formatMs(result.runtime.initEngineMs))}
      </div>
      <div class="result-stack">
        ${benchmarkSection(result.foreground.label, result.foreground)}
        ${benchmarkSection(result.background.label, result.background)}
      </div>
    </section>
  `;
}

function memorySnapshotSection(memory) {
  if (!memory || memory.summary.snapshotCount === 0) {
    return '';
  }

  const summaryCards = [
    metricCard('Snapshots', String(memory.summary.snapshotCount)),
    metricCard('JS Heap Peak', formatBytes(memory.summary.maxUsedJsHeapBytes)),
    metricCard('JS Heap Total Peak', formatBytes(memory.summary.maxTotalJsHeapBytes)),
    metricCard('UA Memory Peak', formatBytes(memory.summary.maxUserAgentBytes)),
  ].join('');

  const snapshotLines = memory.snapshots
    .map((snapshot) => {
      const parts = [
        snapshot.label,
        `source=${snapshot.source}`,
        `used_js_heap=${formatBytes(snapshot.usedJsHeapBytes)}`,
        `total_js_heap=${formatBytes(snapshot.totalJsHeapBytes)}`,
        `js_heap_limit=${formatBytes(snapshot.jsHeapLimitBytes)}`,
        `ua_memory=${formatBytes(snapshot.userAgentBytes)}`,
      ];

      if (snapshot.error) {
        parts.push(`error=${snapshot.error}`);
      }

      return `<p class="result-detail">${escapeHtml(parts.join(' | '))}</p>`;
    })
    .join('');

  return `
    <article class="result-card">
      <h3>Memory Snapshots</h3>
      <div class="metric-grid">${summaryCards}</div>
      ${snapshotLines}
    </article>
  `;
}

function renderBenchmarkReport(report) {
  const backend = report.backend;
  const webGpuTone = report.environment.adapterAvailable ? 'ok' : 'warn';
  const webGpuLabel = report.environment.adapterAvailable
    ? `adapter ready: ${report.environment.adapterLabel}`
    : report.environment.hasNavigatorGpu
      ? 'navigator.gpu present, but no adapter acquired'
      : 'navigator.gpu unavailable';

  benchmarkResultsEl.innerHTML = `
    <div class="metric-grid">
      ${metricCard('Runtime Init', formatMs(report.runtime.initModuleMs))}
      ${metricCard('Model Load', formatMs(report.runtime.loadModelMs))}
      ${metricCard('Engine Init Mean', formatMs(report.runtime.initEngineSummary.initEngineMs.meanMs))}
      ${metricCard('Model Source', report.modelSource.label)}
      ${metricCard('Scenario Count', String(report.benchmark.scenarioCount))}
      ${metricCard('Benchmark Matrix', 'SISO / SILO / LISO / LILO')}
      ${metricCard('Prefill Chunk', String(report.runtime.initConfig?.prefillChunkSize ?? 0))}
      ${metricCard('Scheduler Policy', report.runtime.initConfig?.schedulerPolicy ?? 'balanced')}
      ${metricCard('Decode Reserve', String(report.runtime.initConfig?.decodeTokenReserve ?? 1))}
      ${metricCard('WebGPU', webGpuLabel, webGpuTone)}
      ${metricCard('Runtime Backend', describeRuntimeBackend(report.runtimeBackend), report.runtimeBackend?.webgpuRegistered ? 'ok' : 'warn')}
      ${metricCard('Execution Backend', backend.inferredExecutionBackend)}
      ${metricCard('Runtime Status', backend.runtimeBackendStatus)}
      ${metricCard('Adapter Vendor', backend.hostAdapter.adapterVendor || 'n/a')}
      ${metricCard('JS Heap Peak', formatBytes(report.memory.summary.maxUsedJsHeapBytes))}
      ${metricCard('UA Memory Peak', formatBytes(report.memory.summary.maxUserAgentBytes))}
    </div>
    <div class="result-stack">
      ${report.scenarios.map((scenario) => scenarioSection(scenario)).join('')}
      ${mixedLoadSection(report.mixedLoad)}
      ${memorySnapshotSection(report.memory)}
    </div>
  `;
}

function renderResponseMetrics(response, wallMs, perf) {
  const outputTokenCount = perf?.outputTokenCount ?? (response.text.length > 0 ? 1 : 0);
  const tpotMs =
    response.ttftMs != null && outputTokenCount > 1
      ? round((wallMs - response.ttftMs) / (outputTokenCount - 1))
      : null;
  responseMetaEl.innerHTML = [
    metricCard('Wall', formatMs(wallMs)),
    metricCard('TTFT', response.ttftMs == null ? 'n/a' : formatMs(response.ttftMs)),
    metricCard('TPOT', tpotMs == null ? 'n/a' : formatMs(tpotMs)),
    metricCard('Queue Delay', perf ? formatMs(perf.queueDelayMs) : 'n/a'),
    metricCard('Tail ITL', perf ? formatMs(perf.tailItlMs) : 'n/a'),
    metricCard('Output Tokens', perf ? String(perf.outputTokenCount) : 'n/a'),
    metricCard('Logical Input Tokens', perf ? String(perf.inputTokenCount) : 'n/a'),
    metricCard('Effective Prompt Tokens', perf ? String(perf.promptEvalTokens) : 'n/a'),
    metricCard('Prompt Eval', perf ? formatMs(perf.promptEvalMs) : 'n/a'),
    metricCard('Decode Eval', perf ? formatMs(perf.decodeEvalMs) : 'n/a'),
    metricCard('Sample', perf ? formatMs(perf.sampleMs) : 'n/a'),
  ].join('');
  responseEl.textContent = response.text;
}

function parsePositiveInt(input, fallback) {
  const value = Number.parseInt(input.value, 10);
  if (!Number.isInteger(value) || value <= 0) {
    input.value = String(fallback);
    return fallback;
  }
  return value;
}

function parseNonNegativeInt(input, fallback) {
  const value = Number.parseInt(input.value, 10);
  if (!Number.isInteger(value) || value < 0) {
    input.value = String(fallback);
    return fallback;
  }
  return value;
}

function parseTokenCount() {
  return parsePositiveInt(tokenCountEl, 64);
}

function parseWarmupRuns() {
  return parseNonNegativeInt(warmupRunsEl, 1);
}

function parseMeasuredRuns() {
  return parsePositiveInt(benchmarkRunsEl, 3);
}

function parseSchedulerPolicyMode() {
  const value = schedulerPolicyEl.value;
  if (value === 'latency-first' || value === 'balanced' || value === 'throughput-first') {
    return value;
  }
  schedulerPolicyEl.value = 'balanced';
  return 'balanced';
}

function readBenchmarkConfigFromUi() {
  const prompt = promptTextEl.value.trim();
  if (!prompt) {
    throw new Error('Prompt cannot be empty.');
  }

  return {
    prompt,
    tokenCount: parseTokenCount(),
    warmupRuns: parseWarmupRuns(),
    measuredRuns: parseMeasuredRuns(),
    initConfig: {
      prefillChunkSize: parseNonNegativeInt(prefillChunkSizeEl, 0),
      schedulerPolicy: parseSchedulerPolicyMode(),
      decodeTokenReserve: parseNonNegativeInt(decodeTokenReserveEl, 1),
    },
  };
}

function applyBenchmarkConfigToUi(config = {}) {
  if (typeof config.prompt === 'string') {
    promptTextEl.value = config.prompt;
  }
  if (typeof config.tokenCount === 'number') {
    tokenCountEl.value = String(config.tokenCount);
  }
  if (typeof config.warmupRuns === 'number') {
    warmupRunsEl.value = String(config.warmupRuns);
  }
  if (typeof config.measuredRuns === 'number') {
    benchmarkRunsEl.value = String(config.measuredRuns);
  }
  if (typeof config.initConfig?.prefillChunkSize === 'number') {
    prefillChunkSizeEl.value = String(config.initConfig.prefillChunkSize);
  }
  if (typeof config.initConfig?.schedulerPolicy === 'string') {
    schedulerPolicyEl.value = config.initConfig.schedulerPolicy;
  }
  if (typeof config.initConfig?.decodeTokenReserve === 'number') {
    decodeTokenReserveEl.value = String(config.initConfig.decodeTokenReserve);
  }
}

function setStatus(message) {
  statusEl.textContent = `Status: ${message}`;
}

function registerActionButton(button) {
  actionButtons.push(button);
}

function syncReportDownloadState() {
  downloadReportBtn.disabled = isBusy || !lastBenchmarkReport;
}

function setBusy(nextBusy) {
  isBusy = nextBusy;
  for (const button of actionButtons) {
    button.disabled = nextBusy;
  }
  syncReportDownloadState();
}

function getCurrentModelSelection() {
  const localFile = modelFileInput.files?.[0];
  if (localFile) {
    return {
      type: 'file',
      key: `file:${localFile.name}:${localFile.size}:${localFile.lastModified}`,
    };
  }

  const modelUrl = modelUrlInput.value.trim();
  if (modelUrl) {
    return {
      type: 'url',
      key: `url:${modelUrl}`,
    };
  }

  return null;
}

function resetEngine() {
  engine.close();
  engine = createEngine();
  runtimeReady = false;
  engineReady = false;
  runtimeBackendInfo = null;
  lastLoadedModelPath = null;
  lastLoadedModelSelectionKey = null;
  if (environmentInfo) {
    renderEnvironmentInfo(environmentInfo);
  }
}

async function measureAsync(fn) {
  const start = performance.now();
  const value = await fn();
  return {
    ms: round(performance.now() - start),
    value,
  };
}

async function readAdapterInfo(adapter) {
  if ('info' in adapter && adapter.info) {
    return adapter.info;
  }

  if (typeof adapter.requestAdapterInfo === 'function') {
    try {
      return await adapter.requestAdapterInfo();
    } catch {
      return null;
    }
  }

  return null;
}

async function collectEnvironmentInfo(force = false) {
  if (environmentInfo && !force) {
    return environmentInfo;
  }

  const info = {
    browserLabel: navigator.userAgent,
    language: navigator.language || 'unknown',
    hardwareConcurrency: navigator.hardwareConcurrency ?? null,
    deviceMemory: navigator.deviceMemory ?? null,
    crossOriginIsolated: window.crossOriginIsolated === true,
    hasNavigatorGpu: typeof navigator !== 'undefined' && 'gpu' in navigator,
    adapterAvailable: false,
    adapterLabel: 'none',
    adapterVendor: null,
    adapterArchitecture: null,
    adapterDescription: null,
    adapterError: null,
  };

  if (info.hasNavigatorGpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        info.adapterAvailable = true;
        const adapterInfo = await readAdapterInfo(adapter);
        info.adapterLabel =
          adapterInfo?.description ||
          adapterInfo?.vendor ||
          'available';
        info.adapterVendor = adapterInfo?.vendor ?? null;
        info.adapterArchitecture = adapterInfo?.architecture ?? null;
        info.adapterDescription = adapterInfo?.description ?? null;
      } else {
        info.adapterLabel = 'requestAdapter() returned null';
      }
    } catch (error) {
      info.adapterError = errorMessage(error);
      info.adapterLabel = 'requestAdapter() failed';
    }
  }

  environmentInfo = info;
  renderEnvironmentInfo(info);
  return info;
}

async function ensureBrowserWebGpuReady() {
  const info = await collectEnvironmentInfo();
  if (!info.hasNavigatorGpu) {
    throw new Error('WebGPU is unavailable in this browser session, so browser inference cannot start.');
  }
  if (!info.adapterAvailable) {
    const reason = info.adapterError || info.adapterLabel || 'requestAdapter() did not produce an adapter.';
    throw new Error(`WebGPU adapter unavailable: ${reason}`);
  }
  return info;
}

function renderEnvironmentInfo(info) {
  const cards = [
    metricCard('Browser', info.browserLabel),
    metricCard('Language', info.language),
    metricCard('Threads', info.hardwareConcurrency == null ? 'n/a' : String(info.hardwareConcurrency)),
    metricCard('Device Memory', info.deviceMemory == null ? 'n/a' : `${info.deviceMemory} GiB`),
    metricCard('COI', info.crossOriginIsolated ? 'enabled' : 'disabled', info.crossOriginIsolated ? 'ok' : 'warn'),
    metricCard(
      'WebGPU',
      info.adapterAvailable
        ? info.adapterLabel
        : info.hasNavigatorGpu
          ? 'API present, no adapter'
          : 'API unavailable',
      info.adapterAvailable ? 'ok' : 'warn'
    ),
    metricCard(
      'Runtime Backend',
      describeRuntimeBackend(runtimeBackendInfo),
      runtimeBackendInfo?.webgpuRegistered ? 'ok' : 'warn'
    ),
    metricCard(
      'Runtime Devices',
      describeRuntimeDevices(runtimeBackendInfo),
      runtimeBackendInfo?.devices?.length ? 'ok' : 'warn'
    ),
  ];

  if (info.adapterVendor) {
    cards.push(metricCard('GPU Vendor', info.adapterVendor));
  }

  if (info.adapterArchitecture) {
    cards.push(metricCard('Architecture', info.adapterArchitecture));
  }

  if (info.adapterError) {
    cards.push(metricCard('Adapter Error', info.adapterError, 'warn'));
  }

  if (runtimeBackendInfo && !runtimeBackendInfo.webgpuCompiled) {
    cards.push(metricCard('Build Warning', 'Package was built without ggml-webgpu.', 'warn'));
  } else if (runtimeBackendInfo && runtimeBackendInfo.webgpuCompiled && !runtimeBackendInfo.webgpuRegistered) {
    cards.push(metricCard('Runtime Warning', 'Browser exposes WebGPU, but the runtime did not register a WebGPU backend.', 'warn'));
  }

  environmentEl.innerHTML = cards.join('');
}

async function refreshRuntimeBackendInfo() {
  if (typeof engine.getBackendInfo !== 'function') {
    runtimeBackendInfo = {
      apiAvailable: false,
      webgpuCompiled: false,
      webgpuRegistered: false,
      webgpuDeviceCount: 0,
      gpuOffloadSupported: false,
      engineInitialized: false,
      availableBackends: [],
      devices: [],
    };
  } else {
    runtimeBackendInfo = await engine.getBackendInfo();
  }

  if (environmentInfo) {
    renderEnvironmentInfo(environmentInfo);
  }
  return runtimeBackendInfo;
}

async function loadModelIntoEngine(targetEngine, statusPrefix) {
  const localFile = modelFileInput.files?.[0];

  if (localFile) {
    const { ms, value } = await measureAsync(() =>
      targetEngine.loadModelFromFile(localFile, localFile.name || 'active-model.gguf', (pct) => {
        setStatus(`${statusPrefix} reading local model... ${pct}%`);
      })
    );

    return {
      loadModelMs: ms,
      modelPath: value,
      modelSelectionKey: `file:${localFile.name}:${localFile.size}:${localFile.lastModified}`,
      modelSource: {
        type: 'file',
        label: localFile.name,
        sizeBytes: localFile.size,
      },
    };
  }

  const modelUrl = modelUrlInput.value.trim();
  if (!modelUrl) {
    throw new Error('Choose a local GGUF file or provide a model URL.');
  }

  const { ms, value } = await measureAsync(() =>
    targetEngine.loadModelFromUrl(modelUrl, 'active-model.gguf', (pct) => {
      setStatus(`${statusPrefix} downloading model... ${pct}%`);
    })
  );

  return {
    loadModelMs: ms,
    modelPath: value,
    modelSelectionKey: `url:${modelUrl}`,
    modelSource: {
      type: 'url',
      label: modelUrl,
    },
  };
}

async function initRuntimeCurrentEngine() {
  const { ms } = await measureAsync(() => engine.initModule());
  await refreshRuntimeBackendInfo();
  runtimeReady = true;
  return ms;
}

async function loadAndInitCurrentEngine(statusPrefix, initConfig) {
  if (!runtimeReady) {
    setStatus(`${statusPrefix} initializing runtime...`);
    await initRuntimeCurrentEngine();
  }

  const loadResult = await loadModelIntoEngine(engine, statusPrefix);
  setStatus(`${statusPrefix} initializing engine...`);
  const { ms: initEngineMs } = await measureAsync(() => engine.initEngine(loadResult.modelPath, initConfig));
  engineReady = true;
  lastLoadedModelSource = loadResult.modelSource;
  lastLoadedModelPath = loadResult.modelPath;
  lastLoadedModelSelectionKey = loadResult.modelSelectionKey;
  await refreshRuntimeBackendInfo();

  return {
    loadModelMs: loadResult.loadModelMs,
    initEngineMs,
    modelSource: loadResult.modelSource,
  };
}

async function runPromptGroup(targetEngine, groupLabel, prompt, tokenCount, warmupRuns, measuredRuns, contextKeyFactory) {
  for (let i = 0; i < warmupRuns; i += 1) {
    setStatus(`${groupLabel}: warmup ${i + 1}/${warmupRuns}`);
    await targetEngine.streamPrompt(contextKeyFactory(i), prompt, tokenCount);
  }

  const runs = [];
  const benchmarkStart = performance.now();
  for (let i = 0; i < measuredRuns; i += 1) {
    setStatus(`${groupLabel}: run ${i + 1}/${measuredRuns}`);
    const start = performance.now();
    let ttftMs = null;
    const tokenEventTimes = [];
    const output = await targetEngine.streamPrompt(contextKeyFactory(i + warmupRuns), prompt, {
      nTokens: tokenCount,
      onToken: () => {
        const elapsedMs = round(performance.now() - start);
        tokenEventTimes.push(elapsedMs);
        if (ttftMs == null) {
          ttftMs = elapsedMs;
        }
      },
    });
    const wallMs = round(performance.now() - start);
    const perf = targetEngine.getLastPromptPerformance();
    const outputTokenCount = perf?.outputTokenCount ?? tokenEventTimes.length;
    const itlMsValues = [];
    for (let tokenIndex = 1; tokenIndex < tokenEventTimes.length; tokenIndex += 1) {
      itlMsValues.push(round(tokenEventTimes[tokenIndex] - tokenEventTimes[tokenIndex - 1]));
    }
    const tpotMs =
      ttftMs != null && outputTokenCount > 1
        ? round((wallMs - ttftMs) / (outputTokenCount - 1))
        : null;

    runs.push({
      label: `${groupLabel}-${i + 1}`,
      wallMs,
      ttftMs,
      tpotMs,
      itlMsValues,
      inputTokenCount: perf?.inputTokenCount ?? null,
      outputTokenCount,
      outputLength: output.length,
      outputPreview: output.slice(0, 160).replace(/\s+/g, ' ').trim(),
      perf,
    });
  }

  const benchmarkDurationMs = round(performance.now() - benchmarkStart);
  return {
    benchmarkDurationMs,
    runs,
    summary: summarizeRunGroup(runs, benchmarkDurationMs),
  };
}

function createGroupResult(id, label, warmupRuns, measuredRuns, group) {
  return {
    id,
    label,
    warmupRuns,
    measuredRuns,
    benchmarkDurationMs: group.benchmarkDurationMs,
    runs: group.runs,
    summary: group.summary,
  };
}

async function runScenarioBenchmark(targetEngine, scenario, modelPath, warmupRuns, measuredRuns, initConfig) {
  setStatus(`${scenario.label}: initializing engine...`);
  const { ms: initEngineMs } = await measureAsync(() => targetEngine.initEngine(modelPath, initConfig));
  engineReady = true;

  const coldPrompt = await runPromptGroup(
    targetEngine,
    `${scenario.label}: cold prompt`,
    scenario.prompt,
    scenario.outputTokenLimit,
    0,
    1,
    () => `${scenario.id}-cold`
  );

  const hotFreshContext = await runPromptGroup(
    targetEngine,
    `${scenario.label}: hot fresh context`,
    scenario.prompt,
    scenario.outputTokenLimit,
    warmupRuns,
    measuredRuns,
    (index) => `${scenario.id}-fresh-${index}`
  );

  const hotReuseContext = await runPromptGroup(
    targetEngine,
    `${scenario.label}: hot reused context`,
    scenario.prompt,
    scenario.outputTokenLimit,
    warmupRuns,
    measuredRuns,
    () => `${scenario.id}-reuse`
  );

  return {
    definition: scenario,
    runtime: {
      initEngineMs,
    },
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

async function runQueuedMixedLoadPair(targetEngine, definition, runIndex) {
  const backgroundContextKey = `${definition.background.id}-mixed-${runIndex}`;
  const foregroundContextKey = `${definition.foreground.id}-mixed-${runIndex}`;

  const backgroundStart = performance.now();
  let backgroundTtftMs = null;
  const backgroundTokenEventTimes = [];
  const backgroundRequestId = await targetEngine.queuePrompt(
    backgroundContextKey,
    definition.background.prompt,
    {
      nTokens: definition.background.outputTokenLimit,
      promptFormat: definition.background.promptFormat,
      onToken: () => {
        const elapsedMs = round(performance.now() - backgroundStart);
        backgroundTokenEventTimes.push(elapsedMs);
        if (backgroundTtftMs == null) {
          backgroundTtftMs = elapsedMs;
        }
      },
    }
  );

  const foregroundStart = performance.now();
  let foregroundTtftMs = null;
  const foregroundTokenEventTimes = [];
  const foregroundRequestId = await targetEngine.queuePrompt(
    foregroundContextKey,
    definition.foreground.prompt,
    {
      nTokens: definition.foreground.outputTokenLimit,
      promptFormat: definition.foreground.promptFormat,
      onToken: () => {
        const elapsedMs = round(performance.now() - foregroundStart);
        foregroundTokenEventTimes.push(elapsedMs);
        if (foregroundTtftMs == null) {
          foregroundTtftMs = elapsedMs;
        }
      },
    }
  );

  const foregroundResponse = await targetEngine.runQueuedRequest(foregroundRequestId);
  const foregroundWallMs = round(performance.now() - foregroundStart);
  const backgroundResponse = await targetEngine.runQueuedRequest(backgroundRequestId);
  const backgroundWallMs = round(performance.now() - backgroundStart);

  const toRun = (label, contextKey, wallMs, ttftMs, tokenEventTimes, response) => {
    const perf = response.perf ?? null;
    const outputTokenCount = perf?.outputTokenCount ?? tokenEventTimes.length;
    const itlMsValues = [];
    for (let tokenIndex = 1; tokenIndex < tokenEventTimes.length; tokenIndex += 1) {
      itlMsValues.push(round(tokenEventTimes[tokenIndex] - tokenEventTimes[tokenIndex - 1]));
    }
    const effectiveTtftMs = ttftMs ?? perf?.ttftMs ?? null;
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
      inputTokenCount: perf?.inputTokenCount ?? null,
      outputTokenCount,
      outputLength: response.outputText.length,
      outputPreview: response.outputText.slice(0, 160).replace(/\s+/g, ' ').trim(),
      perf,
    };
  };

  return {
    backgroundRun: toRun(
      `${definition.id}-background-${runIndex + 1}`,
      backgroundContextKey,
      backgroundWallMs,
      backgroundTtftMs,
      backgroundTokenEventTimes,
      backgroundResponse
    ),
    foregroundRun: toRun(
      `${definition.id}-foreground-${runIndex + 1}`,
      foregroundContextKey,
      foregroundWallMs,
      foregroundTtftMs,
      foregroundTokenEventTimes,
      foregroundResponse
    ),
  };
}

async function runMixedLoadBenchmark(targetEngine, definition, modelPath, warmupRuns, measuredRuns, initConfig) {
  setStatus(`${definition.label}: initializing engine...`);
  const { ms: initEngineMs } = await measureAsync(() =>
    targetEngine.initEngine(modelPath, buildPhase4BenchmarkInitConfig(initConfig))
  );
  engineReady = true;

  for (let i = 0; i < warmupRuns; i += 1) {
    setStatus(`${definition.label}: warmup ${i + 1}/${warmupRuns}`);
    await runQueuedMixedLoadPair(targetEngine, definition, i);
  }

  const foregroundRuns = [];
  const backgroundRuns = [];
  const benchmarkStart = performance.now();
  for (let i = 0; i < measuredRuns; i += 1) {
    setStatus(`${definition.label}: run ${i + 1}/${measuredRuns}`);
    const pair = await runQueuedMixedLoadPair(targetEngine, definition, i + warmupRuns);
    backgroundRuns.push(pair.backgroundRun);
    foregroundRuns.push(pair.foregroundRun);
  }

  const benchmarkDurationMs = round(performance.now() - benchmarkStart);
  return {
    definition,
    runtime: {
      initEngineMs,
    },
    foreground: createGroupResult(
      'hotFreshContext',
      `${definition.foreground.label} Under Mixed Load`,
      warmupRuns,
      measuredRuns,
      {
        benchmarkDurationMs,
        runs: foregroundRuns,
        summary: summarizeRunGroup(foregroundRuns, benchmarkDurationMs),
      }
    ),
    background: createGroupResult(
      'hotFreshContext',
      `${definition.background.label} Under Mixed Load`,
      warmupRuns,
      measuredRuns,
      {
        benchmarkDurationMs,
        runs: backgroundRuns,
        summary: summarizeRunGroup(backgroundRuns, benchmarkDurationMs),
      }
    ),
  };
}

function supportsQueuedRequestApi(targetEngine) {
  return (
    targetEngine != null &&
    typeof targetEngine.queuePrompt === 'function' &&
    typeof targetEngine.runQueuedRequest === 'function'
  );
}

function createUnsupportedMixedLoadResult(definition, reason) {
  return {
    definition,
    unsupported: true,
    reason,
    runtime: {
      initEngineMs: null,
    },
  };
}

async function runBrowserBenchmark(config = {}) {
  applyBenchmarkConfigToUi(config);
  const uiConfig = readBenchmarkConfigFromUi();
  const warmupRuns = typeof config.warmupRuns === 'number' ? config.warmupRuns : uiConfig.warmupRuns;
  const measuredRuns = typeof config.measuredRuns === 'number' ? config.measuredRuns : uiConfig.measuredRuns;
  const scenarios = buildBenchmarkScenarios();
  const mixedLoadDefinition = buildMixedLoadDefinition();
  const effectiveInitConfig = buildPhase4BenchmarkInitConfig({
    ...uiConfig.initConfig,
    ...(config.initConfig ?? {}),
  });
  const memorySnapshots = [];
  const selectedModel = getCurrentModelSelection();

  memorySnapshots.push(await captureBrowserMemorySnapshot('before-benchmark'));
  await collectEnvironmentInfo(true);
  await ensureBrowserWebGpuReady();

  let initModuleMs = 0;
  if (!runtimeReady) {
    setStatus('benchmark: initializing runtime...');
    initModuleMs = await initRuntimeCurrentEngine();
    memorySnapshots.push(await captureBrowserMemorySnapshot('after-init-module'));
  } else {
    memorySnapshots.push(await captureBrowserMemorySnapshot('after-init-module-reuse'));
  }

  let loadResult;
  if (
    selectedModel &&
    lastLoadedModelPath &&
    lastLoadedModelSelectionKey === selectedModel.key
  ) {
    loadResult = {
      loadModelMs: 0,
      modelPath: lastLoadedModelPath,
      modelSelectionKey: lastLoadedModelSelectionKey,
      modelSource: lastLoadedModelSource,
      reusedExistingModel: true,
    };
    memorySnapshots.push(await captureBrowserMemorySnapshot('after-model-reuse'));
  } else {
    loadResult = await loadModelIntoEngine(engine, 'benchmark');
    lastLoadedModelSource = loadResult.modelSource;
    lastLoadedModelPath = loadResult.modelPath;
    lastLoadedModelSelectionKey = loadResult.modelSelectionKey;
    memorySnapshots.push(await captureBrowserMemorySnapshot('after-model-load'));
  }

  setStatus('benchmark: initializing engine...');
  const scenarioResults = [];
  for (const scenario of scenarios) {
    const result = await runScenarioBenchmark(
      engine,
      scenario,
      loadResult.modelPath,
      warmupRuns,
      measuredRuns,
      effectiveInitConfig
    );
    scenarioResults.push(result);
    await refreshRuntimeBackendInfo();
    memorySnapshots.push(await captureBrowserMemorySnapshot(`after-${scenario.id}`));
  }

  let mixedLoad;
  if (supportsQueuedRequestApi(engine)) {
    mixedLoad = await runMixedLoadBenchmark(
      engine,
      mixedLoadDefinition,
      loadResult.modelPath,
      warmupRuns,
      measuredRuns,
      effectiveInitConfig
    );
    await refreshRuntimeBackendInfo();
    memorySnapshots.push(await captureBrowserMemorySnapshot('after-mixed-load'));
  } else {
    mixedLoad = createUnsupportedMixedLoadResult(
      mixedLoadDefinition,
      'The loaded benchmark page is using an engine bundle without queuePrompt()/runQueuedRequest(). Hard refresh the page and restart the benchmark dev server to enable the Phase 4 mixed-load fairness run.'
    );
  }

  const report = {
    schemaVersion: 'cogent.benchmark.browser.v5',
    generatedAt: new Date().toISOString(),
    benchmark: {
      preset: 'default',
      warmupRuns,
      measuredRuns,
      scenarioCount: scenarioResults.length,
    },
    environment: environmentInfo,
    runtimeBackend: runtimeBackendInfo,
    backend: buildBenchmarkBackendProfile(environmentInfo, runtimeBackendInfo),
    modelSource: {
      ...loadResult.modelSource,
      sizeMiB:
        typeof loadResult.modelSource.sizeBytes === 'number'
          ? round(loadResult.modelSource.sizeBytes / (1024 * 1024))
          : null,
      reusedExistingModel: loadResult.reusedExistingModel === true,
    },
    runtime: {
      initModuleMs,
      loadModelMs: loadResult.loadModelMs,
      initConfig: effectiveInitConfig,
      initEngineSummary: {
        initEngineMs: summarize([
          ...scenarioResults.map((scenario) => scenario.runtime.initEngineMs),
          ...(typeof mixedLoad?.runtime?.initEngineMs === 'number'
            ? [mixedLoad.runtime.initEngineMs]
            : []),
        ]),
      },
    },
    memory: {
      snapshots: memorySnapshots,
      summary: summarizeMemorySnapshots(memorySnapshots),
    },
    scenarios: scenarioResults,
    mixedLoad,
  };

  lastBenchmarkReport = report;
  syncReportDownloadState();
  renderBenchmarkReport(report);

  const sampleRun = report.scenarios
    .flatMap((scenario) => [
      ...scenario.hotReuseContext.runs,
      ...scenario.hotFreshContext.runs,
      ...scenario.coldPrompt.runs,
    ])[0] ?? null;

  if (sampleRun) {
    responseMetaEl.innerHTML = [
      metricCard('Benchmark Sample', sampleRun.label),
      metricCard('Wall', `${sampleRun.wallMs} ms`),
      metricCard('Output Chars', String(sampleRun.outputLength)),
    ].join('');
    responseEl.textContent = sampleRun.outputPreview || '(empty response)';
  }

  setStatus('browser benchmark complete');
  return report;
}

async function runSinglePrompt(config = {}) {
  applyBenchmarkConfigToUi(config);
  const { prompt, tokenCount } = readBenchmarkConfigFromUi();

  if (!engineReady) {
    throw new Error('Engine is not initialized yet.');
  }

  const start = performance.now();
  let ttftMs = null;
  responseEl.textContent = '';
  const text = await engine.streamPrompt(config.contextKey ?? 'browser-single', prompt, {
    nTokens: tokenCount,
    onToken: (token) => {
      if (ttftMs == null) {
        ttftMs = round(performance.now() - start);
      }
      responseEl.textContent += token;
    },
  });
  const wallMs = performance.now() - start;
  const perf = engine.getLastPromptPerformance();

  renderResponseMetrics({ text, ttftMs }, wallMs, perf);
  setStatus(`single inference complete in ${formatMs(wallMs)}`);

  return {
    text,
    wallMs: round(wallMs),
    ttftMs,
    perf,
  };
}

const statusEl = document.querySelector('#status');
const environmentEl = document.querySelector('#environment');
const responseMetaEl = document.querySelector('#responseMeta');
const responseEl = document.querySelector('#response');
const benchmarkResultsEl = document.querySelector('#benchmarkResults');
const modelUrlInput = document.querySelector('#modelUrl');
const modelFileInput = document.querySelector('#modelFile');
const promptTextEl = document.querySelector('#promptText');
const tokenCountEl = document.querySelector('#tokenCount');
const warmupRunsEl = document.querySelector('#warmupRuns');
const benchmarkRunsEl = document.querySelector('#benchmarkRuns');
const prefillChunkSizeEl = document.querySelector('#prefillChunkSize');
const schedulerPolicyEl = document.querySelector('#schedulerPolicy');
const decodeTokenReserveEl = document.querySelector('#decodeTokenReserve');
const initRuntimeBtn = document.querySelector('#initRuntimeBtn');
const loadModelBtn = document.querySelector('#loadModelBtn');
const runPromptBtn = document.querySelector('#runPromptBtn');
const runBenchmarkBtn = document.querySelector('#runBenchmarkBtn');
const downloadReportBtn = document.querySelector('#downloadReportBtn');

const actionButtons = [];
let engine = createEngine();
let runtimeReady = false;
let engineReady = false;
let lastBenchmarkReport = null;
let lastLoadedModelSource = null;
let lastLoadedModelPath = null;
let lastLoadedModelSelectionKey = null;
let environmentInfo = null;
let runtimeBackendInfo = null;
let isBusy = false;

[
  initRuntimeBtn,
  loadModelBtn,
  runPromptBtn,
  runBenchmarkBtn,
].forEach(registerActionButton);

initRuntimeBtn.addEventListener('click', async () => {
  setBusy(true);
  setStatus('initializing runtime...');

  try {
    const initModuleMs = await initRuntimeCurrentEngine();
    await collectEnvironmentInfo(true);
    setStatus(`runtime ready in ${formatMs(initModuleMs)}`);
  } catch (error) {
    setStatus(`runtime init failed: ${errorMessage(error)}`);
  } finally {
    setBusy(false);
  }
});

loadModelBtn.addEventListener('click', async () => {
  setBusy(true);
  setStatus('loading model...');

  try {
    await ensureBrowserWebGpuReady();
    const result = await loadAndInitCurrentEngine('manual load', readBenchmarkConfigFromUi().initConfig);
    const sourceLabel = result.modelSource.type === 'file'
      ? `${result.modelSource.label} (${formatMiB(result.modelSource.sizeBytes)})`
      : result.modelSource.label;
    responseMetaEl.innerHTML = [
      metricCard('Model Load', formatMs(result.loadModelMs)),
      metricCard('Engine Init', formatMs(result.initEngineMs)),
      metricCard('Source', sourceLabel),
      metricCard('Runtime Backend', describeRuntimeBackend(runtimeBackendInfo), runtimeBackendInfo?.webgpuRegistered ? 'ok' : 'warn'),
    ].join('');
    setStatus('engine initialized');
  } catch (error) {
    setStatus(`model init failed: ${errorMessage(error)}`);
  } finally {
    setBusy(false);
  }
});

runPromptBtn.addEventListener('click', async () => {
  setBusy(true);
  setStatus('running single inference...');

  try {
    await runSinglePrompt();
  } catch (error) {
    setStatus(`single inference failed: ${errorMessage(error)}`);
  } finally {
    setBusy(false);
  }
});

runBenchmarkBtn.addEventListener('click', async () => {
  setBusy(true);
  setStatus('starting browser benchmark...');

  try {
    await runBrowserBenchmark();
  } catch (error) {
    setStatus(`browser benchmark failed: ${errorMessage(error)}`);
  } finally {
    setBusy(false);
  }
});

downloadReportBtn.addEventListener('click', () => {
  if (!lastBenchmarkReport) {
    return;
  }

  const blob = new Blob([`${JSON.stringify(lastBenchmarkReport, null, 2)}\n`], {
    type: 'application/json',
  });
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = objectUrl;
  link.download = `cogent-browser-benchmark-${Date.now()}.json`;
  link.click();
  URL.revokeObjectURL(objectUrl);
});

function disposeApp() {
  engine.close();
}

window.addEventListener('beforeunload', disposeApp, { once: true });
if (import.meta.hot) {
  import.meta.hot.dispose(disposeApp);
}

const benchApi = {
  version: 1,
  getEnvironment: () => environmentInfo,
  getRuntimeBackend: () => runtimeBackendInfo,
  getLastReport: () => lastBenchmarkReport,
  isRuntimeReady: () => runtimeReady,
  isEngineReady: () => engineReady,
  collectEnvironmentInfo: (force = false) => collectEnvironmentInfo(force),
  initRuntime: async () => {
    const initModuleMs = await initRuntimeCurrentEngine();
    await collectEnvironmentInfo(true);
    return {
      initModuleMs,
      runtimeBackend: runtimeBackendInfo,
    };
  },
  loadConfiguredModelAndInitEngine: async (config = {}) => loadAndInitCurrentEngine('automation load', config.initConfig),
  runSinglePrompt: async (config = {}) => runSinglePrompt(config),
  runBenchmark: async (config = {}) => runBrowserBenchmark(config),
};

Object.defineProperty(window, '__cogentBench', {
  value: Object.freeze(benchApi),
  configurable: true,
});

collectEnvironmentInfo().catch((error) => {
  renderEnvironmentInfo({
    browserLabel: navigator.userAgent,
    language: navigator.language || 'unknown',
    hardwareConcurrency: navigator.hardwareConcurrency ?? null,
    deviceMemory: navigator.deviceMemory ?? null,
    crossOriginIsolated: window.crossOriginIsolated === true,
    hasNavigatorGpu: false,
    adapterAvailable: false,
    adapterLabel: errorMessage(error),
    adapterVendor: null,
    adapterArchitecture: null,
    adapterDescription: null,
    adapterError: errorMessage(error),
  });
});
