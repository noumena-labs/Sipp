import * as THREE from 'three';
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

const app = document.querySelector('#app');
app.innerHTML = `
  <div class="panel">
    <div class="eyebrow">Phase 3</div>
    <h1>CogentEngine Browser Benchmark</h1>
    <p class="intro">
      Browser-hosted benchmark harness for the inference runtime. Use this to measure the
      actual browser path instead of the Bun or Node host runtimes.
    </p>

    <section class="section">
      <div class="section-header">
        <h2>Environment</h2>
      </div>
      <div id="environment" class="metric-grid metric-grid-compact">
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
      <div class="row">
        <label>Model URL</label>
        <input id="modelUrl" placeholder="https://.../model.gguf" />
      </div>
      <div class="row">
        <label>Local GGUF File</label>
        <input id="modelFile" type="file" accept=".gguf" />
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
      <div class="row">
        <label>Prompt Text</label>
        <textarea id="promptText">Describe what this glowing object is seeing.</textarea>
      </div>
      <div class="field-grid">
        <div class="row">
          <label>Max Tokens</label>
          <input id="tokenCount" type="number" min="1" max="512" value="64" />
        </div>
        <div class="row">
          <label>Warmup Runs</label>
          <input id="warmupRuns" type="number" min="0" max="10" value="1" />
        </div>
        <div class="row">
          <label>Measured Runs</label>
          <input id="benchmarkRuns" type="number" min="1" max="10" value="3" />
        </div>
      </div>
      <div class="button-row">
        <button id="runPromptBtn" type="button">Run Single Inference</button>
        <button id="runBenchmarkBtn" type="button">Run Full Browser Benchmark</button>
      </div>
    </section>

    <p id="status" class="status">Status: idle</p>

    <section class="section">
      <div class="section-header">
        <h2>Response</h2>
      </div>
      <div id="responseMeta" class="metric-grid metric-grid-compact">
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
          Run the browser benchmark to capture runtime init, model load, engine init, cold prompt,
          hot fresh-context, and hot reused-context timings.
        </p>
      </div>
    </section>
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

function escapeHtml(value) {
  return value
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

function detectWebGlSupport() {
  const canvas = document.createElement('canvas');

  try {
    const webgl2 = canvas.getContext('webgl2', { failIfMajorPerformanceCaveat: true });
    if (webgl2) {
      return { supported: true, contextName: 'webgl2' };
    }

    const webgl = canvas.getContext('webgl', { failIfMajorPerformanceCaveat: true })
      || canvas.getContext('experimental-webgl', { failIfMajorPerformanceCaveat: true });
    if (webgl) {
      return { supported: true, contextName: 'webgl' };
    }

    return { supported: false, contextName: null };
  } catch (error) {
    return {
      supported: false,
      contextName: null,
      error: errorMessage(error),
    };
  }
}

function createGraphicsRuntime(targetApp) {
  const webGlSupport = detectWebGlSupport();
  if (!webGlSupport.supported) {
    return {
      available: false,
      message: webGlSupport.error
        ? `disabled: ${webGlSupport.error}`
        : 'disabled: browser WebGL is unavailable',
      renderer: null,
      scene: null,
      camera: null,
      knot: null,
      shell: null,
    };
  }

  try {
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    targetApp.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#050814');
    scene.fog = new THREE.Fog('#050814', 6, 14);

    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 0.6, 4.2);

    const hemiLight = new THREE.HemisphereLight('#9dd7ff', '#091320', 1.2);
    scene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight('#52b8ff', 1.8);
    keyLight.position.set(3, 2, 2);
    scene.add(keyLight);

    const knot = new THREE.Mesh(
      new THREE.TorusKnotGeometry(0.72, 0.24, 220, 32),
      new THREE.MeshStandardMaterial({
        color: '#1e7cff',
        emissive: '#16306b',
        emissiveIntensity: 0.6,
        metalness: 0.25,
        roughness: 0.22,
      })
    );
    scene.add(knot);

    const shell = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.75, 3),
      new THREE.MeshBasicMaterial({
        color: '#1f55a2',
        transparent: true,
        opacity: 0.14,
        wireframe: true,
      })
    );
    scene.add(shell);

    return {
      available: true,
      message: 'active',
      renderer,
      scene,
      camera,
      knot,
      shell,
    };
  } catch (error) {
    return {
      available: false,
      message: `disabled: ${errorMessage(error)}`,
      renderer: null,
      scene: null,
      camera: null,
      knot: null,
      shell: null,
    };
  }
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

function summarizeThroughput(perfRuns) {
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
      promptTokensPerSecond: summarizePromptThroughput(perfRuns),
      decodeTokensPerSecond: summarizeThroughput(perfRuns),
    },
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
    metricCard('Prompt Eval tok/s', runtime.promptTokensPerSecond == null ? 'n/a' : String(runtime.promptTokensPerSecond)),
  ].join('');

  const preview = group.runs[0]?.outputPreview?.trim() || '(empty response)';

  return `
    <article class="result-card">
      <h3>${escapeHtml(title)}</h3>
      <div class="metric-grid metric-grid-compact">${metrics}</div>
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
      <p class="result-preview">${escapeHtml(preview)}</p>
    </article>
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
      <div class="metric-grid metric-grid-compact">${summaryCards}</div>
      ${snapshotLines}
    </article>
  `;
}

function renderBenchmarkReport(report) {
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
      ${metricCard('Engine Init', formatMs(report.runtime.initEngineMs))}
      ${metricCard('Model Source', report.modelSource.label)}
      ${metricCard('Prompt Length', `${report.config.prompt.length} chars`)}
      ${metricCard('WebGPU', webGpuLabel, webGpuTone)}
      ${metricCard('JS Heap Peak', formatBytes(report.memory.summary.maxUsedJsHeapBytes))}
      ${metricCard('UA Memory Peak', formatBytes(report.memory.summary.maxUserAgentBytes))}
    </div>
    <div class="result-stack">
      ${benchmarkSection('Cold Prompt', report.coldPrompt)}
      ${benchmarkSection('Hot Prompt: Fresh Context', report.hotFreshContext)}
      ${benchmarkSection('Hot Prompt: Reused Context', report.hotReuseContext)}
      ${memorySnapshotSection(report.memory)}
    </div>
  `;
}

const appButtons = [];

function registerActionButton(button) {
  appButtons.push(button);
}

function setStatus(message) {
  statusEl.textContent = `Status: ${message}`;
}

function applyResponseColor(text) {
  if (!graphics.knot) {
    return;
  }
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash * 31 + text.charCodeAt(i)) >>> 0;
  }
  const hue = hash % 360;
  graphics.knot.material.color.setHSL(hue / 360, 0.8, 0.55);
  graphics.knot.material.emissive.setHSL(hue / 360, 0.7, 0.22);
}

function setBusy(isBusy) {
  appButtons.forEach((button) => {
    button.disabled = isBusy;
  });
  downloadReportBtn.disabled = isBusy || !lastBenchmarkReport;
}

function syncReportDownloadState() {
  downloadReportBtn.disabled = !lastBenchmarkReport;
}

function parseTokenCount() {
  const parsed = Number.parseInt(tokenCountEl.value, 10);
  const tokenCount = Number.isFinite(parsed) ? Math.min(512, Math.max(1, parsed)) : 64;
  tokenCountEl.value = String(tokenCount);
  return tokenCount;
}

function parseWarmupRuns() {
  const parsed = Number.parseInt(warmupRunsEl.value, 10);
  const runs = Number.isFinite(parsed) ? Math.min(10, Math.max(0, parsed)) : 1;
  warmupRunsEl.value = String(runs);
  return runs;
}

function parseMeasuredRuns() {
  const parsed = Number.parseInt(benchmarkRunsEl.value, 10);
  const runs = Number.isFinite(parsed) ? Math.min(10, Math.max(1, parsed)) : 3;
  benchmarkRunsEl.value = String(runs);
  return runs;
}

function renderResponseMetrics(response, wallMs, perf) {
  const ttftLabel = response.ttftMs == null ? 'n/a' : formatMs(response.ttftMs);
  const cards = [
    metricCard('Wall', formatMs(wallMs)),
    metricCard('TTFT', ttftLabel),
    metricCard('Chars', String(response.text.length)),
    metricCard(
      'Prompt tok/s',
      perf && perf.promptEvalMs > 0 && perf.promptEvalTokens > 0
        ? String(round((perf.promptEvalTokens * 1000) / perf.promptEvalMs))
        : 'n/a'
    ),
    metricCard(
      'Decode tok/s',
      perf && perf.decodeEvalMs > 0 && perf.outputTokenCount > 0
        ? String(round((perf.outputTokenCount * 1000) / perf.decodeEvalMs))
        : 'n/a'
    ),
    metricCard('Prompt Eval', perf ? formatMs(perf.promptEvalMs) : 'n/a'),
    metricCard('Decode Eval', perf ? formatMs(perf.decodeEvalMs) : 'n/a'),
    metricCard('Sample', perf ? formatMs(perf.sampleMs) : 'n/a'),
  ];

  responseMetaEl.innerHTML = cards.join('');
  responseEl.textContent = response.text;
}

function resetEngine() {
  engine.close();
  engine = createEngine();
  runtimeReady = false;
  engineReady = false;
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

let environmentInfo = null;

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
      '3D Background',
      graphics.available ? 'WebGL active' : graphics.message,
      graphics.available ? 'ok' : 'warn'
    ),
    metricCard(
      'WebGPU',
      info.adapterAvailable
        ? info.adapterLabel
        : info.hasNavigatorGpu
          ? 'API present, no adapter'
          : 'API unavailable',
      info.adapterAvailable ? 'ok' : 'warn'
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

  environmentEl.innerHTML = cards.join('');
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
    modelSource: {
      type: 'url',
      label: modelUrl,
    },
  };
}

async function initRuntimeCurrentEngine() {
  const { ms } = await measureAsync(() => engine.initModule());
  runtimeReady = true;
  return ms;
}

async function loadAndInitCurrentEngine(statusPrefix) {
  if (!runtimeReady) {
    setStatus(`${statusPrefix} initializing runtime...`);
    await initRuntimeCurrentEngine();
  }

  const loadResult = await loadModelIntoEngine(engine, statusPrefix);
  setStatus(`${statusPrefix} initializing engine...`);
  const { ms: initEngineMs } = await measureAsync(() => engine.initEngine(loadResult.modelPath));
  engineReady = true;
  lastLoadedModelSource = loadResult.modelSource;

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

async function runBrowserBenchmark() {
  const prompt = promptTextEl.value.trim();
  if (!prompt) {
    throw new Error('Prompt cannot be empty.');
  }

  const tokenCount = parseTokenCount();
  const warmupRuns = parseWarmupRuns();
  const measuredRuns = parseMeasuredRuns();
  const memorySnapshots = [];

  resetEngine();
  memorySnapshots.push(await captureBrowserMemorySnapshot('after-reset'));
  await collectEnvironmentInfo();

  setStatus('benchmark: initializing runtime...');
  const initModuleMs = await initRuntimeCurrentEngine();
  memorySnapshots.push(await captureBrowserMemorySnapshot('after-init-module'));

  const loadResult = await loadModelIntoEngine(engine, 'benchmark');
  memorySnapshots.push(await captureBrowserMemorySnapshot('after-model-load'));
  setStatus('benchmark: initializing engine...');
  const { ms: initEngineMs } = await measureAsync(() => engine.initEngine(loadResult.modelPath));
  engineReady = true;
  lastLoadedModelSource = loadResult.modelSource;
  memorySnapshots.push(await captureBrowserMemorySnapshot('after-engine-init'));

  const coldPrompt = await runPromptGroup(
    engine,
    'cold prompt',
    prompt,
    tokenCount,
    0,
    1,
    () => 'browser-bench-cold'
  );
  memorySnapshots.push(await captureBrowserMemorySnapshot('after-cold-prompt'));

  const hotFreshContext = await runPromptGroup(
    engine,
    'hot fresh context',
    prompt,
    tokenCount,
    warmupRuns,
    measuredRuns,
    (index) => `browser-bench-fresh-${index}`
  );
  memorySnapshots.push(await captureBrowserMemorySnapshot('after-hot-fresh-context'));

  const hotReuseContext = await runPromptGroup(
    engine,
    'hot reused context',
    prompt,
    tokenCount,
    warmupRuns,
    measuredRuns,
    () => 'browser-bench-reuse'
  );
  memorySnapshots.push(await captureBrowserMemorySnapshot('after-hot-reused-context'));

  const sampleOutput =
    hotReuseContext.runs[0]?.outputPreview ||
    hotFreshContext.runs[0]?.outputPreview ||
    coldPrompt.runs[0]?.outputPreview ||
    '';

  if (sampleOutput) {
    applyResponseColor(sampleOutput);
  }

  return {
    schemaVersion: 'cogent.benchmark.browser.v2',
    generatedAt: new Date().toISOString(),
    environment: environmentInfo,
    modelSource: {
      ...loadResult.modelSource,
      sizeMiB:
        typeof loadResult.modelSource.sizeBytes === 'number'
          ? round(loadResult.modelSource.sizeBytes / (1024 * 1024))
          : null,
    },
    config: {
      prompt,
      tokenCount,
      warmupRuns,
      measuredRuns,
    },
    runtime: {
      initModuleMs,
      loadModelMs: loadResult.loadModelMs,
      initEngineMs,
    },
    memory: {
      snapshots: memorySnapshots,
      summary: summarizeMemorySnapshots(memorySnapshots),
    },
    coldPrompt,
    hotFreshContext,
    hotReuseContext,
  };
}

let engine = createEngine();
let runtimeReady = false;
let engineReady = false;
let lastBenchmarkReport = null;
let lastLoadedModelSource = null;
let sceneEnergyTarget = 0.45;
let sceneEnergy = sceneEnergyTarget;
let animationFrameId = 0;
let isDisposed = false;

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
const initRuntimeBtn = document.querySelector('#initRuntimeBtn');
const loadModelBtn = document.querySelector('#loadModelBtn');
const runPromptBtn = document.querySelector('#runPromptBtn');
const runBenchmarkBtn = document.querySelector('#runBenchmarkBtn');
const downloadReportBtn = document.querySelector('#downloadReportBtn');

[
  initRuntimeBtn,
  loadModelBtn,
  runPromptBtn,
  runBenchmarkBtn,
].forEach(registerActionButton);

const graphics = createGraphicsRuntime(app);

if (!graphics.available) {
  statusEl.textContent = 'Status: WebGL unavailable; running benchmark UI without the 3D background.';
}

initRuntimeBtn.addEventListener('click', async () => {
  setBusy(true);
  setStatus('initializing runtime...');

  try {
    const initModuleMs = await initRuntimeCurrentEngine();
    await collectEnvironmentInfo();
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
    const result = await loadAndInitCurrentEngine('manual load');
    const sourceLabel = result.modelSource.type === 'file'
      ? `${result.modelSource.label} (${formatMiB(result.modelSource.sizeBytes)})`
      : result.modelSource.label;
    responseMetaEl.innerHTML = [
      metricCard('Model Load', formatMs(result.loadModelMs)),
      metricCard('Engine Init', formatMs(result.initEngineMs)),
      metricCard('Source', sourceLabel),
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
    if (!engineReady) {
      throw new Error('Engine is not initialized yet.');
    }

    const prompt = promptTextEl.value.trim();
    if (!prompt) {
      throw new Error('Prompt cannot be empty.');
    }

    const tokenCount = parseTokenCount();
    const start = performance.now();
    let ttftMs = null;
    responseEl.textContent = '';
    const text = await engine.streamPrompt('browser-single', prompt, {
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
    sceneEnergyTarget = Math.min(2.2, Math.max(0.55, text.length / 140));
    applyResponseColor(text);
    const graphicsSuffix = graphics.available ? '' : ' (3D background disabled)';
    setStatus(`single inference complete in ${formatMs(wallMs)}${graphicsSuffix}`);
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
    await ensureBrowserWebGpuReady();
    const report = await runBrowserBenchmark();
    lastBenchmarkReport = report;
    syncReportDownloadState();
    renderBenchmarkReport(report);

    const sampleRun =
      report.hotReuseContext.runs[0] ||
      report.hotFreshContext.runs[0] ||
      report.coldPrompt.runs[0] ||
      null;

    if (sampleRun) {
      responseMetaEl.innerHTML = [
        metricCard('Benchmark Sample', sampleRun.label),
        metricCard('Wall', `${sampleRun.wallMs} ms`),
        metricCard('Output Chars', String(sampleRun.outputLength)),
      ].join('');
      responseEl.textContent = sampleRun.outputPreview || '(empty response)';
    }

    setStatus('browser benchmark complete');
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

function handleResize() {
  if (!graphics.camera || !graphics.renderer) {
    return;
  }
  graphics.camera.aspect = window.innerWidth / window.innerHeight;
  graphics.camera.updateProjectionMatrix();
  graphics.renderer.setSize(window.innerWidth, window.innerHeight);
}

window.addEventListener('resize', handleResize);

function disposeMaterial(material) {
  if (Array.isArray(material)) {
    material.forEach((entry) => entry.dispose());
    return;
  }
  material.dispose();
}

function disposeDemo() {
  if (isDisposed) {
    return;
  }
  isDisposed = true;

  window.removeEventListener('resize', handleResize);
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = 0;
  }

  if (graphics.knot) {
    graphics.knot.geometry.dispose();
    disposeMaterial(graphics.knot.material);
  }
  if (graphics.shell) {
    graphics.shell.geometry.dispose();
    disposeMaterial(graphics.shell.material);
  }
  graphics.renderer?.dispose();
  engine.close();
}

window.addEventListener('beforeunload', disposeDemo, { once: true });
if (import.meta.hot) {
  import.meta.hot.dispose(disposeDemo);
}

function animate() {
  if (!graphics.renderer || !graphics.scene || !graphics.camera || !graphics.knot || !graphics.shell) {
    return;
  }
  sceneEnergy += (sceneEnergyTarget - sceneEnergy) * 0.03;
  graphics.knot.rotation.x += 0.0035 * sceneEnergy;
  graphics.knot.rotation.y += 0.0056 * sceneEnergy;
  graphics.shell.rotation.y -= 0.0024 * sceneEnergy;
  graphics.shell.rotation.z += 0.0008 * sceneEnergy;
  graphics.knot.material.emissiveIntensity = 0.45 + sceneEnergy * 0.35;

  graphics.renderer.render(graphics.scene, graphics.camera);
  animationFrameId = requestAnimationFrame(animate);
}

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

animate();
