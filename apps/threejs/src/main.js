import * as THREE from 'three';
import { CogentEngine, getBundledRuntimeUrls } from 'cogent-engine';
import './style.css';

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

function summarize(values) {
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

function summarizeRunGroup(runs) {
  const perfRuns = runs.map((run) => run.perf);

  return {
    wall: summarize(runs.map((run) => run.wallMs)),
    decodeTokensPerSecond: summarizeThroughput(perfRuns),
    avgPromptEvalMs: averagePerfMetric(perfRuns, (perf) => perf.promptEvalMs),
    avgDecodeEvalMs: averagePerfMetric(perfRuns, (perf) => perf.decodeEvalMs),
    avgSampleMs: averagePerfMetric(perfRuns, (perf) => perf.sampleMs),
    avgOutputTokenCount: averagePerfMetric(perfRuns, (perf) => perf.outputTokenCount),
  };
}

function benchmarkSection(title, group) {
  const summary = group.summary;
  const metrics = [
    metricCard('Wall Median', `${summary.wall.medianMs} ms`),
    metricCard('Wall Mean', `${summary.wall.meanMs} ms`),
    metricCard('Decode tok/s', summary.decodeTokensPerSecond == null ? 'n/a' : String(summary.decodeTokensPerSecond)),
    metricCard('Prompt Eval', summary.avgPromptEvalMs == null ? 'n/a' : `${summary.avgPromptEvalMs} ms`),
    metricCard('Decode Eval', summary.avgDecodeEvalMs == null ? 'n/a' : `${summary.avgDecodeEvalMs} ms`),
    metricCard('Sample', summary.avgSampleMs == null ? 'n/a' : `${summary.avgSampleMs} ms`),
  ].join('');

  const preview = group.runs[0]?.outputPreview?.trim() || '(empty response)';

  return `
    <article class="result-card">
      <h3>${escapeHtml(title)}</h3>
      <div class="metric-grid metric-grid-compact">${metrics}</div>
      <p class="result-detail">
        min=${summary.wall.minMs} ms
        median=${summary.wall.medianMs} ms
        mean=${summary.wall.meanMs} ms
        p95=${summary.wall.p95Ms} ms
        max=${summary.wall.maxMs} ms
      </p>
      <p class="result-preview">${escapeHtml(preview)}</p>
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
    </div>
    <div class="result-stack">
      ${benchmarkSection('Cold Prompt', report.coldPrompt)}
      ${benchmarkSection('Hot Prompt: Fresh Context', report.hotFreshContext)}
      ${benchmarkSection('Hot Prompt: Reused Context', report.hotReuseContext)}
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
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash * 31 + text.charCodeAt(i)) >>> 0;
  }
  const hue = hash % 360;
  knot.material.color.setHSL(hue / 360, 0.8, 0.55);
  knot.material.emissive.setHSL(hue / 360, 0.7, 0.22);
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
  const cards = [
    metricCard('Wall', formatMs(wallMs)),
    metricCard('Chars', String(response.length)),
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
  responseEl.textContent = response;
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
    await targetEngine.prompt(contextKeyFactory(i), prompt, tokenCount);
  }

  const runs = [];
  for (let i = 0; i < measuredRuns; i += 1) {
    setStatus(`${groupLabel}: run ${i + 1}/${measuredRuns}`);
    const start = performance.now();
    const output = await targetEngine.prompt(contextKeyFactory(i + warmupRuns), prompt, tokenCount);
    const wallMs = round(performance.now() - start);
    const perf = targetEngine.getLastPromptPerformance();

    runs.push({
      label: `${groupLabel}-${i + 1}`,
      wallMs,
      outputLength: output.length,
      outputPreview: output.slice(0, 160).replace(/\s+/g, ' ').trim(),
      perf,
    });
  }

  return {
    runs,
    summary: summarizeRunGroup(runs),
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

  resetEngine();
  await collectEnvironmentInfo();

  setStatus('benchmark: initializing runtime...');
  const initModuleMs = await initRuntimeCurrentEngine();

  const loadResult = await loadModelIntoEngine(engine, 'benchmark');
  setStatus('benchmark: initializing engine...');
  const { ms: initEngineMs } = await measureAsync(() => engine.initEngine(loadResult.modelPath));
  engineReady = true;
  lastLoadedModelSource = loadResult.modelSource;

  const coldPrompt = await runPromptGroup(
    engine,
    'cold prompt',
    prompt,
    tokenCount,
    0,
    1,
    () => 'browser-bench-cold'
  );

  const hotFreshContext = await runPromptGroup(
    engine,
    'hot fresh context',
    prompt,
    tokenCount,
    warmupRuns,
    measuredRuns,
    (index) => `browser-bench-fresh-${index}`
  );

  const hotReuseContext = await runPromptGroup(
    engine,
    'hot reused context',
    prompt,
    tokenCount,
    warmupRuns,
    measuredRuns,
    () => 'browser-bench-reuse'
  );

  const sampleOutput =
    hotReuseContext.runs[0]?.outputPreview ||
    hotFreshContext.runs[0]?.outputPreview ||
    coldPrompt.runs[0]?.outputPreview ||
    '';

  if (sampleOutput) {
    applyResponseColor(sampleOutput);
  }

  return {
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

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
app.appendChild(renderer.domElement);

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
    await collectEnvironmentInfo();
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
    const response = await engine.prompt('browser-single', prompt, tokenCount);
    const wallMs = performance.now() - start;
    const perf = engine.getLastPromptPerformance();

    renderResponseMetrics(response, wallMs, perf);
    sceneEnergyTarget = Math.min(2.2, Math.max(0.55, response.length / 140));
    applyResponseColor(response);
    setStatus(`single inference complete in ${formatMs(wallMs)}`);
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
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
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

  knot.geometry.dispose();
  disposeMaterial(knot.material);
  shell.geometry.dispose();
  disposeMaterial(shell.material);
  renderer.dispose();
  engine.close();
}

window.addEventListener('beforeunload', disposeDemo, { once: true });
if (import.meta.hot) {
  import.meta.hot.dispose(disposeDemo);
}

function animate() {
  sceneEnergy += (sceneEnergyTarget - sceneEnergy) * 0.03;
  knot.rotation.x += 0.0035 * sceneEnergy;
  knot.rotation.y += 0.0056 * sceneEnergy;
  shell.rotation.y -= 0.0024 * sceneEnergy;
  shell.rotation.z += 0.0008 * sceneEnergy;
  knot.material.emissiveIntensity = 0.45 + sceneEnergy * 0.35;

  renderer.render(scene, camera);
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
