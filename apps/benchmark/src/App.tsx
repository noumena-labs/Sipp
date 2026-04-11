import { useState, useEffect, useRef } from 'react';
import { CogentEngine, getBundledRuntimeUrls } from 'cogent-engine';
import { MetricCard } from './components/MetricCard';
import { runScenarioBenchmark, supportsQueuedRequestApi, runMixedLoadBenchmark, captureBrowserMemorySnapshot } from './lib/benchmark-runner';
import type { ConfigOptions, EnvironmentInfo, ScenarioResult, MixedLoadResult, MemorySnapshot, RequestObservability } from './lib/types';
import { formatMs, formatBytes, round } from './lib/utils';
import { buildBenchmarkScenarios, describeRuntimeBackend, buildMixedLoadDefinition, buildPhase4BenchmarkInitConfig, buildBenchmarkBackendProfile, summarizeMemorySnapshots } from './lib/helpers';

const WORKER_TRANSPORT_PRESETS = {
  default: { bufferedTokenLimit: 8, flushIntervalMs: 16 },
  'low-buffer': { bufferedTokenLimit: 2, flushIntervalMs: 4 },
  'no-buffer': { bufferedTokenLimit: 1, flushIntervalMs: 0 },
} as const;

export default function App() {
  const [engine, setEngine] = useState<CogentEngine | null>(null);
  const [status, setStatus] = useState<string>('idle');
  const [isBusy, setIsBusy] = useState(false);
  const [envInfo, setEnvInfo] = useState<EnvironmentInfo | null>(null);
  const [backendInfo, setBackendInfo] = useState<any>(null);
  const [scenarioResults, setScenarioResults] = useState<ScenarioResult[]>([]);
  const [mixedLoadResult, setMixedLoadResult] = useState<MixedLoadResult | null>(null);
  const [memorySnapshots, setMemorySnapshots] = useState<MemorySnapshot[]>([]);
  const [benchmarkReport, setBenchmarkReport] = useState<any>(null);

  const [lastRunResponse, setLastRunResponse] = useState<string>('');
  const [lastRunMetrics, setLastRunMetrics] = useState<any>(null);

  const [isModuleInitialized, setIsModuleInitialized] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isEngineInitialized, setIsEngineInitialized] = useState(false);
  const [activeModelPath, setActiveModelPath] = useState('');
  const [loadModelMs, setLoadModelMs] = useState(0);
  const [modelSourceInfo, setModelSourceInfo] = useState<any>(null);
  const [lastJsHeapHeapSnapshot, setLastJsHeapSnapshot] = useState<number | null>(null);
  const [includeDetailedMemory, setIncludeDetailedMemory] = useState(false);

  const [config, setConfig] = useState<ConfigOptions>({
    prompt: 'Describe how to benchmark browser-hosted inference.',
    tokenCount: 64,
    warmupRuns: 1,
    measuredRuns: 3,
    workerTransport: {
      preset: 'default',
      bufferedTokenLimit: WORKER_TRANSPORT_PRESETS.default.bufferedTokenLimit,
      flushIntervalMs: WORKER_TRANSPORT_PRESETS.default.flushIntervalMs,
    },
    initConfig: {
      prefillChunkSize: 0,
      schedulerPolicy: 'balanced',
      decodeTokenReserve: 1,
    }
  });

  const [modelType, setModelType] = useState<'url' | 'file'>('url');
  const [modelUrl, setModelUrl] = useState('https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Reset engine initialization if config or model changes
  useEffect(() => {
    setIsEngineInitialized(false);
  }, [
    config.initConfig.prefillChunkSize,
    config.initConfig.schedulerPolicy,
    config.initConfig.decodeTokenReserve,
    config.workerTransport.bufferedTokenLimit,
    config.workerTransport.flushIntervalMs,
    modelUrl,
    modelType
  ]);

  useEffect(() => {
    async function init() {
      const hasGpu = 'gpu' in navigator;
      const info: EnvironmentInfo = {
        browserLabel: navigator.userAgent,
        language: navigator.language || 'unknown',
        hardwareConcurrency: navigator.hardwareConcurrency ?? null,
        // @ts-ignore
        deviceMemory: navigator.deviceMemory ?? null,
        crossOriginIsolated: window.crossOriginIsolated === true,
        hasNavigatorGpu: hasGpu,
        adapterAvailable: false,
        adapterLabel: 'none',
        adapterVendor: null,
        adapterArchitecture: null,
        adapterDescription: null,
        adapterError: null,
      };

      if (hasGpu) {
        try {
          const adapter = await navigator.gpu.requestAdapter();
          if (adapter) {
            info.adapterAvailable = true;
            // @ts-ignore
            info.adapterLabel = adapter.info?.description || adapter.info?.vendor || 'available';
            // @ts-ignore
            info.adapterVendor = adapter.info?.vendor ?? null;
            // @ts-ignore
            info.adapterArchitecture = adapter.info?.architecture ?? null;
          }
        } catch (e: any) {
          info.adapterError = e.message;
        }
      }
      setEnvInfo(info);
    }
    init();
  }, []);

  useEffect(() => {
    const nextEngine = new CogentEngine({
      ...getBundledRuntimeUrls(),
      workerMaxBufferedTokens: config.workerTransport.bufferedTokenLimit,
      workerTokenFlushIntervalMs: config.workerTransport.flushIntervalMs,
    });
    setEngine(nextEngine);
    setIsModuleInitialized(false);
    setIsModelLoaded(false);
    setIsEngineInitialized(false);
    setActiveModelPath('');
    setLoadModelMs(0);
    setModelSourceInfo(null);
    return () => {
      nextEngine.close();
    };
  }, [
    config.workerTransport.bufferedTokenLimit,
    config.workerTransport.flushIntervalMs,
  ]);

  const applyWorkerTransportPreset = (
    preset: ConfigOptions['workerTransport']['preset']
  ) => {
    if (preset === 'custom') {
      setConfig((current) => ({
        ...current,
        workerTransport: {
          ...current.workerTransport,
          preset,
        },
      }));
      return;
    }

    const nextPreset = WORKER_TRANSPORT_PRESETS[preset];
    setConfig((current) => ({
      ...current,
      workerTransport: {
        preset,
        bufferedTokenLimit: nextPreset.bufferedTokenLimit,
        flushIntervalMs: nextPreset.flushIntervalMs,
      },
    }));
  };

  const handleInitRuntime = async () => {
    if (!engine) return;
    setIsBusy(true);
    try {
      setStatus('initializing runtime...');
      await engine.initModule();
      setIsModuleInitialized(true);
      const bInfo = await engine.getBackendObservability();
      setBackendInfo(bInfo);
      setStatus('runtime initialized');
    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
    } finally {
      setIsBusy(false);
    }
  };

  const handleLoadModel = async () => {
    if (!engine) return;
    setIsBusy(true);
    setIsModelLoaded(false);
    try {
      if (!isModuleInitialized) {
        setStatus('initializing module first...');
        await engine.initModule();
        setIsModuleInitialized(true);
      }

      setStatus('loading model...');
      const startLoad = performance.now();
      let finalModelPath = "";
      let mSource: any = {};

      if (modelType === 'file' && fileInputRef.current?.files?.[0]) {
        const f = fileInputRef.current.files[0];
        finalModelPath = await engine.loadModelFromFile(f, f.name || 'active-model.gguf', (pct) => setStatus(`reading model... ${pct}%`));
        mSource = { type: 'file', label: f.name, sizeBytes: f.size };
      } else {
        finalModelPath = await engine.loadModelFromUrl(modelUrl, 'active-model.gguf', (pct) => setStatus(`downloading model... ${pct}%`));
        mSource = { type: 'url', label: modelUrl, sizeBytes: null };
      }

      const ms = round(performance.now() - startLoad);
      setLoadModelMs(ms);
      setActiveModelPath(finalModelPath);
      setModelSourceInfo(mSource);
      setIsModelLoaded(true);
      
      setStatus('initializing engine...');
      await engine.initEngine(finalModelPath, buildPhase4BenchmarkInitConfig(config.initConfig));
      setIsEngineInitialized(true);

      setStatus('model loaded and engine initialized');
      
      const bInfo = await engine.getBackendObservability();
      setBackendInfo(bInfo);
      
      const mem = await captureBrowserMemorySnapshot('after-load', includeDetailedMemory);
      setLastJsHeapSnapshot(mem.usedJsHeapBytes);

    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
    } finally {
      setIsBusy(false);
    }
  };

  const handleRunSinglePrompt = async () => {
    if (!engine) return;
    setIsBusy(true);
    setLastRunResponse('');
    setLastRunMetrics(null);
    try {
      if (!isModuleInitialized) {
        setStatus('initializing module...');
        await engine.initModule();
        setIsModuleInitialized(true);
      }
      
      let finalModelPath = activeModelPath;
      if (!isModelLoaded) {
        setStatus('loading model first...');
        if (modelType === 'file' && fileInputRef.current?.files?.[0]) {
          const f = fileInputRef.current.files[0];
          finalModelPath = await engine.loadModelFromFile(f, f.name || 'active-model.gguf', (pct) => setStatus(`reading model... ${pct}%`));
        } else {
          finalModelPath = await engine.loadModelFromUrl(modelUrl, 'active-model.gguf', (pct) => setStatus(`downloading model... ${pct}%`));
        }
        setIsModelLoaded(true);
        setActiveModelPath(finalModelPath);
      }

      setBackendInfo(await engine.getBackendObservability());
      
      if (!isEngineInitialized) {
        setStatus('initializing engine...');
        await engine.initEngine(finalModelPath, buildPhase4BenchmarkInitConfig(config.initConfig));
        setIsEngineInitialized(true);
      }

      setStatus('Running inference...');
      const start = performance.now();
      let ttftMs: number | null = null;
      let outputTokenCount = 0;
      const tEvents: number[] = [];
      const requestId = await engine.queuePrompt('single-run-context', config.prompt, {
        nTokens: config.tokenCount,
        onToken: (token) => {
          setLastRunResponse(prev => prev + token);
          const eMs = round(performance.now() - start);
          tEvents.push(eMs);
          if (ttftMs == null) ttftMs = eMs;
        }
      });
      const response = await engine.runQueuedRequest(requestId);
      const wallMs = round(performance.now() - start);
      const perf =
        (response.requestObservability ?? response.runtimeObservability ?? null) as RequestObservability | null;
      const runtimeAggregatePerf =
        typeof engine.getRuntimeAggregateObservability === 'function'
          ? engine.getRuntimeAggregateObservability()
          : (typeof engine.getRuntimeObservability === 'function' ? engine.getRuntimeObservability() : null);
      const transportObservability =
        typeof engine.getTransportObservability === 'function'
          ? engine.getTransportObservability()
          : null;
      outputTokenCount = perf?.outputTokenCount ?? tEvents.length;
      const tpotMs = ttftMs != null && outputTokenCount > 1 ? round((wallMs - ttftMs) / (outputTokenCount - 1)) : null;
      const nativeDecodeTokensPerSecond =
        perf != null && perf.decodeEvalMs > 0 && perf.outputTokenCount > 0
          ? round((perf.outputTokenCount * 1000) / perf.decodeEvalMs)
          : null;
      
      setLastRunResponse(response.outputText);
      setLastRunMetrics({
        wallMs,
        appObservedTtftMs: ttftMs,
        appObservedTpotMs: tpotMs,
        nativeTtftMs: perf?.ttftMs ?? null,
        nativeMeanItlMs: perf?.meanItlMs ?? null,
        nativeTailItlMs: perf?.tailItlMs ?? null,
        nativeDecodeTokensPerSecond,
        perf,
        runtimeAggregatePerf,
        transportObservability,
      });
      setStatus('idle');
      
      // Perform memory snapshot after setting isBusy to false to prevent UI freeze
      setTimeout(async () => {
        const mem = await captureBrowserMemorySnapshot('after-inference', includeDetailedMemory);
        setLastJsHeapSnapshot(mem.usedJsHeapBytes);
      }, 50);
    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
    } finally {
      setIsBusy(false);
    }
  };

  const handleRunBenchmark = async () => {
    if (!engine || !envInfo) return;
    setIsBusy(true);
    setScenarioResults([]);
    setMixedLoadResult(null);
    setBenchmarkReport(null);
    let memSnaps: MemorySnapshot[] = [];
    
    try {
      let initModuleMs = 0;
      if (!isModuleInitialized) {
        const startInit = performance.now();
        await engine.initModule();
        initModuleMs = round(performance.now() - startInit);
        setIsModuleInitialized(true);
      }
      
      memSnaps.push(await captureBrowserMemorySnapshot('after-init-module', includeDetailedMemory));
      setMemorySnapshots([...memSnaps]);

      setBackendInfo(await engine.getBackendObservability());
      
      let finalModelPath = activeModelPath;
      let lMs = loadModelMs;
      let mSource = modelSourceInfo;

      if (!isModelLoaded) {
        const startLoad = performance.now();
        if (modelType === 'file' && fileInputRef.current?.files?.[0]) {
          const f = fileInputRef.current.files[0];
          finalModelPath = await engine.loadModelFromFile(f, f.name || 'active-model.gguf', (pct) => setStatus(`reading model... ${pct}%`));
          mSource = { type: 'file', label: f.name, sizeBytes: f.size };
        } else {
          finalModelPath = await engine.loadModelFromUrl(modelUrl, 'active-model.gguf', (pct) => setStatus(`downloading model... ${pct}%`));
          mSource = { type: 'url', label: modelUrl, sizeBytes: null };
        }
        lMs = round(performance.now() - startLoad);
        setIsModelLoaded(true);
        setActiveModelPath(finalModelPath);
        setModelSourceInfo(mSource);
        setLoadModelMs(lMs);
      }

      memSnaps.push(await captureBrowserMemorySnapshot('after-model-load', includeDetailedMemory));
      setMemorySnapshots([...memSnaps]);

      const effectiveInitConfig = buildPhase4BenchmarkInitConfig(config.initConfig);
      const scenarios = buildBenchmarkScenarios(config.prompt, config.tokenCount);
      const results: ScenarioResult[] = [];
      let totalInitEngineMs = 0;

      for (const scenario of scenarios) {
        // Only initialize on first scenario if not already done, 
        // OR if you want to test fresh engine init for each scenario.
        // Usually, for benchmark, we want to see the performance of a WARM engine after the first scenario.
        const res = await runScenarioBenchmark(
          engine,
          scenario,
          finalModelPath,
          config.warmupRuns,
          config.measuredRuns,
          effectiveInitConfig,
          setStatus,
          isEngineInitialized // Pass whether to skip initial init
        );
        setIsEngineInitialized(true);
        totalInitEngineMs += res.runtime.initEngineMs;
        results.push(res);
        setScenarioResults([...results]);
        setBackendInfo(await engine.getBackendObservability());
        memSnaps.push(await captureBrowserMemorySnapshot(`after-${scenario.id}`, includeDetailedMemory));
        setMemorySnapshots([...memSnaps]);
      }

      const mixedLoadDef = buildMixedLoadDefinition();
      let mLoadResult: MixedLoadResult | null = null;
      if (supportsQueuedRequestApi(engine)) {
        mLoadResult = await runMixedLoadBenchmark(
          engine,
          mixedLoadDef,
          finalModelPath,
          config.warmupRuns,
          config.measuredRuns,
          effectiveInitConfig,
          setStatus,
          isEngineInitialized
        );
        totalInitEngineMs += (mLoadResult.runtime.initEngineMs || 0);
        setMixedLoadResult(mLoadResult);
        setBackendInfo(await engine.getBackendObservability());
        memSnaps.push(await captureBrowserMemorySnapshot('after-mixed-load', includeDetailedMemory));
        setMemorySnapshots([...memSnaps]);
      } else {
        mLoadResult = {
          definition: mixedLoadDef,
          unsupported: true,
          reason: 'Engine bundle does not support queuePrompt()/runQueuedRequest().',
          runtime: { initEngineMs: null },
        };
        setMixedLoadResult(mLoadResult);
      }

      const finalBackendInfo = await engine.getBackendObservability();
      const report = {
        schemaVersion: 'cogent.benchmark.browser.v7',
        generatedAt: new Date().toISOString(),
        benchmark: {
          preset: 'default',
          warmupRuns: config.warmupRuns,
          measuredRuns: config.measuredRuns,
          scenarioCount: results.length,
        },
        environment: envInfo,
        runtimeBackend: finalBackendInfo,
        backend: buildBenchmarkBackendProfile(envInfo, finalBackendInfo),
        modelSource: {
          ...mSource,
          sizeMiB: typeof mSource?.sizeBytes === 'number' ? round(mSource.sizeBytes / (1024 * 1024)) : null,
          reusedExistingModel: lMs < 100,
        },
        runtime: {
          initModuleMs,
          loadModelMs: lMs,
          initConfig: effectiveInitConfig,
          workerTransportConfig: {
            preset: config.workerTransport.preset,
            bufferedTokenLimit: config.workerTransport.bufferedTokenLimit,
            flushIntervalMs: config.workerTransport.flushIntervalMs,
          },
          transportObservability:
            typeof engine.getTransportObservability === 'function'
              ? engine.getTransportObservability()
              : null,
          initEngineSummary: {
            initEngineMs: { meanMs: round(totalInitEngineMs / (results.length + (mLoadResult && !mLoadResult.unsupported ? 1 : 0))) }
          },
        },
        memory: {
          snapshots: memSnaps,
          summary: summarizeMemorySnapshots(memSnaps),
        },
        scenarios: results,
        mixedLoad: mLoadResult,
      };

      setBenchmarkReport(report);
      setStatus('benchmark complete');
      setLastJsHeapSnapshot(memSnaps[memSnaps.length - 1].usedJsHeapBytes);
    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
    } finally {
      setIsBusy(false);
    }
  };

  const handleDownloadJson = () => {
    if (!benchmarkReport) return;
    const blob = new Blob([JSON.stringify(benchmarkReport, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `browser-benchmark-${new Date().toISOString().replace(/:/g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const renderGroup = (title: string, res: any) => {
    if (!res) return null;
    return (
      <div className="result-card" style={{ background: 'rgba(255,255,255,0.02)' }}>
        <h4>{title}</h4>
        <div className="metric-grid">
          <MetricCard label="Req/s" value={res.summary.serving.requestThroughputRps ?? 'n/a'} />
          <MetricCard label="Output tok/s" value={res.summary.serving.outputTokenThroughputTps ?? 'n/a'} />
          <MetricCard label="App TTFT" value={res.summary.serving.appObservedTtftMs ? formatMs(res.summary.serving.appObservedTtftMs.meanMs) : 'n/a'} />
          <MetricCard label="App ITL" value={res.summary.serving.appObservedItlMs ? formatMs(res.summary.serving.appObservedItlMs.meanMs) : 'n/a'} />
          <MetricCard label="Native TTFT" value={res.summary.runtime.nativeTtftMs ? formatMs(res.summary.runtime.nativeTtftMs.meanMs) : 'n/a'} />
          <MetricCard label="Native ITL" value={res.summary.runtime.nativeMeanItlMs ? formatMs(res.summary.runtime.nativeMeanItlMs.meanMs) : 'n/a'} />
          <MetricCard label="Native Decode" value={res.summary.runtime.nativeDecodeTokensPerSecond ? `${res.summary.runtime.nativeDecodeTokensPerSecond.meanMs} tok/s` : 'n/a'} />
          <MetricCard label="Mean E2EL" value={res.summary.serving.e2elMs ? formatMs(res.summary.serving.e2elMs.meanMs) : 'n/a'} />
        </div>
      </div>
    );
  };

  return (
    <div className="shell">
      <header className="hero">
        <div className="eyebrow">Browser Benchmark</div>
        <h1>CogentEngine React Benchmark</h1>
        <p>Browser-hosted benchmark harness for the WebGPU inference path, rewritten in React + TypeScript.</p>
      </header>
      <div className="layout">
        <div className="column">
          <section className="section">
            <div className="section-header"><h2>Environment</h2></div>
            <div className="metric-grid">
              <MetricCard label="Browser" value={envInfo?.browserLabel || 'collecting...'} />
              <MetricCard label="WebGPU" value={envInfo?.adapterAvailable ? 'ready' : 'unavailable'} tone={envInfo?.adapterAvailable ? 'ok' : 'warn'} />
              <MetricCard label="Physical Memory" value={envInfo?.deviceMemory ? `${envInfo.deviceMemory} GiB` : 'n/a'} />
              <MetricCard label="Logical Cores" value={envInfo?.hardwareConcurrency || 'n/a'} />
              <MetricCard label="JS Heap Snapshot" value={formatBytes(lastJsHeapHeapSnapshot)} />
              <MetricCard label="Backend Summary" value={describeRuntimeBackend(backendInfo)} tone={backendInfo?.webgpuRegistered ? 'ok' : 'warn'} />
            </div>
          </section>

          <section className="section">
            <div className="section-header"><h2>Model Source</h2></div>
            <div className="field-grid">
              <div className="row">
                <label>
                  <input type="radio" checked={modelType === 'url'} onChange={() => setModelType('url')} /> URL
                </label>
                <label>
                  <input type="radio" checked={modelType === 'file'} onChange={() => setModelType('file')} /> File
                </label>
              </div>
              <div className="row">
                {modelType === 'url' ? (
                  <input key="url-input" value={modelUrl} onChange={(e) => setModelUrl(e.target.value)} placeholder="https://.../model.gguf" />
                ) : (
                  <input key="file-input" type="file" accept=".gguf" ref={fileInputRef} />
                )}
              </div>
              <div className="button-row">
                <button type="button" onClick={handleInitRuntime} disabled={isBusy}>Init Runtime</button>
                <button type="button" onClick={handleLoadModel} disabled={isBusy}>Load Model</button>
              </div>
            </div>
          </section>

          <section className="section">
            <div className="section-header"><h2>Configuration</h2></div>
            <div className="field-grid">
              <div className="row">
                <label>Prompt Text</label>
                <textarea value={config.prompt} onChange={(e) => setConfig({ ...config, prompt: e.target.value })} />
              </div>
              <div className="row">
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input type="checkbox" checked={includeDetailedMemory} onChange={(e) => setIncludeDetailedMemory(e.target.checked)} />
                  Detailed Memory Tracking (can be slow)
                </label>
              </div>
            </div>
            <div className="field-grid field-grid-compact">
              <div className="row">
                <label>Max Tokens</label>
                <input type="number" value={Number.isNaN(config.tokenCount) ? '' : config.tokenCount} onChange={(e) => setConfig({ ...config, tokenCount: parseInt(e.target.value, 10) })} />
              </div>
              <div className="row">
                <label>Warmup Runs</label>
                <input type="number" value={Number.isNaN(config.warmupRuns) ? '' : config.warmupRuns} onChange={(e) => setConfig({ ...config, warmupRuns: parseInt(e.target.value, 10) })} />
              </div>
              <div className="row">
                <label>Measured Runs</label>
                <input type="number" value={Number.isNaN(config.measuredRuns) ? '' : config.measuredRuns} onChange={(e) => setConfig({ ...config, measuredRuns: parseInt(e.target.value, 10) })} />
              </div>
            </div>
            <div className="field-grid field-grid-compact">
              <div className="row">
                <label>Prefill Chunk</label>
                <input type="number" value={Number.isNaN(config.initConfig.prefillChunkSize) ? '' : config.initConfig.prefillChunkSize} onChange={(e) => setConfig({ ...config, initConfig: { ...config.initConfig, prefillChunkSize: parseInt(e.target.value, 10) } })} />
              </div>
              <div className="row">
                <label>Scheduler Policy</label>
                <select value={config.initConfig.schedulerPolicy} onChange={(e) => setConfig({ ...config, initConfig: { ...config.initConfig, schedulerPolicy: e.target.value } })}>
                  <option value="latency-first">latency-first</option>
                  <option value="balanced">balanced</option>
                  <option value="throughput-first">throughput-first</option>
                </select>
              </div>
              <div className="row">
                <label>Decode Reserve</label>
                <input type="number" value={Number.isNaN(config.initConfig.decodeTokenReserve) ? '' : config.initConfig.decodeTokenReserve} onChange={(e) => setConfig({ ...config, initConfig: { ...config.initConfig, decodeTokenReserve: parseInt(e.target.value, 10) } })} />
              </div>
            </div>
            <div className="field-grid field-grid-compact">
              <div className="row">
                <label>Worker Token Mode</label>
                <select
                  value={config.workerTransport.preset}
                  onChange={(e) =>
                    applyWorkerTransportPreset(
                      e.target.value as ConfigOptions['workerTransport']['preset']
                    )
                  }
                >
                  <option value="default">default</option>
                  <option value="low-buffer">low-buffer</option>
                  <option value="no-buffer">no-buffer</option>
                  <option value="custom">custom</option>
                </select>
              </div>
              <div className="row">
                <label>Buffered Tokens</label>
                <input
                  type="number"
                  value={
                    Number.isNaN(config.workerTransport.bufferedTokenLimit)
                      ? ''
                      : config.workerTransport.bufferedTokenLimit
                  }
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      workerTransport: {
                        ...config.workerTransport,
                        preset: 'custom',
                        bufferedTokenLimit: parseInt(e.target.value, 10),
                      },
                    })
                  }
                />
              </div>
              <div className="row">
                <label>Flush Interval</label>
                <input
                  type="number"
                  value={
                    Number.isNaN(config.workerTransport.flushIntervalMs)
                      ? ''
                      : config.workerTransport.flushIntervalMs
                  }
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      workerTransport: {
                        ...config.workerTransport,
                        preset: 'custom',
                        flushIntervalMs: parseInt(e.target.value, 10),
                      },
                    })
                  }
                />
              </div>
            </div>
            <div className="button-row">
              <button type="button" onClick={handleRunSinglePrompt} disabled={isBusy}>Run Single Inference</button>
              <button type="button" onClick={handleRunBenchmark} disabled={isBusy}>Run Browser Benchmark</button>
            </div>
          </section>

          <p className="status">Status: {status}</p>
        </div>

        <div className="column">
          <section className="section">
            <div className="section-header"><h2>Response</h2></div>
            <div className="metric-grid">
              {lastRunMetrics ? (
                <>
                  <MetricCard label="Speed" value={lastRunMetrics.perf ? `${round((lastRunMetrics.perf.outputTokenCount * 1000) / lastRunMetrics.wallMs)} tok/s` : 'n/a'} />
                  <MetricCard label="Total Latency" value={formatMs(lastRunMetrics.wallMs)} />
                  <MetricCard label="App TTFT" value={lastRunMetrics.appObservedTtftMs ? formatMs(lastRunMetrics.appObservedTtftMs) : 'n/a'} />
                  <MetricCard label="App TPOT" value={lastRunMetrics.appObservedTpotMs ? formatMs(lastRunMetrics.appObservedTpotMs) : 'n/a'} />
                  <MetricCard label="Native TTFT" value={lastRunMetrics.nativeTtftMs ? formatMs(lastRunMetrics.nativeTtftMs) : 'n/a'} />
                  <MetricCard label="Native ITL" value={lastRunMetrics.nativeMeanItlMs ? formatMs(lastRunMetrics.nativeMeanItlMs) : 'n/a'} />
                  <MetricCard label="Native Tail ITL" value={lastRunMetrics.nativeTailItlMs ? formatMs(lastRunMetrics.nativeTailItlMs) : 'n/a'} />
                  <MetricCard label="Native Decode" value={lastRunMetrics.nativeDecodeTokensPerSecond ? `${lastRunMetrics.nativeDecodeTokensPerSecond} tok/s` : 'n/a'} />
                  <MetricCard label="Output Tokens" value={lastRunMetrics.perf ? String(lastRunMetrics.perf.outputTokenCount) : 'n/a'} />
                  <MetricCard label="Logical Input" value={lastRunMetrics.perf ? String(lastRunMetrics.perf.inputTokenCount) : 'n/a'} />
                  <MetricCard label="Prompt Eval" value={lastRunMetrics.perf ? formatMs(lastRunMetrics.perf.promptEvalMs) : 'n/a'} />
                  <MetricCard label="Worker Mode" value={config.workerTransport.preset} />
                  <MetricCard
                    label="Flushes"
                    value={lastRunMetrics.transportObservability ? String(lastRunMetrics.transportObservability.flushCount ?? 'n/a') : 'n/a'}
                  />
                  <MetricCard
                    label="Max Buffered"
                    value={lastRunMetrics.transportObservability ? String(lastRunMetrics.transportObservability.maxObservedBufferedTokenCount ?? 'n/a') : 'n/a'}
                  />
                </>
              ) : (
                <MetricCard label="Last Run" value="No inference yet" />
              )}
            </div>
            <div className="response" style={{ marginTop: '16px', padding: '16px', background: 'var(--bg-layer)', border: '1px solid var(--border-subtle)', borderRadius: '6px', whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>
              {lastRunResponse}
            </div>
          </section>

          <section className="section">
            <div className="section-header">
              <h2>Results</h2>
              <button className="secondary-button" type="button" onClick={handleDownloadJson} disabled={!benchmarkReport}>
                Download JSON
              </button>
            </div>
            <div className="benchmark-results">
              {scenarioResults.map((res, i) => (
                <div key={i} className="result-card">
                  <h3>{res.definition.label}</h3>
                  <div className="metric-grid">
                    <MetricCard label="Scenario" value={res.definition.id.toUpperCase()} />
                    <MetricCard label="Engine Init" value={formatMs(res.runtime.initEngineMs)} />
                  </div>
                  <div className="result-stack" style={{ marginTop: '12px' }}>
                    {renderGroup("Cold Prompt", res.coldPrompt)}
                    {renderGroup("Hot Prompt: Fresh", res.hotFreshContext)}
                    {renderGroup("Hot Prompt: Reuse", res.hotReuseContext)}
                  </div>
                </div>
              ))}

              {mixedLoadResult && !mixedLoadResult.unsupported && (
                <div className="result-card">
                  <h3>{mixedLoadResult.definition.label}</h3>
                  <div className="metric-grid">
                    <MetricCard label="Background" value={mixedLoadResult.definition.background.label} />
                    <MetricCard label="Foreground" value={mixedLoadResult.definition.foreground.label} />
                    <MetricCard label="Concurrency" value={mixedLoadResult.definition.concurrency} />
                  </div>
                  <div className="result-stack" style={{ marginTop: '12px' }}>
                    {renderGroup("Foreground", mixedLoadResult.foreground)}
                    {renderGroup("Background", mixedLoadResult.background)}
                  </div>
                </div>
              )}
              {mixedLoadResult && mixedLoadResult.unsupported && (
                <div className="result-card">
                  <h3>{mixedLoadResult.definition.label}</h3>
                  <div className="metric-grid">
                    <MetricCard label="Status" value="Skipped" tone="warn" />
                  </div>
                  <p className="result-detail">{mixedLoadResult.reason}</p>
                </div>
              )}

              {memorySnapshots.length > 0 && benchmarkReport?.memory?.summary && (
                <div className="result-card">
                  <h3>Memory Snapshots</h3>
                  <div className="metric-grid">
                    <MetricCard label="Snapshots" value={benchmarkReport.memory.summary.snapshotCount} />
                    <MetricCard label="JS Heap Peak" value={formatBytes(benchmarkReport.memory.summary.maxUsedJsHeapBytes)} />
                    <MetricCard label="UA Memory Peak" value={formatBytes(benchmarkReport.memory.summary.maxUserAgentBytes)} />
                  </div>
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
