import type { CogentEngine, ObservabilitySnapshot } from 'cogentlm';
import type { ScenarioDefinition } from './types';
import { countWords } from './utils';

export function classifyPromptBucket(prompt: string): string {
  const wordCount = countWords(prompt);
  if (wordCount <= 16) return 'short';
  if (wordCount <= 64) return 'medium';
  return 'long';
}

export function classifyOutputBucket(tokenCount: number): string {
  if (tokenCount <= 32) return 'short';
  if (tokenCount <= 96) return 'medium';
  return 'long';
}

export function buildBenchmarkScenarios(shortPrompt: string, shortOutput: number): ScenarioDefinition[] {
  const DEFAULT_SHORT_OUTPUT_TOKENS = shortOutput;
  const DEFAULT_LONG_OUTPUT_TOKENS = 128;
  const LONG_PROMPT = 'You are evaluating a browser-hosted inference runtime built with TypeScript, WebAssembly, and llama.cpp. Describe how you would benchmark cold start, module initialization, model load, engine initialization, prompt evaluation throughput, decode throughput, reused-context performance, and TTFT. Keep the answer concise but explain why prompt length and output length should be swept separately.';

  const defs = [
    {
      id: 'siso',
      label: 'Short Input / Short Output',
      prompt: shortPrompt,
      outputTokenLimit: DEFAULT_SHORT_OUTPUT_TOKENS,
    },
    {
      id: 'silo',
      label: 'Short Input / Long Output',
      prompt: shortPrompt,
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

  return defs.map((scenario) => ({
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

export function describeExecutionMode(targetEngine: any): string {
  return targetEngine == null ? 'unknown' : 'managed';
}

export function describeRuntimeObservability(targetEngine: CogentEngine | null): string {
  if (targetEngine == null) return 'unknown';
  const snapshot = targetEngine.observability.current();
  return `${snapshot.mode}:${snapshot.state}`;
}

export function describeBackendProfiling(info: ObservabilitySnapshot['profile'] | null | undefined): string {
  if (!info) return 'inactive';
  return info.profilingEnabled ? 'enabled' : 'disabled';
}

export function describeRuntimeBackend(info: ObservabilitySnapshot['profile'] | null | undefined): string {
  if (!info) return 'runtime not initialized';
  if (!info.webgpuCompiled) return 'CPU-only build';
  if (!info.webgpuRegistered) return 'WebGPU backend unavailable at runtime';
  return `WebGPU backend ready (${info.webgpuDeviceCount} device${info.webgpuDeviceCount === 1 ? '' : 's'})`;
}

export function describeRuntimeDevices(info: ObservabilitySnapshot['profile'] | null | undefined): string {
  if (!info || !Array.isArray(info.devices) || info.devices.length === 0) {
    return 'none';
  }
  return info.devices
    .map((device) => `${device.backendName || device.type}:${device.description || device.name || device.type}`)
    .join(' | ');
}

export function buildMixedLoadDefinition(): any {
  return {
    id: 'mixed-lilo-vs-siso',
    label: 'Mixed Load: LILO Background vs SISO Foreground',
    background: {
      id: 'mixed-background-lilo',
      label: 'Background Long Input / Long Output',
      prompt: 'You are evaluating a browser-hosted inference runtime built with TypeScript, WebAssembly, and llama.cpp. Describe how you would benchmark cold start, module initialization, model load, engine initialization, prompt evaluation throughput, decode throughput, reused-context performance, and TTFT. Keep the answer concise but explain why prompt length and output length should be swept separately.',
      promptBucket: 'long',
      promptChars: 341,
      promptWords: 43,
      outputTokenLimit: 128,
      outputBucket: 'long',
      promptFormat: 'auto-chat',
      contextBucket: 'single-request',
      concurrency: 1,
    },
    foreground: {
      id: 'mixed-foreground-siso',
      label: 'Foreground Short Input / Short Output',
      prompt: 'Write one sentence about measuring inference performance.',
      promptBucket: 'short',
      promptChars: 53,
      promptWords: 6,
      outputTokenLimit: 16,
      outputBucket: 'short',
      promptFormat: 'auto-chat',
      contextBucket: 'single-request',
      concurrency: 1,
    },
    concurrency: 2,
  };
}

export function buildPhase4BenchmarkInitConfig(initConfig: any = {}): any {
  return {
    ...initConfig,
    sampling: initConfig.sampling == null ? undefined : { ...initConfig.sampling },
    nSeqMax: Math.max(initConfig.nSeqMax ?? 1, 2),
    maxCachedSessions: Math.max(initConfig.maxCachedSessions ?? 8, 2),
  };
}

export function maxNullable(values: (number | null | undefined)[]): number | null {
  const filtered = values.filter((v): v is number => v != null && Number.isFinite(v));
  if (filtered.length === 0) return null;
  return Math.max(...filtered);
}

export function summarizeMemorySnapshots(memorySnapshots: any[]): any {
  return {
    snapshotCount: memorySnapshots.length,
    maxUsedJsHeapBytes: maxNullable(memorySnapshots.map((s) => s.usedJsHeapBytes)),
    maxTotalJsHeapBytes: maxNullable(memorySnapshots.map((s) => s.totalJsHeapBytes)),
    maxUserAgentBytes: maxNullable(memorySnapshots.map((s) => s.userAgentBytes)),
    finalSnapshot: memorySnapshots.length > 0 ? memorySnapshots[memorySnapshots.length - 1] : null,
  };
}

export function buildBenchmarkBackendProfile(environment: any, runtimeBackend: any): any {
  const runtimeDevices = Array.isArray(runtimeBackend?.devices) ? runtimeBackend.devices : [];
  const acceleratorDevices = runtimeDevices.filter((d: any) => d.type !== 'cpu');
  const notes: string[] = [];

  if (!environment?.hasNavigatorGpu) notes.push('navigator.gpu is unavailable in this browser session.');
  else if (!environment.adapterAvailable) notes.push('navigator.gpu is present, but requestAdapter() did not produce a usable adapter.');

  if (!runtimeBackend?.webgpuCompiled) notes.push('The package build did not include ggml-webgpu.');
  else if (!runtimeBackend.webgpuRegistered) notes.push('ggml-webgpu was compiled, but the runtime did not register a usable WebGPU backend.');
  else if ((runtimeBackend.webgpuDeviceCount ?? 0) <= 0) notes.push('ggml-webgpu was registered, but it reported no runtime devices.');

  return {
    requestedExecutionMode: runtimeBackend ? (runtimeBackend.webgpuRegistered ? 'gpu-offload' : 'cpu-only') : 'unknown',
    requestedGpuLayers: null,
    inferredExecutionBackend: (environment?.adapterAvailable && runtimeBackend?.webgpuRegistered && runtimeBackend?.webgpuDeviceCount > 0 && runtimeBackend?.gpuOffloadSupported) ? 'webgpu' : (runtimeBackend ? 'cpu' : 'unknown'),
    runtimeBackendStatus: !runtimeBackend ? 'unknown' : (!runtimeBackend.webgpuCompiled ? 'not-compiled' : (!runtimeBackend.webgpuRegistered ? 'compiled-not-registered' : ((runtimeBackend.webgpuDeviceCount ?? 0) <= 0 ? 'registered-no-devices' : 'webgpu-ready'))),
    gpuOffloadSupported: runtimeBackend?.gpuOffloadSupported ?? null,
    availableBackends: runtimeBackend?.availableBackends?.map((b: any) => b.name) ?? [],
    backendRegistries: runtimeBackend?.availableBackends ?? [],
    runtimeDeviceCount: runtimeDevices.length,
    runtimeAcceleratorDeviceCount: acceleratorDevices.length,
    runtimeDeviceLabels: runtimeDevices.map((d: any) => `${d.backendName || d.type}:${d.description || d.name || d.type}`),
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
