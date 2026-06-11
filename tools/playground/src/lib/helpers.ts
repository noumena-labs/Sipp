import type { ModelLoadOptions, ObservabilitySnapshot } from '@noumena-labs/cogentlm';
import type { BenchmarkOperation, MixedLoadDefinition, ScenarioDefinition } from './types';
import { countWords } from './utils';

type BenchmarkRuntimeOptions = NonNullable<ModelLoadOptions['runtime']>;

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

const DEFAULT_LONG_PROMPT = 'You are evaluating a browser-hosted inference runtime built with TypeScript, WebAssembly, and llama.cpp. Describe how you would benchmark cold start, module initialization, model load, runtime initialization, prompt evaluation throughput, decode throughput, reused-context performance, and TTFT. Keep the answer concise but explain why prompt length and output length should be swept separately.';

export interface BenchmarkPromptSet {
  longPrompt: string;
  mixedForegroundPrompt: string;
}

export const DEFAULT_BENCHMARK_PROMPTS: BenchmarkPromptSet = {
  longPrompt: DEFAULT_LONG_PROMPT,
  mixedForegroundPrompt: 'Write one sentence about measuring inference performance.',
};

export const ENCODER_DECODER_BENCHMARK_PROMPTS: BenchmarkPromptSet = {
  longPrompt: 'translate English to German: Browser inference should measure cold start, model loading time, prompt throughput, decode throughput, repeated prompt reuse, and time to first token.',
  mixedForegroundPrompt: 'translate English to German: Measure inference performance.',
};

export function buildBenchmarkScenarios(
  shortPrompt: string,
  shortOutput: number,
  promptSet: BenchmarkPromptSet = DEFAULT_BENCHMARK_PROMPTS
): ScenarioDefinition[] {
  const DEFAULT_SHORT_OUTPUT_TOKENS = shortOutput;
  const DEFAULT_LONG_OUTPUT_TOKENS = 128;

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
      prompt: promptSet.longPrompt,
      outputTokenLimit: DEFAULT_SHORT_OUTPUT_TOKENS,
    },
    {
      id: 'lilo',
      label: 'Long Input / Long Output',
      prompt: promptSet.longPrompt,
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

export function describeRuntimeBackend(info: ObservabilitySnapshot['profile'] | null | undefined): string {
  if (!info) return 'runtime not initialized';
  if (!info.webgpuCompiled) return 'CPU-only build';
  if (!info.webgpuRegistered) return 'WebGPU backend unavailable at runtime';
  const adapter = describeEngineAdapter(info);
  return `WebGPU backend ready (${info.webgpuDeviceCount} device${info.webgpuDeviceCount === 1 ? '' : 's'})${
    adapter == null ? '' : ` — ${adapter}`
  }`;
}

/**
 * GPU identity seen by the engine's execution scope, e.g. "nvidia ampere".
 * On hybrid-GPU machines this is the only signal that tells whether
 * inference landed on the integrated or the discrete GPU.
 */
export function describeEngineAdapter(
  info: ObservabilitySnapshot['profile'] | null | undefined
): string | null {
  const adapter = info?.webgpuAdapter;
  if (adapter == null) {
    return null;
  }
  const device = adapter.device?.trim() ?? '';
  const label = device !== ''
    ? device
    : [adapter.vendor, adapter.architecture]
        .map((part) => part?.trim() ?? '')
        .filter((part) => part !== '')
        .join(' ');
  return label === '' ? null : label;
}

export function buildMixedLoadDefinition(
  operation: Exclude<BenchmarkOperation, 'embed'>,
  promptSet: BenchmarkPromptSet = DEFAULT_BENCHMARK_PROMPTS
): MixedLoadDefinition {
  return {
    id: 'mixed-lilo-vs-siso',
    label: 'Mixed Load: LILO Background vs SISO Foreground',
    background: {
      id: 'mixed-background-lilo',
      label: 'Background Long Input / Long Output',
      prompt: promptSet.longPrompt,
      promptBucket: 'long',
      promptChars: promptSet.longPrompt.length,
      promptWords: countWords(promptSet.longPrompt),
      outputTokenLimit: 128,
      outputBucket: 'long',
      promptMode: operation,
      contextBucket: 'single-request',
      concurrency: 1,
    },
    foreground: {
      id: 'mixed-foreground-siso',
      label: 'Foreground Short Input / Short Output',
      prompt: promptSet.mixedForegroundPrompt,
      promptBucket: 'short',
      promptChars: promptSet.mixedForegroundPrompt.length,
      promptWords: countWords(promptSet.mixedForegroundPrompt),
      outputTokenLimit: 16,
      outputBucket: 'short',
      promptMode: operation,
      contextBucket: 'single-request',
      concurrency: 1,
    },
    concurrency: 2,
  };
}

export function runtimeOptionsForMixedLoad(
  runtime: BenchmarkRuntimeOptions,
  concurrency: number
): BenchmarkRuntimeOptions {
  const context = runtime.context ?? {};
  return {
    ...runtime,
    context: {
      ...context,
      n_parallel: Math.max(context.n_parallel ?? 1, concurrency),
    },
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

type BenchmarkEnvironmentInfo = {
  readonly hasNavigatorGpu?: boolean;
  readonly adapterAvailable?: boolean;
  readonly adapterLabel?: string | null;
  readonly adapterVendor?: string | null;
  readonly adapterArchitecture?: string | null;
  readonly adapterDescription?: string | null;
  readonly adapterInfo?: {
    readonly vendor?: string;
    readonly architecture?: string;
    readonly device?: string;
    readonly description?: string;
  } | null;
};

type RuntimeBackendInfo = {
  readonly webgpuCompiled?: boolean;
  readonly webgpuRegistered?: boolean;
  readonly webgpuDeviceCount?: number;
  readonly gpuOffloadSupported?: boolean;
  readonly availableBackends?: readonly {
    readonly name: string;
    readonly deviceCount: number;
  }[];
  readonly devices?: readonly {
    readonly name?: string;
    readonly description?: string;
    readonly type?: string;
    readonly backendName?: string;
  }[];
  readonly webgpuAdapter?: {
    readonly vendor?: string;
    readonly architecture?: string;
    readonly device?: string;
    readonly description?: string;
  } | null;
};

export type BenchmarkBackendProfile = {
  readonly requestedExecutionMode: string;
  readonly requestedGpuLayers: unknown;
  readonly inferredExecutionBackend: string;
  readonly runtimeBackendStatus: string;
  readonly gpuOffloadSupported: boolean | null;
  readonly availableBackends: readonly string[];
  readonly backendRegistries: NonNullable<RuntimeBackendInfo['availableBackends']>;
  readonly runtimeDeviceCount: number;
  readonly runtimeAcceleratorDeviceCount: number;
  readonly runtimeDeviceLabels: readonly string[];
  readonly runtimeDevices: NonNullable<RuntimeBackendInfo['devices']>;
  readonly engineAdapter: {
    readonly vendor: string | null;
    readonly architecture: string | null;
    readonly device: string | null;
    readonly description: string | null;
  } | null;
  readonly hostAdapter: {
    readonly apiAvailable: boolean;
    readonly adapterAvailable: boolean;
    readonly adapterLabel: string | null;
    readonly adapterVendor: string | null;
    readonly adapterArchitecture: string | null;
    readonly adapterDescription: string | null;
  };
  readonly notes: readonly string[];
};

export function buildBenchmarkBackendProfile(
  environment: BenchmarkEnvironmentInfo,
  runtimeBackend: RuntimeBackendInfo | null | undefined,
  requestedGpuLayers: unknown = null
): BenchmarkBackendProfile {
  const runtimeDevices = Array.isArray(runtimeBackend?.devices) ? runtimeBackend.devices : [];
  const acceleratorDevices = runtimeDevices.filter((device) => device.type !== 'cpu');
  const adapterInfo = environment?.adapterInfo;
  const engineAdapterInfo = runtimeBackend?.webgpuAdapter;
  const engineAdapter = engineAdapterInfo == null
    ? null
    : {
        vendor: engineAdapterInfo.vendor ?? null,
        architecture: engineAdapterInfo.architecture ?? null,
        device: engineAdapterInfo.device ?? null,
        description: engineAdapterInfo.description ?? null,
      };
  const notes: string[] = [];

  if (!environment.hasNavigatorGpu) {
    notes.push('navigator.gpu is unavailable in this browser session.');
  } else if (!environment.adapterAvailable) {
    notes.push('navigator.gpu is present, but requestAdapter() did not produce a usable adapter.');
  }

  const hostVendor = environment?.adapterVendor ?? adapterInfo?.vendor ?? null;
  if (
    engineAdapter?.vendor != null &&
    engineAdapter.vendor !== '' &&
    hostVendor != null &&
    hostVendor !== '' &&
    engineAdapter.vendor !== hostVendor
  ) {
    notes.push(
      `Engine adapter vendor (${engineAdapter.vendor}) differs from the main-thread adapter (${hostVendor}); ` +
        'the GPU process likely restarted onto a different physical GPU mid-session.'
    );
  }

  if (!runtimeBackend?.webgpuCompiled) {
    notes.push('The package build did not include ggml-webgpu.');
  } else if (!runtimeBackend.webgpuRegistered) {
    notes.push('ggml-webgpu was compiled, but the runtime did not register a usable WebGPU backend.');
  } else if ((runtimeBackend.webgpuDeviceCount ?? 0) <= 0) {
    notes.push('ggml-webgpu was registered, but it reported no runtime devices.');
  }

  const runtimeBackendStatus =
    runtimeBackend == null
      ? 'unknown'
      : !runtimeBackend.webgpuCompiled
        ? 'not-compiled'
        : !runtimeBackend.webgpuRegistered
          ? 'compiled-not-registered'
          : (runtimeBackend.webgpuDeviceCount ?? 0) <= 0
            ? 'registered-no-devices'
            : 'webgpu-ready';

  return {
    requestedExecutionMode:
      runtimeBackend == null
        ? 'unknown'
        : runtimeBackend.webgpuRegistered
          ? 'gpu-offload'
          : 'cpu-only',
    requestedGpuLayers,
    inferredExecutionBackend:
      environment.adapterAvailable &&
      runtimeBackend?.webgpuRegistered &&
      runtimeBackend.webgpuDeviceCount != null &&
      runtimeBackend.webgpuDeviceCount > 0 &&
      runtimeBackend.gpuOffloadSupported
        ? 'webgpu'
        : runtimeBackend != null
          ? 'cpu'
          : 'unknown',
    runtimeBackendStatus,
    gpuOffloadSupported: runtimeBackend?.gpuOffloadSupported ?? null,
    availableBackends: runtimeBackend?.availableBackends?.map((backend) => backend.name) ?? [],
    backendRegistries: runtimeBackend?.availableBackends ?? [],
    runtimeDeviceCount: runtimeDevices.length,
    runtimeAcceleratorDeviceCount: acceleratorDevices.length,
    runtimeDeviceLabels: runtimeDevices.map((device) =>
      `${device.backendName || device.type}:${device.description || device.name || device.type}`
    ),
    runtimeDevices,
    engineAdapter,
    hostAdapter: {
      apiAvailable: Boolean(environment?.hasNavigatorGpu),
      adapterAvailable: Boolean(environment?.adapterAvailable),
      adapterLabel: environment?.adapterLabel ?? adapterInfo?.device ?? adapterInfo?.description ?? null,
      adapterVendor: environment?.adapterVendor ?? adapterInfo?.vendor ?? null,
      adapterArchitecture: environment?.adapterArchitecture ?? adapterInfo?.architecture ?? null,
      adapterDescription: environment?.adapterDescription ?? adapterInfo?.description ?? null,
    },
    notes,
  };
}
