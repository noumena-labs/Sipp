import type { SippClientOptions } from './browser-client.js';
import {
  QueryError,
  type BrowserBackendPreference,
  type WebGpuAdapterInfo,
} from '../models/types.js';
import { currentLocationOrigin, resolveUrl } from '../utils/url.js';

const VITE_OPTIMIZED_DEPS_SEGMENT = '/node_modules/.vite/deps/';
const INTERNAL_PACKAGE_ROOT = 'node_modules/@noumena-labs/sipp';
const PUBLIC_PACKAGE_ROOT = 'node_modules/@sipphq/sipp';

export interface RuntimeUrls {
  moduleUrl: string;
  wasmUrl: string;
  threading: WasmThreadingMode;
}

/** The backend one Worker runs, resolved once before its runtime is chosen. */
export interface ResolvedBrowserBackend {
  readonly backend: Exclude<BrowserBackendPreference, 'auto'>;
  readonly webgpuAdapter: WebGpuAdapterInfo | null;
}

export interface RuntimeAssetSelection extends RuntimeUrls {
  readonly backend: ResolvedBrowserBackend;
}

export type WasmThreadingMode = 'single-thread' | 'pthread';

interface RuntimeUrlResolutionOptions {
  readonly bundledRuntimeUrls?: () => RuntimeUrls;
}

interface RuntimeAssetSelectionOptions {
  readonly importerUrl?: string;
  readonly probeWasmJspi?: () => Promise<boolean>;
}

type NavigatorWithGpu = Navigator & {
  gpu?: {
    requestAdapter(): Promise<NavigatorGpuAdapter | null>;
  };
};
type NavigatorGpuAdapter = {
  readonly features?: {
    has(feature: string): boolean;
  };
  readonly info?: Partial<WebGpuAdapterInfo> | null;
  requestAdapterInfo?: () => Promise<Partial<WebGpuAdapterInfo> | null>;
};

const DEFAULT_BUNDLED_RUNTIME = 'sipp-wasm-pthread';
/** Built without WebGPU, so a CPU session never creates a WebGPU device. */
const CPU_BUNDLED_RUNTIME = 'sipp-wasm-pthread-cpu-nojspi';
const CPU_RESOLVED_BACKEND: ResolvedBrowserBackend = { backend: 'cpu', webgpuAdapter: null };

const JSPI_PROBE_MODULE = Uint8Array.from([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
  0x02, 0x07, 0x01, 0x01, 0x6d, 0x01, 0x73, 0x00, 0x00,
  0x03, 0x02, 0x01, 0x00,
  0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x01,
  0x0a, 0x06, 0x01, 0x04, 0x00, 0x10, 0x00, 0x0b,
]);
/** Verifies that JSPI can suspend and resume an actual Wasm export. */
const supportsFunctionalWasmJspi = memoizeWasmJspiProbe(runFunctionalWasmJspiProbe);

function normalizeOptionalString(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed == null || trimmed.length === 0 ? undefined : trimmed;
}

export function resolveOptimizedPackageAssetUrl(
  packageRelativePath: string,
  importerUrl: string
): string | null {
  let parsed: URL;

  try {
    parsed = new URL(importerUrl);
  } catch {
    return null;
  }

  const optimizedDepsIndex = parsed.pathname.indexOf(VITE_OPTIMIZED_DEPS_SEGMENT);
  if (optimizedDepsIndex < 0) {
    return null;
  }

  const packageRoot = packageRootForOptimizedDependency(
    parsed.pathname.slice(optimizedDepsIndex + VITE_OPTIMIZED_DEPS_SEGMENT.length)
  );
  if (packageRoot == null) {
    return null;
  }

  const basePath = parsed.pathname.slice(0, optimizedDepsIndex);
  const normalizedRelativePath = packageRelativePath.replace(/^\/+/, '');
  parsed.pathname = `${basePath}/${packageRoot}/${normalizedRelativePath}`;
  parsed.search = '';
  parsed.hash = '';

  return parsed.toString();
}

function packageRootForOptimizedDependency(optimizedPath: string): string | null {
  const fileName = optimizedPath.split('/')[0] ?? '';
  if (fileName.startsWith('@noumena-labs_sipp')) {
    return INTERNAL_PACKAGE_ROOT;
  }
  if (fileName.startsWith('@sipphq_sipp')) {
    return PUBLIC_PACKAGE_ROOT;
  }
  return null;
}

/** @internal Exported for tests; not part of the package's public surface. */
export function supportsWasmPthreads(): boolean {
  return (
    typeof SharedArrayBuffer !== 'undefined' &&
    globalThis.crossOriginIsolated === true &&
    typeof Worker !== 'undefined'
  );
}

export function resolveRuntimeThreadingMode(
  config: Pick<
    SippClientOptions,
    'moduleUrl' | 'wasmUrl' | 'wasmThreading'
  >
): WasmThreadingMode {
  const hasSelectedRuntimeOverride =
    normalizeOptionalString(config.moduleUrl) != null ||
    normalizeOptionalString(config.wasmUrl) != null;

  if (config.wasmThreading === 'single-thread' && hasSelectedRuntimeOverride) {
    return 'single-thread';
  }

  if (config.wasmThreading === 'single-thread') {
    throw new Error(
      'The bundled Sipp browser runtime is pthread-only. Provide moduleUrl and wasmUrl for a custom single-thread runtime.'
    );
  }

  assertWasmPthreadsSupported();
  return 'pthread';
}

function assertWasmPthreadsSupported(): void {
  if (supportsWasmPthreads()) {
    return;
  }
  throw new Error(
    'The bundled Sipp browser runtime requires SharedArrayBuffer and cross-origin isolation. Serve the app with COOP/COEP headers, or set wasmThreading: "single-thread" with moduleUrl and wasmUrl for a custom single-thread runtime.'
  );
}

function bundledRuntimeUrls(
  artifactName: string,
  importerUrl: string = import.meta.url
): RuntimeUrls {
  const optimizedRuntimeAssetsUrl = resolveOptimizedPackageAssetUrl(
    'dist/esm/engine/runtime-assets.js',
    importerUrl
  );
  const runtimeAssetsBaseUrl = optimizedRuntimeAssetsUrl ?? import.meta.url;

  return {
    moduleUrl: new URL(
      `../../wasm/${artifactName}.js`,
      runtimeAssetsBaseUrl
    ).toString(),
    wasmUrl: new URL(
      `../../wasm/${artifactName}.wasm`,
      runtimeAssetsBaseUrl
    ).toString(),
    threading: 'pthread',
  };
}

/** @internal Exported for tests; not part of the package's public surface. */
export function memoizeWasmJspiProbe(
  probe: () => Promise<boolean>
): () => Promise<boolean> {
  let result: Promise<boolean> | null = null;
  return async () => {
    result ??= probe().catch(() => false);
    return await result;
  };
}

async function runFunctionalWasmJspiProbe(): Promise<boolean> {
  if (typeof WebAssembly === 'undefined') {
    return false;
  }
  const jspi = WebAssembly as typeof WebAssembly & {
    readonly Suspending?: new (
      callback: () => Promise<number>
    ) => WebAssembly.ImportValue;
    readonly promising?: (
      exported: WebAssembly.ExportValue
    ) => () => Promise<number>;
  };
  if (typeof jspi.Suspending !== 'function' || typeof jspi.promising !== 'function') {
    return false;
  }

  try {
    let resumed = false;
    const suspended = new jspi.Suspending(async () => {
      await Promise.resolve();
      resumed = true;
      return 37;
    });
    const instantiated = await WebAssembly.instantiate(JSPI_PROBE_MODULE, {
      m: { s: suspended },
    });
    const instance = instantiated instanceof WebAssembly.Instance
      ? instantiated
      : instantiated.instance;
    const run = instance.exports.run;
    if (typeof run !== 'function') {
      return false;
    }
    const result = await jspi.promising(run)();
    return resumed && result === 37;
  } catch {
    return false;
  }
}

/**
 * Resolves the Worker's backend, then the runtime that runs it. A resolved CPU
 * backend selects the bundled CPU-only runtime, so CPU sessions never
 * initialize WebGPU. Explicit `moduleUrl`/`wasmUrl` assets are used as given.
 */
export async function resolveRuntimeAssetSelection(
  config: Pick<
    SippClientOptions,
    | 'moduleUrl'
    | 'wasmUrl'
    | 'trustedOrigins'
    | 'wasmThreading'
  > & { readonly backend?: BrowserBackendPreference },
  options: RuntimeAssetSelectionOptions = {}
): Promise<RuntimeAssetSelection> {
  if (hasRuntimeUrlOverride(config)) {
    return {
      ...resolveRuntimeUrls(config),
      backend: await resolveBrowserBackend(config.backend, true),
    };
  }

  let supportsJspi = false;
  try {
    supportsJspi = await (options.probeWasmJspi ?? supportsFunctionalWasmJspi)();
  } catch {
    supportsJspi = false;
  }
  const backend = await resolveBrowserBackend(config.backend, supportsJspi);
  const artifactName = backend.backend === 'cpu'
    ? CPU_BUNDLED_RUNTIME
    : DEFAULT_BUNDLED_RUNTIME;
  return {
    ...resolveRuntimeUrls(config, {
      bundledRuntimeUrls: () => bundledRuntimeUrls(artifactName, options.importerUrl),
    }),
    backend,
  };
}

/**
 * Resolves a backend preference against this scope's WebGPU adapter. The
 * wasm backend requests its adapter from the same scope, so the result
 * matches the GPU inference runs on. ggml-webgpu registers no device without
 * `shader-f16`, so both `auto` and `webgpu` require it.
 */
async function resolveBrowserBackend(
  backend: BrowserBackendPreference | undefined,
  supportsWebGpuRuntime: boolean
): Promise<ResolvedBrowserBackend> {
  if (!supportsWebGpuRuntime) {
    if (backend === 'webgpu') {
      throw new QueryError(
        'UNSUPPORTED_OPERATION',
        'The active browser runtime is CPU-only because this browser did not pass the JSPI ' +
          'suspend/resume probe. Remove backend: "webgpu", or supply WebGPU-capable ' +
          'moduleUrl and wasmUrl assets.'
      );
    }
    return CPU_RESOLVED_BACKEND;
  }
  if (backend === 'cpu') {
    return CPU_RESOLVED_BACKEND;
  }
  const gpu = (globalThis.navigator as NavigatorWithGpu | undefined)?.gpu;
  const adapter = gpu == null ? null : await gpu.requestAdapter();
  if (adapter?.features?.has('shader-f16') !== true) {
    if (backend === 'webgpu') {
      throw new QueryError(
        'UNSUPPORTED_OPERATION',
        adapter == null
          ? 'backend: "webgpu" requires WebGPU, but this browser provides no WebGPU adapter.'
          : 'backend: "webgpu" requires the WebGPU "shader-f16" feature, which this ' +
            'adapter does not support. Use backend: "auto" or "cpu".'
      );
    }
    return CPU_RESOLVED_BACKEND;
  }
  return { backend: 'webgpu', webgpuAdapter: await readWebGpuAdapterInfo(adapter) };
}

/** Reads the adapter identity, which tells hybrid-GPU machines' GPUs apart. */
async function readWebGpuAdapterInfo(
  adapter: NavigatorGpuAdapter
): Promise<WebGpuAdapterInfo | null> {
  const info = adapter.info ?? (await adapter.requestAdapterInfo?.()) ?? null;
  if (info == null) {
    return null;
  }
  return {
    vendor: info.vendor ?? '',
    architecture: info.architecture ?? '',
    device: info.device ?? '',
    description: info.description ?? '',
  };
}

function hasRuntimeUrlOverride(
  config: Pick<
    SippClientOptions,
    'moduleUrl' | 'wasmUrl'
  >
): boolean {
  return (
    normalizeOptionalString(config.moduleUrl) != null ||
    normalizeOptionalString(config.wasmUrl) != null
  );
}

function resolveTrustedOrigins(configuredOrigins: SippClientOptions['trustedOrigins']): Set<string> {
  if (configuredOrigins != null && configuredOrigins.length > 0) {
    const allowed = new Set<string>();
    for (const originValue of configuredOrigins) {
      allowed.add(resolveUrl(originValue, 'trustedOrigins').origin);
    }
    return allowed;
  }

  const origin = currentLocationOrigin();
  return origin == null ? new Set<string>() : new Set([origin]);
}

export function resolveRuntimeUrls(
  config: Pick<
    SippClientOptions,
    | 'moduleUrl'
    | 'wasmUrl'
    | 'trustedOrigins'
    | 'wasmThreading'
  >,
  options: RuntimeUrlResolutionOptions = {}
): RuntimeUrls {
  const configuredModuleUrl = normalizeOptionalString(config.moduleUrl);
  const configuredWasmUrl = normalizeOptionalString(config.wasmUrl);

  if ((configuredModuleUrl == null) !== (configuredWasmUrl == null)) {
    throw new Error(
      'Both "moduleUrl" and "wasmUrl" must be provided when overriding SippClient runtime assets.'
    );
  }

  const threading = resolveRuntimeThreadingMode(config);
  let resolved: { moduleUrl: URL; wasmUrl: URL };
  if (threading === 'single-thread') {
    resolved = {
      moduleUrl: resolveUrl(configuredModuleUrl!, 'moduleUrl'),
      wasmUrl: resolveUrl(configuredWasmUrl!, 'wasmUrl'),
    };
  } else if (configuredModuleUrl != null) {
    resolved = {
      moduleUrl: resolveUrl(configuredModuleUrl, 'moduleUrl'),
      wasmUrl: resolveUrl(configuredWasmUrl!, 'wasmUrl'),
    };
  } else {
    const defaults = options.bundledRuntimeUrls?.();
    if (defaults == null) {
      throw new Error(
        'Bundled runtime assets must be selected asynchronously inside the Worker.'
      );
    }
    resolved = {
      moduleUrl: resolveUrl(defaults.moduleUrl, 'moduleUrl'),
      wasmUrl: resolveUrl(defaults.wasmUrl, 'wasmUrl'),
    };
  }

  const trustedOrigins = resolveTrustedOrigins(config.trustedOrigins);
  if (trustedOrigins.size > 0) {
    if (!trustedOrigins.has(resolved.moduleUrl.origin)) {
      throw new Error(
        `Blocked moduleUrl origin "${resolved.moduleUrl.origin}". Add it to trustedOrigins to allow it.`
      );
    }
    if (!trustedOrigins.has(resolved.wasmUrl.origin)) {
      throw new Error(
        `Blocked wasmUrl origin "${resolved.wasmUrl.origin}". Add it to trustedOrigins to allow it.`
      );
    }
  }

  return {
    moduleUrl: resolved.moduleUrl.toString(),
    wasmUrl: resolved.wasmUrl.toString(),
    threading,
  };
}
