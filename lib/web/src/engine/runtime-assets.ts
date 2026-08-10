import type { SippClientOptions } from './browser-client.js';
import { currentLocationOrigin, resolveUrl } from '../utils/url.js';

const VITE_OPTIMIZED_DEPS_SEGMENT = '/node_modules/.vite/deps/';
const INTERNAL_PACKAGE_ROOT = 'node_modules/@noumena-labs/sipp';
const PUBLIC_PACKAGE_ROOT = 'node_modules/@sipphq/sipp';

export interface RuntimeUrls {
  moduleUrl: string;
  wasmUrl: string;
  threading: WasmThreadingMode;
}

export interface RuntimeAssetSelection extends RuntimeUrls {
  readonly backendConstraint: RuntimeBackendConstraint | null;
}

export type WasmThreadingMode = 'single-thread' | 'pthread';
export type RuntimeBackendConstraint = 'cpu-only';

interface BundledRuntimeAsset {
  readonly artifactName: string;
  readonly backendConstraint: RuntimeBackendConstraint | null;
}

interface RuntimeUrlResolutionOptions {
  readonly bundledRuntimeUrls?: () => RuntimeUrls;
}

interface RuntimeAssetSelectionOptions {
  readonly importerUrl?: string;
  readonly probeWasmJspi?: () => Promise<boolean>;
}

const DEFAULT_BUNDLED_RUNTIME: BundledRuntimeAsset = {
  artifactName: 'sipp-wasm-pthread',
  backendConstraint: null,
};
const CPU_NOJSPI_BUNDLED_RUNTIME: BundledRuntimeAsset = {
  artifactName: 'sipp-wasm-pthread-cpu-nojspi',
  backendConstraint: 'cpu-only',
};

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
  runtime: BundledRuntimeAsset,
  importerUrl: string = import.meta.url
): RuntimeUrls {
  const optimizedRuntimeAssetsUrl = resolveOptimizedPackageAssetUrl(
    'dist/esm/engine/runtime-assets.js',
    importerUrl
  );
  const runtimeAssetsBaseUrl = optimizedRuntimeAssetsUrl ?? import.meta.url;

  return {
    moduleUrl: new URL(
      `../../wasm/${runtime.artifactName}.js`,
      runtimeAssetsBaseUrl
    ).toString(),
    wasmUrl: new URL(
      `../../wasm/${runtime.artifactName}.wasm`,
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

/** Resolves one bundled or explicit runtime choice after the functional probe. */
export async function resolveRuntimeAssetSelection(
  config: Pick<
    SippClientOptions,
    | 'moduleUrl'
    | 'wasmUrl'
    | 'trustedOrigins'
    | 'wasmThreading'
  >,
  options: RuntimeAssetSelectionOptions = {}
): Promise<RuntimeAssetSelection> {
  if (hasRuntimeUrlOverride(config)) {
    return {
      ...resolveRuntimeUrls(config),
      backendConstraint: null,
    };
  }

  let supportsJspi = false;
  try {
    supportsJspi = await (options.probeWasmJspi ?? supportsFunctionalWasmJspi)();
  } catch {
    supportsJspi = false;
  }
  const runtime = supportsJspi
    ? DEFAULT_BUNDLED_RUNTIME
    : CPU_NOJSPI_BUNDLED_RUNTIME;
  return {
    ...resolveRuntimeUrls(config, {
      bundledRuntimeUrls: () => bundledRuntimeUrls(runtime, options.importerUrl),
    }),
    backendConstraint: runtime.backendConstraint,
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
