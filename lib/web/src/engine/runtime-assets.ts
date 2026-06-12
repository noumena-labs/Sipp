import type { SippClientOptions } from './browser-client.js';
import { currentLocationOrigin, resolveUrl } from '../utils/url.js';

const VITE_OPTIMIZED_DEPS_SEGMENT = '/node_modules/.vite/deps/';
const INTERNAL_PACKAGE_ROOT = 'node_modules/@noumena-labs/sipp';
const PUBLIC_PACKAGE_ROOT = 'node_modules/sipp';

export interface RuntimeUrls {
  moduleUrl: string;
  wasmUrl: string;
  threading: WasmThreadingMode;
}

export type WasmThreadingPreference = 'single-thread' | 'pthread';
export type WasmThreadingMode = 'single-thread' | 'pthread';

function normalizeOptionalString(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed == null || trimmed.length === 0 ? undefined : trimmed;
}

function parseConfiguredUrl(rawUrl: string, fieldName: string): URL {
  return resolveUrl(rawUrl, fieldName);
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
  if (fileName.startsWith('sipp')) {
    return PUBLIC_PACKAGE_ROOT;
  }
  return null;
}

export function getDefaultRuntimeUrls(importerUrl: string = import.meta.url): RuntimeUrls {
  const threading = resolveRuntimeThreadingMode({});
  return getDefaultRuntimeUrlsForThreading(threading, importerUrl);
}

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
    'moduleUrl' | 'wasmUrl' | 'pthreadModuleUrl' | 'pthreadWasmUrl' | 'wasmThreading'
  >
): WasmThreadingMode {
  const configuredSingleThread =
    normalizeOptionalString(config.moduleUrl) != null ||
    normalizeOptionalString(config.wasmUrl) != null;
  const configuredPthread =
    normalizeOptionalString(config.pthreadModuleUrl) != null ||
    normalizeOptionalString(config.pthreadWasmUrl) != null;

  if (configuredSingleThread && !configuredPthread && config.wasmThreading !== 'pthread') {
    return 'single-thread';
  }

  switch (config.wasmThreading ?? 'single-thread') {
    case 'single-thread':
      return 'single-thread';
    case 'pthread':
      assertWasmPthreadsSupported();
      return 'pthread';
  }
}

function assertWasmPthreadsSupported(): void {
  if (supportsWasmPthreads()) {
    return;
  }
  throw new Error(
    'The pthread wasm runtime requires SharedArrayBuffer and cross-origin isolation. Serve the app with COOP/COEP headers or use wasmThreading: "single-thread".'
  );
}

function getDefaultRuntimeUrlsForThreading(
  threading: WasmThreadingMode,
  importerUrl: string = import.meta.url
): RuntimeUrls {
  const optimizedRuntimeAssetsUrl = resolveOptimizedPackageAssetUrl(
    'dist/esm/engine/runtime-assets.js',
    importerUrl
  );
  const artifactPrefix = threading === 'pthread' ? 'sipp-wasm-pthread' : 'sipp-wasm';

  const urls = optimizedRuntimeAssetsUrl == null
    ? {
      moduleUrl: new URL(`../../wasm/${artifactPrefix}.js`, import.meta.url).toString(),
      wasmUrl: new URL(`../../wasm/${artifactPrefix}.wasm`, import.meta.url).toString(),
    }
    : {
      moduleUrl: new URL(`../../wasm/${artifactPrefix}.js`, optimizedRuntimeAssetsUrl).toString(),
      wasmUrl: new URL(`../../wasm/${artifactPrefix}.wasm`, optimizedRuntimeAssetsUrl).toString(),
    };

  return {
    ...urls,
    threading,
  };
}

function resolveTrustedOrigins(configuredOrigins: SippClientOptions['trustedOrigins']): Set<string> {
  if (configuredOrigins != null && configuredOrigins.length > 0) {
    const allowed = new Set<string>();
    for (const originValue of configuredOrigins) {
      allowed.add(parseConfiguredUrl(originValue, 'trustedOrigins').origin);
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
    | 'pthreadModuleUrl'
    | 'pthreadWasmUrl'
    | 'trustedOrigins'
    | 'wasmThreading'
  >
): RuntimeUrls {
  const configuredModuleUrl = normalizeOptionalString(config.moduleUrl);
  const configuredWasmUrl = normalizeOptionalString(config.wasmUrl);
  const configuredPthreadModuleUrl = normalizeOptionalString(config.pthreadModuleUrl);
  const configuredPthreadWasmUrl = normalizeOptionalString(config.pthreadWasmUrl);

  if ((configuredModuleUrl == null) !== (configuredWasmUrl == null)) {
    throw new Error(
      'Both "moduleUrl" and "wasmUrl" must be provided when overriding SippClient runtime assets.'
    );
  }

  if ((configuredPthreadModuleUrl == null) !== (configuredPthreadWasmUrl == null)) {
    throw new Error(
      'Both "pthreadModuleUrl" and "pthreadWasmUrl" must be provided when overriding SippClient pthread runtime assets.'
    );
  }

  const threading = resolveRuntimeThreadingMode(config);
  const resolved =
    threading === 'pthread'
      ? configuredPthreadModuleUrl == null
        ? configuredModuleUrl != null && config.wasmThreading === 'pthread'
          ? {
            moduleUrl: parseConfiguredUrl(configuredModuleUrl, 'moduleUrl'),
            wasmUrl: parseConfiguredUrl(configuredWasmUrl!, 'wasmUrl'),
          }
          : (() => {
            const defaults = getDefaultRuntimeUrlsForThreading('pthread');
            return {
              moduleUrl: new URL(defaults.moduleUrl),
              wasmUrl: new URL(defaults.wasmUrl),
            };
          })()
        : {
          moduleUrl: parseConfiguredUrl(configuredPthreadModuleUrl, 'pthreadModuleUrl'),
          wasmUrl: parseConfiguredUrl(configuredPthreadWasmUrl!, 'pthreadWasmUrl'),
        }
      : configuredModuleUrl == null
        ? (() => {
          const defaults = getDefaultRuntimeUrlsForThreading('single-thread');
          return {
            moduleUrl: new URL(defaults.moduleUrl),
            wasmUrl: new URL(defaults.wasmUrl),
          };
        })()
        : {
          moduleUrl: parseConfiguredUrl(configuredModuleUrl, 'moduleUrl'),
          wasmUrl: parseConfiguredUrl(configuredWasmUrl!, 'wasmUrl'),
        };

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
