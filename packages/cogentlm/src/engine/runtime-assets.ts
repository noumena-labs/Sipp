import type { CogentConfig } from './engine-options.js';
import { resolveOptimizedPackageAssetUrl } from '../runtime/package-assets.js';
import { currentLocationOrigin, resolveUrl } from '../utils/url.js';

export interface RuntimeUrls {
  moduleUrl: string;
  wasmUrl: string;
}

function normalizeOptionalString(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed == null || trimmed.length === 0 ? undefined : trimmed;
}

function parseConfiguredUrl(rawUrl: string, fieldName: string): URL {
  return resolveUrl(rawUrl, fieldName);
}

export function getDefaultRuntimeUrls(importerUrl: string = import.meta.url): RuntimeUrls {
  const optimizedRuntimeAssetsUrl = resolveOptimizedPackageAssetUrl(
    'dist/esm/engine/runtime-assets.js',
    importerUrl
  );

  return optimizedRuntimeAssetsUrl == null
    ? {
      moduleUrl: new URL('../../wasm/cogentlm-wasm.js', import.meta.url).toString(),
      wasmUrl: new URL('../../wasm/cogentlm-wasm.wasm', import.meta.url).toString(),
    }
    : {
      moduleUrl: new URL('../../wasm/cogentlm-wasm.js', optimizedRuntimeAssetsUrl).toString(),
      wasmUrl: new URL('../../wasm/cogentlm-wasm.wasm', optimizedRuntimeAssetsUrl).toString(),
    };
}

export function resolveTrustedOrigins(
  configuredOrigins: CogentConfig['trustedOrigins']
): Set<string> {
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
  config: Pick<CogentConfig, 'moduleUrl' | 'wasmUrl' | 'trustedOrigins'>
): RuntimeUrls {
  const configuredModuleUrl = normalizeOptionalString(config.moduleUrl);
  const configuredWasmUrl = normalizeOptionalString(config.wasmUrl);

  if ((configuredModuleUrl == null) !== (configuredWasmUrl == null)) {
    throw new Error(
      'Both "moduleUrl" and "wasmUrl" must be provided when overriding CogentEngine runtime assets.'
    );
  }

  const resolved =
    configuredModuleUrl == null
      ? (() => {
        const defaults = getDefaultRuntimeUrls();
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
  };
}
