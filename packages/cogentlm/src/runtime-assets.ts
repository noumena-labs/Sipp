import type { CogentConfig } from './cogent-config.js';

export interface RuntimeUrls {
  moduleUrl: string;
  wasmUrl: string;
}

function normalizeOptionalString(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed == null || trimmed.length === 0 ? undefined : trimmed;
}

function currentLocationHref(): string | undefined {
  return typeof globalThis.location?.href === 'string' ? globalThis.location.href : undefined;
}

function currentLocationOrigin(): string | undefined {
  return typeof globalThis.location?.origin === 'string' ? globalThis.location.origin : undefined;
}

function parseConfiguredUrl(rawUrl: string, fieldName: string): URL {
  try {
    const baseHref = currentLocationHref();
    return baseHref == null ? new URL(rawUrl) : new URL(rawUrl, baseHref);
  } catch {
    throw new Error(`Invalid ${fieldName} value "${rawUrl}".`);
  }
}

export function getDefaultRuntimeUrls(): RuntimeUrls {
  return {
    moduleUrl: new URL('../wasm/cogentlm-wasm.js', import.meta.url).toString(),
    wasmUrl: new URL('../wasm/cogentlm-wasm.wasm', import.meta.url).toString(),
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
      ? {
          moduleUrl: new URL(getDefaultRuntimeUrls().moduleUrl),
          wasmUrl: new URL(getDefaultRuntimeUrls().wasmUrl),
        }
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
