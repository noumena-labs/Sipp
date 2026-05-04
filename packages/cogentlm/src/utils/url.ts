export function currentLocationHref(): string | undefined {
  return typeof globalThis.location?.href === 'string' ? globalThis.location.href : undefined;
}

export function currentLocationOrigin(): string | undefined {
  const location = globalThis.location;
  if (typeof location?.origin === 'string') {
    return location.origin;
  }

  const href = currentLocationHref();
  if (href == null) {
    return undefined;
  }

  try {
    return new URL(href).origin;
  } catch {
    return undefined;
  }
}

export function resolveUrl(rawUrl: string, fieldName: string): URL {
  try {
    const baseHref = currentLocationHref();
    return baseHref == null ? new URL(rawUrl) : new URL(rawUrl, baseHref);
  } catch {
    throw new Error(`Invalid ${fieldName} value "${rawUrl}".`);
  }
}
