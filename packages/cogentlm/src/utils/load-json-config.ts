export async function loadJsonConfig(
  configUrl: string,
  options: {
    readonly fetch?: typeof globalThis.fetch;
    readonly signal?: AbortSignal;
    readonly fetchLabel: string;
    readonly httpLabel: string;
  }
): Promise<unknown> {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (typeof fetchImpl !== 'function') {
    throw new Error(
      `${options.fetchLabel} requires a fetch implementation. Pass \`fetch\` explicitly in this runtime.`
    );
  }

  const response = await fetchImpl(configUrl, { signal: options.signal });
  if (!response.ok) {
    throw new Error(`${options.httpLabel} HTTP ${response.status}`);
  }

  return await response.json();
}
