export function stableJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => stableJson(entry === undefined ? null : entry)).join(',')}]`;
  }
  if (value != null && typeof value === 'object') {
    return `{${Object.entries(value as Record<string, unknown>)
      .filter(([, entry]) => entry !== undefined)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => `${JSON.stringify(key)}:${stableJson(entry)}`)
      .join(',')}}`;
  }
  const json = JSON.stringify(value);
  if (json === undefined) {
    throw new TypeError('stableJson does not support undefined, functions, or symbols at the top level.');
  }
  return json;
}
