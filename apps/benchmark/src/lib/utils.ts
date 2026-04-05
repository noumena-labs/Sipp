export function round(value: number): number {
  return Number(value.toFixed(3));
}

export function formatMs(value: number): string {
  return `${round(value)} ms`;
}

export function formatMiB(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes) || bytes < 0) {
    return 'n/a';
  }
  if (bytes >= 1024 * 1024) {
    return formatMiB(bytes);
  }
  if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(2)} KiB`;
  }
  return `${bytes} B`;
}

export function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

export async function measureAsync<T>(fn: () => Promise<T>): Promise<{ ms: number; value: T }> {
  const start = performance.now();
  const value = await fn();
  return {
    ms: round(performance.now() - start),
    value,
  };
}

export function maxNullable(values: (number | null | undefined)[]): number | null {
  const filtered = values.filter((value): value is number => value != null && Number.isFinite(value));
  if (filtered.length === 0) {
    return null;
  }
  return Math.max(...filtered);
}
