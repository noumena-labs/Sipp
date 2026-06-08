import type { TimeBucketSnapshot } from './api.js';

export interface ChartPoint {
  readonly time: string;
  readonly requests: number;
  readonly errors: number;
  readonly rateLimited: number;
  readonly blocked: number;
  readonly tokens: number;
  readonly latency: number;
}

export function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-US').format(value);
}

export function formatBytes(value: number): string {
  const units = ['B', 'KiB', 'MiB', 'GiB'] as const;
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remaining = Math.floor(seconds % 60);
  if (hours > 0) {
    return `${hours}h ${minutes}m ${remaining}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${remaining}s`;
  }
  return `${remaining}s`;
}

export function errorRate(requests: number, errors: number): string {
  if (requests === 0) {
    return '0.0%';
  }
  return `${((errors / requests) * 100).toFixed(1)}%`;
}

export function toChartPoints(buckets: readonly TimeBucketSnapshot[]): readonly ChartPoint[] {
  return buckets.map((bucket) => ({
    time: new Date(bucket.unixSeconds * 1000).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    }),
    requests: bucket.requests,
    errors: bucket.errors,
    rateLimited: bucket.rateLimited,
    blocked: bucket.blocked,
    tokens: bucket.totalTokens,
    latency: bucket.averageLatencyMs,
  }));
}

export function sanitizeConcurrencyLimit(value: string): number | null {
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    return null;
  }
  const parsed = Number(trimmed);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error('Concurrency limit must be a positive integer.');
  }
  return parsed;
}

export function sanitizeRateLimit(value: string): number {
  const parsed = Number(value.trim());
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error('Rate limit must be a positive integer.');
  }
  return parsed;
}
