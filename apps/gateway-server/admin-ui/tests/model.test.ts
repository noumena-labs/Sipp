import { describe, expect, test } from 'bun:test';
import {
  errorRate,
  formatBytes,
  formatDuration,
  sanitizeConcurrencyLimit,
  sanitizeRateLimit,
  toChartPoints,
} from '../src/model.js';

describe('dashboard model helpers', () => {
  test('formats operational values', () => {
    expect(formatBytes(1048576)).toBe('1.0 MiB');
    expect(formatDuration(3661)).toBe('1h 1m 1s');
    expect(errorRate(10, 2)).toBe('20.0%');
    expect(errorRate(0, 2)).toBe('0.0%');
  });

  test('validates runtime controls', () => {
    expect(sanitizeConcurrencyLimit('')).toBeNull();
    expect(sanitizeConcurrencyLimit('4')).toBe(4);
    expect(sanitizeRateLimit('60')).toBe(60);
    expect(() => sanitizeConcurrencyLimit('0')).toThrow();
    expect(() => sanitizeRateLimit('1.5')).toThrow();
  });

  test('maps time-series buckets to chart points', () => {
    const points = toChartPoints([
      {
        unixSeconds: 100,
        requests: 3,
        errors: 1,
        rateLimited: 2,
        blocked: 0,
        inputTokens: 4,
        outputTokens: 5,
        totalTokens: 9,
        averageLatencyMs: 12,
      },
    ]);
    expect(points[0].requests).toBe(3);
    expect(points[0].tokens).toBe(9);
    expect(points[0].latency).toBe(12);
  });
});
