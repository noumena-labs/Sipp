import { describe, expect, it } from 'bun:test';
import { sippViteConfig } from '../src/vite.js';

describe('sippViteConfig', () => {
  it('configures cross-origin isolation for development and preview', () => {
    const config = sippViteConfig();
    const expectedHeaders = {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    };

    expect(config.server.headers).toEqual(expectedHeaders);
    expect(config.preview.headers).toEqual(expectedHeaders);
  });
});
