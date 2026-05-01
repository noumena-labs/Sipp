import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const examplesDir = fileURLToPath(new URL('.', import.meta.url));
const cogentlmEntry = path.resolve(
  examplesDir,
  '../../packages/cogentlm/dist/esm/index.js'
);

export default defineConfig({
  resolve: {
    alias: {
      'cogentlm': cogentlmEntry,
    },
    preserveSymlinks: true,
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: {
    exclude: ['cogentlm'],
  },
});
