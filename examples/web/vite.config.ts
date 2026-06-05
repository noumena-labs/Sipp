import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const exampleDir = fileURLToPath(new URL('.', import.meta.url));
const cogentlmDistDir = path.resolve(
  exampleDir,
  '../../.build/artifacts/npm/cogentlm/dist/esm',
);
const cogentlmEntry = path.join(cogentlmDistDir, 'index.js');

export default defineConfig({
  build: {
    outDir: path.resolve(exampleDir, '../../.build/artifacts/examples/web'),
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      '@noumena-labs/cogentlm': cogentlmEntry,
    },
    preserveSymlinks: true,
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: {
    exclude: ['@noumena-labs/cogentlm'],
  },
});
