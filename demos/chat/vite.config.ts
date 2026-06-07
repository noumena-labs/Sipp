import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const examplesDir = fileURLToPath(new URL('.', import.meta.url));
const cogentlmDistDir = path.resolve(
  examplesDir,
  '../../.build/artifacts/npm/cogentlm/dist/esm'
);
const cogentlmEntry = path.join(cogentlmDistDir, 'index.js');
const appOutDir = path.resolve(examplesDir, '../../.build/artifacts/demos/chat');

export default defineConfig({
  build: {
    outDir: appOutDir,
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
  optimizeDeps: {
    exclude: ['@noumena-labs/cogentlm'],
  },
});
