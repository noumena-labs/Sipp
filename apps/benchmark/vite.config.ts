import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { cogentEngineDistWatch } from '../cogentlm-dist-watch';

const benchmarkAppDir = fileURLToPath(new URL('.', import.meta.url));
const cogentEngineDistDir = path.resolve(
  benchmarkAppDir,
  '../../.build/artifacts/npm/cogentlm-browser/dist/esm'
);
const cogentEngineEntry = path.join(cogentEngineDistDir, 'index.js');
const appOutDir = path.resolve(benchmarkAppDir, '../../.build/artifacts/apps/benchmark');

export default defineConfig({
  plugins: [react(), cogentEngineDistWatch()],
  build: {
    outDir: appOutDir,
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      // Use the built workspace entry directly so Vite does not serve the package
      // through an immutable /node_modules dependency URL that can stay stale
      // across local package rebuilds.
      '@noumena-labs/cogentlm-browser': cogentEngineEntry,
    },
    preserveSymlinks: true,
  },
  optimizeDeps: {
    exclude: ['@noumena-labs/cogentlm-browser'],
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
});
