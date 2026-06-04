import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { cogentClientDistWatch } from '../cogentlm-dist-watch';

const benchmarkAppDir = fileURLToPath(new URL('.', import.meta.url));
const cogentClientDistDir = path.resolve(
  benchmarkAppDir,
  '../../.build/artifacts/npm/cogentlm/dist/esm'
);
const cogentClientEntry = path.join(cogentClientDistDir, 'index.js');
const appOutDir = path.resolve(benchmarkAppDir, '../../.build/artifacts/apps/benchmark');

export default defineConfig({
  plugins: [react(), cogentClientDistWatch()],
  build: {
    outDir: appOutDir,
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      // Use the built workspace entry directly so Vite does not serve the package
      // through an immutable /node_modules dependency URL that can stay stale
      // across local package rebuilds.
      '@noumena-labs/cogentlm': cogentClientEntry,
    },
    preserveSymlinks: true,
  },
  optimizeDeps: {
    exclude: ['@noumena-labs/cogentlm'],
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
