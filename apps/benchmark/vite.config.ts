import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { cogentEngineDistWatch } from '../cogentlm-dist-watch';

const benchmarkAppDir = fileURLToPath(new URL('.', import.meta.url));
const cogentEngineEntry = path.resolve(
  benchmarkAppDir,
  '../../packages/cogentlm/dist/esm/index.js'
);

export default defineConfig({
  plugins: [react(), cogentEngineDistWatch()],
  resolve: {
    alias: {
      // Use the built workspace entry directly so Vite does not serve the package
      // through an immutable /node_modules dependency URL that can stay stale
      // across local package rebuilds.
      'cogentlm': cogentEngineEntry,
    },
    preserveSymlinks: true,
  },
  optimizeDeps: {
    exclude: ['cogentlm'],
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
