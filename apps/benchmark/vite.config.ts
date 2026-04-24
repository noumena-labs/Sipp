import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const benchmarkAppDir = fileURLToPath(new URL('.', import.meta.url));
const cogentEngineEntry = path.resolve(
  benchmarkAppDir,
  '../../packages/cogent-engine/dist/esm/index.js'
);

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Use the built workspace entry directly so Vite does not serve the package
      // through an immutable /node_modules dependency URL that can stay stale
      // across local package rebuilds.
      '@noumena-labs/cogent-engine': cogentEngineEntry,
    },
    preserveSymlinks: true,
  },
  optimizeDeps: {
    exclude: ['@noumena-labs/cogent-engine'],
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
