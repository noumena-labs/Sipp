import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { cogentEngineDistWatch } from '../cogent-engine-dist-watch';

const proactiveUiAppDir = fileURLToPath(new URL('.', import.meta.url));
const cogentEngineEntry = path.resolve(
  proactiveUiAppDir,
  '../../packages/cogent-engine/dist/esm/index.js'
);

export default defineConfig({
  plugins: [react(), cogentEngineDistWatch()],
  resolve: {
    alias: {
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
