import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { cogentEngineDistWatch } from '../cogentlm-dist-watch';

const proactiveUiAppDir = fileURLToPath(new URL('.', import.meta.url));
const cogentEngineEntry = path.resolve(
  proactiveUiAppDir,
  '../../packages/cogentlm/dist/esm/index.js'
);

export default defineConfig({
  plugins: [react(), cogentEngineDistWatch()],
  resolve: {
    alias: {
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
