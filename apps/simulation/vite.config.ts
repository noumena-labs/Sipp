import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const simAppDir = fileURLToPath(new URL('.', import.meta.url));
const cogentEngineEntry = path.resolve(
  simAppDir,
  '../../packages/cogent-engine/dist/esm/index.js'
);
const cogentEngineCharacterEntry = path.resolve(
  simAppDir,
  '../../packages/cogent-engine/dist/esm/character/index.js'
);
const cogentEngineOrchestratorEntry = path.resolve(
  simAppDir,
  '../../packages/cogent-engine/dist/esm/orchestrator/index.js'
);

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      'cogent-engine/orchestrator': cogentEngineOrchestratorEntry,
      'cogent-engine/character': cogentEngineCharacterEntry,
      'cogent-engine': cogentEngineEntry,
    },
    preserveSymlinks: true,
  },
  optimizeDeps: {
    exclude: ['cogent-engine'],
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
