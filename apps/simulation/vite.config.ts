import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { cogentEngineDistWatch } from '../cogentlm-dist-watch';

const simAppDir = fileURLToPath(new URL('.', import.meta.url));
const cogentEngineEntry = path.resolve(
  simAppDir,
  '../../packages/npm/dist/esm/index.js'
);
const cogentEngineCharacterEntry = path.resolve(
  simAppDir,
  '../../packages/npm/dist/esm/character/index.js'
);
const cogentEngineDirectorEntry = path.resolve(
  simAppDir,
  '../../packages/npm/dist/esm/orchestrator/index.js'
);

export default defineConfig({
  plugins: [react(), cogentEngineDistWatch()],
  resolve: {
    alias: {
      '@noumena-labs/cogentlm-browser/director': cogentEngineDirectorEntry,
      '@noumena-labs/cogentlm-browser/character': cogentEngineCharacterEntry,
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
