import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { cogentEngineDistWatch } from '../cogentlm-dist-watch';

const simAppDir = fileURLToPath(new URL('.', import.meta.url));
const cogentEngineEntry = path.resolve(
  simAppDir,
  '../../packages/cogentlm/dist/esm/index.js'
);
const cogentEngineCharacterEntry = path.resolve(
  simAppDir,
  '../../packages/cogentlm/dist/esm/character/index.js'
);
const cogentEngineDirectorEntry = path.resolve(
  simAppDir,
  '../../packages/cogentlm/dist/esm/orchestrator/index.js'
);

export default defineConfig({
  plugins: [react(), cogentEngineDistWatch()],
  resolve: {
    alias: {
      '@noumena-labs/cogentlm/director': cogentEngineDirectorEntry,
      '@noumena-labs/cogentlm/character': cogentEngineCharacterEntry,
      'cogentlm/director': cogentEngineDirectorEntry,
      'cogentlm/character': cogentEngineCharacterEntry,
      '@noumena-labs/cogentlm': cogentEngineEntry,
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
