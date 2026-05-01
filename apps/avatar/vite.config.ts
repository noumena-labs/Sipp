import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { cogentEngineDistWatch } from '../cogentlm-dist-watch';

const avatarAppDir = fileURLToPath(new URL('.', import.meta.url));
const cogentEngineEntry = path.resolve(
  avatarAppDir,
  '../../packages/cogentlm/dist/esm/index.js'
);
const cogentEngineCharacterEntry = path.resolve(
  avatarAppDir,
  '../../packages/cogentlm/dist/esm/character/index.js'
);

export default defineConfig({
  plugins: [react(), cogentEngineDistWatch()],
  resolve: {
    alias: {
      // Resolve both the root package entry and the ./character subpath
      // directly at the built ESM files so we pick up local rebuilds without
      // going through a cached /node_modules dependency URL.
      'cogentlm/character': cogentEngineCharacterEntry,
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
