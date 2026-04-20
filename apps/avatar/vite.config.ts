import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const avatarAppDir = fileURLToPath(new URL('.', import.meta.url));
const cogentEngineEntry = path.resolve(
  avatarAppDir,
  '../../packages/cogent-engine/dist/esm/index.js'
);
const cogentEngineCharacterEntry = path.resolve(
  avatarAppDir,
  '../../packages/cogent-engine/dist/esm/character/index.js'
);

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Resolve both the root package entry and the ./character subpath
      // directly at the built ESM files so we pick up local rebuilds without
      // going through a cached /node_modules dependency URL.
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
