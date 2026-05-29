import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { cogentEngineDistWatch } from '../cogentlm-dist-watch';

const avatarAppDir = fileURLToPath(new URL('.', import.meta.url));
const cogentEngineDistDir = path.resolve(
  avatarAppDir,
  '../../.build/artifacts/npm/cogentlm-browser/dist/esm'
);
const cogentEngineEntry = path.join(cogentEngineDistDir, 'index.js');
const cogentEngineCharacterEntry = path.join(cogentEngineDistDir, 'character/index.js');
const appOutDir = path.resolve(avatarAppDir, '../../.build/artifacts/apps/avatar');

export default defineConfig({
  plugins: [react(), cogentEngineDistWatch()],
  build: {
    outDir: appOutDir,
    emptyOutDir: true,
  },
  resolve: {
    alias: {
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
