import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { cogentClientDistWatch } from '../cogentlm-dist-watch';

const avatarAppDir = fileURLToPath(new URL('.', import.meta.url));
const cogentClientDistDir = path.resolve(
  avatarAppDir,
  '../../.build/artifacts/npm/cogentlm/dist/esm'
);
const cogentClientEntry = path.join(cogentClientDistDir, 'index.js');
const cogentClientCharacterEntry = path.join(cogentClientDistDir, 'character/index.js');
const appOutDir = path.resolve(avatarAppDir, '../../.build/artifacts/apps/avatar');

export default defineConfig({
  plugins: [react(), cogentClientDistWatch()],
  build: {
    outDir: appOutDir,
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      '@noumena-labs/cogentlm/character': cogentClientCharacterEntry,
      '@noumena-labs/cogentlm': cogentClientEntry,
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
