import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { cogentClientDistWatch } from '../cogentlm-dist-watch';

const simAppDir = fileURLToPath(new URL('.', import.meta.url));
const cogentClientDistDir = path.resolve(
  simAppDir,
  '../../.build/artifacts/npm/cogentlm-browser/dist/esm'
);
const cogentClientEntry = path.join(cogentClientDistDir, 'index.js');
const cogentClientCharacterEntry = path.join(cogentClientDistDir, 'character/index.js');
const cogentClientDirectorEntry = path.join(cogentClientDistDir, 'orchestrator/index.js');
const appOutDir = path.resolve(simAppDir, '../../.build/artifacts/apps/simulation');

export default defineConfig({
  plugins: [react(), cogentClientDistWatch()],
  build: {
    outDir: appOutDir,
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      '@noumena-labs/cogentlm-browser/director': cogentClientDirectorEntry,
      '@noumena-labs/cogentlm-browser/character': cogentClientCharacterEntry,
      '@noumena-labs/cogentlm-browser': cogentClientEntry,
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
