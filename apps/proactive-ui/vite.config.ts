import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { cogentEngineDistWatch } from '../cogentlm-dist-watch';

const proactiveUiAppDir = fileURLToPath(new URL('.', import.meta.url));
const cogentEngineDistDir = path.resolve(
  proactiveUiAppDir,
  '../../.build/artifacts/npm/cogentlm-browser/dist/esm'
);
const cogentEngineEntry = path.join(cogentEngineDistDir, 'index.js');
const appOutDir = path.resolve(proactiveUiAppDir, '../../.build/artifacts/apps/proactive-ui');

export default defineConfig({
  plugins: [react(), cogentEngineDistWatch()],
  build: {
    outDir: appOutDir,
    emptyOutDir: true,
  },
  resolve: {
    alias: {
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
