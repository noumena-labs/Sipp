import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { sippClientDistWatch } from '../sipp-dist-watch';

const proactiveUiAppDir = fileURLToPath(new URL('.', import.meta.url));
const sippClientDistDir = path.resolve(
  proactiveUiAppDir,
  '../../.build/artifacts/npm/sipp/dist/esm'
);
const sippClientEntry = path.join(sippClientDistDir, 'index.js');
const appOutDir = path.resolve(proactiveUiAppDir, '../../.build/artifacts/demos/proactive-ui');

export default defineConfig({
  plugins: [react(), sippClientDistWatch()],
  build: {
    outDir: appOutDir,
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      '@noumena-labs/sipp': sippClientEntry,
    },
    preserveSymlinks: true,
  },
  optimizeDeps: {
    exclude: ['@noumena-labs/sipp'],
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
