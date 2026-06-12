import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const examplesDir = fileURLToPath(new URL('.', import.meta.url));
const sippDistDir = path.resolve(
  examplesDir,
  '../../.build/artifacts/npm/sipp/dist/esm'
);
const sippEntry = path.join(sippDistDir, 'index.js');
const appOutDir = path.resolve(examplesDir, '../../.build/artifacts/demos/chat');

export default defineConfig({
  build: {
    outDir: appOutDir,
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      '@noumena-labs/sipp': sippEntry,
    },
    preserveSymlinks: true,
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  // The chat demo forces wasmThreading: 'pthread', which requires
  // cross-origin isolation; without preview headers `vite preview` serves a
  // page where client construction fails outright.
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: {
    exclude: ['@noumena-labs/sipp'],
  },
});
