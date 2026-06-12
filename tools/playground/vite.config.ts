import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { sippClientDistWatch } from '../../demos/sipp-dist-watch';

const playgroundAppDir = fileURLToPath(new URL('.', import.meta.url));
const sippClientDistDir = path.resolve(
  playgroundAppDir,
  '../../.build/artifacts/npm/sipp/dist/esm'
);
const sippClientEntry = path.join(sippClientDistDir, 'index.js');
const appOutDir = path.resolve(playgroundAppDir, '../../.build/artifacts/tools/playground');

export default defineConfig({
  plugins: [react(), sippClientDistWatch()],
  build: {
    outDir: appOutDir,
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      // Use the built workspace entry directly so Vite does not serve the package
      // through an immutable /node_modules dependency URL that can stay stale
      // across local package rebuilds.
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
