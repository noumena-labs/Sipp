import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { cogentClientDistWatch } from '../../demos/cogentlm-dist-watch';

const playgroundAppDir = fileURLToPath(new URL('.', import.meta.url));
const cogentClientDistDir = path.resolve(
  playgroundAppDir,
  '../../.build/artifacts/npm/cogentlm/dist/esm'
);
const cogentClientEntry = path.join(cogentClientDistDir, 'index.js');
const appOutDir = path.resolve(playgroundAppDir, '../../.build/artifacts/tools/playground');

export default defineConfig({
  plugins: [react(), cogentClientDistWatch()],
  build: {
    outDir: appOutDir,
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      // Use the built workspace entry directly so Vite does not serve the package
      // through an immutable /node_modules dependency URL that can stay stale
      // across local package rebuilds.
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
