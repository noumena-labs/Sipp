import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { sippViteConfig } from '../../lib/web/src/vite.js';

const examplesDir = fileURLToPath(new URL('.', import.meta.url));
const sippDistDir = path.resolve(
  examplesDir,
  '../../.build/artifacts/npm/sipp/dist/esm'
);
const sippEntry = path.join(sippDistDir, 'index.js');
const appOutDir = path.resolve(examplesDir, '../../.build/artifacts/demos/chat');

export default defineConfig({
  ...sippViteConfig(),
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
  optimizeDeps: {
    exclude: ['@noumena-labs/sipp'],
  },
});
