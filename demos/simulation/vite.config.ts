import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { sippViteConfig } from '../../lib/web/src/vite.js';
import { sippClientDistWatch } from '../sipp-dist-watch';

const simAppDir = fileURLToPath(new URL('.', import.meta.url));
const sippClientDistDir = path.resolve(
  simAppDir,
  '../../.build/artifacts/npm/sipp/dist/esm'
);
const sippClientEntry = path.join(sippClientDistDir, 'index.js');
const sippClientCharacterEntry = path.join(sippClientDistDir, 'character/index.js');
const sippClientDirectorEntry = path.join(sippClientDistDir, 'orchestrator/index.js');
const appOutDir = path.resolve(simAppDir, '../../.build/artifacts/demos/simulation');

export default defineConfig({
  ...sippViteConfig(),
  plugins: [react(), sippClientDistWatch()],
  build: {
    outDir: appOutDir,
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      '@noumena-labs/sipp/director': sippClientDirectorEntry,
      '@noumena-labs/sipp/character': sippClientCharacterEntry,
      '@noumena-labs/sipp': sippClientEntry,
    },
    preserveSymlinks: true,
  },
  optimizeDeps: {
    exclude: ['@noumena-labs/sipp'],
  },
});
