import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const exampleDir = fileURLToPath(new URL('.', import.meta.url));
const sippDistDir = path.resolve(
  exampleDir,
  '../../.build/artifacts/npm/sipp/dist/esm',
);
const sippEntry = path.join(sippDistDir, 'index.js');
const pageEntries = {
  index: path.resolve(exampleDir, 'index.html'),
  query: path.resolve(exampleDir, 'query.html'),
  chat: path.resolve(exampleDir, 'chat.html'),
  embed: path.resolve(exampleDir, 'embed.html'),
  gatewayLocal: path.resolve(exampleDir, 'gateway_local.html'),
  gatewayQuery: path.resolve(exampleDir, 'gateway_query.html'),
  gatewayChat: path.resolve(exampleDir, 'gateway_chat.html'),
  gatewayEmbed: path.resolve(exampleDir, 'gateway_embed.html'),
};

export default defineConfig({
  build: {
    outDir: path.resolve(exampleDir, '../../.build/artifacts/examples/web'),
    emptyOutDir: true,
    rollupOptions: {
      input: pageEntries,
    },
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
