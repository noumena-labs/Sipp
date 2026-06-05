import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const exampleDir = fileURLToPath(new URL('.', import.meta.url));
const cogentlmDistDir = path.resolve(
  exampleDir,
  '../../.build/artifacts/npm/cogentlm/dist/esm',
);
const cogentlmEntry = path.join(cogentlmDistDir, 'index.js');
const pageEntries = {
  index: path.resolve(exampleDir, 'index.html'),
  query: path.resolve(exampleDir, 'query.html'),
  chat: path.resolve(exampleDir, 'chat.html'),
  embed: path.resolve(exampleDir, 'embed.html'),
  remoteGatewayQuery: path.resolve(exampleDir, 'remote_gateway_query.html'),
  remoteGatewayChat: path.resolve(exampleDir, 'remote_gateway_chat.html'),
  remoteGatewayEmbed: path.resolve(exampleDir, 'remote_gateway_embed.html'),
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
      '@noumena-labs/cogentlm': cogentlmEntry,
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
    exclude: ['@noumena-labs/cogentlm'],
  },
});
