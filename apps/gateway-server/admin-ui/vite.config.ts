import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const gatewayProxy = process.env.SIPP_GATEWAY_ADMIN_PROXY ?? 'http://127.0.0.1:9090';

export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/admin/api': gatewayProxy,
    },
  },
});
