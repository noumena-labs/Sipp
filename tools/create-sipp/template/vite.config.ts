import { defineConfig } from 'vite';
import { sippViteConfig } from '@sipphq/sipp/vite';

export default defineConfig({
  ...sippViteConfig(),
  optimizeDeps: {
    exclude: ['@sipphq/sipp'],
  },
});
