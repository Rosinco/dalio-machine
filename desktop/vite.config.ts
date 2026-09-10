import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: './',
  clearScreen: false,
  server: { port: 1420, strictPort: true },
  build: { chunkSizeWarningLimit: 1800, target: 'es2022', rollupOptions: { output: {
    manualChunks(id) {
      if (id.includes('/maplibre-gl/')) return 'map';
      if (id.includes('/echarts/') || id.includes('/zrender/')) return 'charts';
    },
  } } },
});
