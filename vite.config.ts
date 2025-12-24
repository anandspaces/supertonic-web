import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(),tailwindcss()],
  server: {
    port: 4020
  }
  // assetsInclude: ['**/*.wasm', '**/*.onnx'],
  // optimizeDeps: {
  //   exclude: ['onnxruntime-web']
  // },
  // server: {
  //   headers: {
  //     'Cross-Origin-Embedder-Policy': 'require-corp',
  //     'Cross-Origin-Opener-Policy': 'same-origin'
  //   },
  //   fs: {
  //     // Allow serving files from onnxruntime-web
  //     allow: ['..']
  //   }
  // },
  // build: {
  //   rollupOptions: {
  //     external: []
  //   }
  // }
})
