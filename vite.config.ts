import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
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
