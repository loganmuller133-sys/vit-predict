import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    cssMinify: false
  },
  server: {
    port: 5173,
    proxy: {
      '/predict': 'http://localhost:8000',
      '/history': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/fetch': 'http://localhost:8000',
      '/odds': 'http://localhost:8000'
    }
  }
})
