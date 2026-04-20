import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Tutte le chiamate a /api/* vengono inoltrate a http://localhost:8000
      // In questo modo dal frontend scrivi fetch("/api/pipeline/generate")
      // e Vite le inoltra al backend FastAPI, bypassando CORS in sviluppo.
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})