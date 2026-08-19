import { defineConfig } from 'vite';
import typegpu from 'unplugin-typegpu/vite';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  // `unplugin-typegpu` lifts the AST of every function tagged with the `'use gpu'`
  // directive into `tinyest` metadata at build time. Without it, TypeGPU cannot
  // transpile TypeScript kernels to WGSL — every compute kernel in this project
  // depends on it.
  plugins: [typegpu({ include: [/\.ts$/] })],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  build: {
    target: 'esnext',
    sourcemap: true,
  },
  server: { port: 5173 },
});
