import { defineConfig } from 'vitest/config';
import typegpu from 'unplugin-typegpu/vite';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  plugins: [typegpu({ include: [/\.ts$/] })],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  test: {
    globals: true,
    environment: 'node',
    // Dawn's native addon is not safe to instantiate many times in parallel,
    // and each instance holds a software Vulkan context.
    pool: 'forks',
    poolOptions: { forks: { singleFork: true } },
    testTimeout: 600_000,
    hookTimeout: 600_000,
    include: ['test/**/*.test.ts'],
  },
});
