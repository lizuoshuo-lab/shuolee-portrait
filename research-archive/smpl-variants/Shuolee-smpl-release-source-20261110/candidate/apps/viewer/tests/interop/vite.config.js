import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import { readFileSync } from 'node:fs';

export default defineConfig({
  root: fileURLToPath(new URL('../../', import.meta.url)),
  base: './',
  server: { host: '127.0.0.1', port: 4873, strictPort: true },
  plugins: [{
    name: 'interop-fixture-and-license-notices',
    generateBundle() {
      for (const file of ['anisotropic-sh3.ply', 'anisotropic-sh3.json', 'THIRD_PARTY_NOTICES.md']) {
        this.emitFile({ type: 'asset', fileName: `tests/interop/${file}`,
          source: readFileSync(new URL(`./${file}`, import.meta.url)) });
      }
    },
  }],
  build: {
    target: 'es2022', outDir: 'tests/interop/dist', emptyOutDir: true,
    rollupOptions: {
      input: ['index.html', 'mkkellogg.html', 'playcanvas.html'].map((file) =>
        fileURLToPath(new URL(`./${file}`, import.meta.url))),
    },
  },
});
