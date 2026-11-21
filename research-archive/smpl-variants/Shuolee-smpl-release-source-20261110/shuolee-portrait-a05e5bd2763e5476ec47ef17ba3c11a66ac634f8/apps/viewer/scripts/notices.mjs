import { readFile, mkdir, writeFile } from 'node:fs/promises';
const root = new URL('../', import.meta.url);
const names = ['three', '@sparkjsdev/spark'];
const parts = ['Shuolee viewer: third-party license notices\nDependencies are bundled locally; no CDN is needed to open a model.\n'];
for (const name of names) {
  const dir = new URL(`node_modules/${name}/`, root);
  const pkg = JSON.parse(await readFile(new URL('package.json', dir), 'utf8'));
  parts.push(`\n===== ${name} ${pkg.version} =====\n${await readFile(new URL('LICENSE', dir), 'utf8')}`);
}
await mkdir(new URL('public/docs/', root), { recursive: true });
await writeFile(new URL('public/docs/THIRD_PARTY_NOTICES.txt', root), parts.join('\n'));
