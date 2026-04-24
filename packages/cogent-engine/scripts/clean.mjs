import { rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');

const cleanTargets = [
  path.join(projectRoot, 'build'),
  path.join(projectRoot, 'build-wasm-dev'),
  path.join(projectRoot, 'build-bun-mem32'),
  path.join(projectRoot, 'dist', 'esm'),
  path.join(projectRoot, 'dist', 'types'),
  path.join(projectRoot, 'dist', 'wasm'),
  path.join(projectRoot, 'dist', 'wasm-bun'),
  path.join(projectRoot, 'a.out.js'),
  path.join(projectRoot, 'a.out.wasm')
];

for (const targetPath of cleanTargets) {
  await rm(targetPath, { recursive: true, force: true });
  console.log(`[clean] removed ${targetPath}`);
}
