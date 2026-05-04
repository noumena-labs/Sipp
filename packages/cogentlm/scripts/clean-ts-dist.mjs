import { rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');

const targets = [
  path.join(projectRoot, 'dist', 'esm'),
  path.join(projectRoot, 'dist', 'types'),
];

for (const target of targets) {
  await rm(target, { recursive: true, force: true });
  console.log(`[clean-ts-dist] removed ${target}`);
}
