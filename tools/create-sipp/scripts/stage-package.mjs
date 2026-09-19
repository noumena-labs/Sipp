import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const packageDir = fileURLToPath(new URL('..', import.meta.url));
const repoRoot = path.resolve(packageDir, '..', '..');
const stageDir = path.join(repoRoot, '.build', 'artifacts', 'npm', 'create-sipp');

await rm(stageDir, { recursive: true, force: true });
await mkdir(stageDir, { recursive: true });

const packageJson = JSON.parse(
  await readFile(path.join(packageDir, 'package.json'), 'utf8'),
);
const packageFiles = packageJson.files;
if (
  !Array.isArray(packageFiles) ||
  packageFiles.some((entry) => typeof entry !== 'string')
) {
  throw new Error('create-sipp package.json must declare a string files array.');
}
delete packageJson.scripts;

await writeFile(
  path.join(stageDir, 'package.json'),
  `${JSON.stringify(packageJson, null, 2)}\n`,
);
await Promise.all(
  packageFiles.map((entry) =>
    cp(path.join(packageDir, entry), path.join(stageDir, entry), {
      recursive: true,
    }),
  ),
);
