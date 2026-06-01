import { access, copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const packageDir = fileURLToPath(new URL('..', import.meta.url));
const repoRoot = path.resolve(packageDir, '..', '..');
const stageDir = path.join(repoRoot, '.build', 'artifacts', 'npm', 'cogentlm-browser');
const packageJsonPath = path.join(packageDir, 'package.json');

const packageJson = JSON.parse(await readFile(packageJsonPath, 'utf8'));
packageJson.files = ['dist', 'README.md'];
packageJson.repository = {
  ...packageJson.repository,
  directory: 'packages/npm',
};

await mkdir(stageDir, { recursive: true });
await writeFile(
  path.join(stageDir, 'package.json'),
  `${JSON.stringify(packageJson, null, 2)}\n`,
);

const readmePath = path.join(packageDir, 'README.md');
try {
  await access(readmePath);
  await copyFile(readmePath, path.join(stageDir, 'README.md'));
} catch {
  // README is optional in this workspace package.
}
