import { access, copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const packageDir = fileURLToPath(new URL('..', import.meta.url));
const repoRoot = path.resolve(packageDir, '..', '..');
const stageDir = path.join(repoRoot, '.build', 'artifacts', 'npm', 'sipp');
const packageJsonPath = path.join(packageDir, 'package.json');

const packageJson = JSON.parse(await readFile(packageJsonPath, 'utf8'));
packageJson.files = ['dist', 'LICENSE', 'README.md', 'THIRD_PARTY_NOTICES.md'];
packageJson.repository = {
  ...packageJson.repository,
  directory: 'lib/web',
};

await mkdir(stageDir, { recursive: true });
await writeFile(
  path.join(stageDir, 'package.json'),
  `${JSON.stringify(packageJson, null, 2)}\n`,
);

const packageReadmePath = path.join(packageDir, 'README.md');
const rootReadmePath = path.join(repoRoot, 'README.md');
await copyFile(path.join(repoRoot, 'LICENSE'), path.join(stageDir, 'LICENSE'));
await copyFile(
  path.join(repoRoot, 'THIRD_PARTY_NOTICES.md'),
  path.join(stageDir, 'THIRD_PARTY_NOTICES.md'),
);
try {
  await access(packageReadmePath);
  await copyFile(packageReadmePath, path.join(stageDir, 'README.md'));
} catch {
  await copyFile(rootReadmePath, path.join(stageDir, 'README.md'));
}
