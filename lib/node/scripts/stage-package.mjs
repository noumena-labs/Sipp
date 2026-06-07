import { copyFile, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const packageDir = fileURLToPath(new URL('..', import.meta.url));
const repoRoot = path.resolve(packageDir, '..', '..');
const sourceArtifactDir = path.join(repoRoot, '.build', 'artifacts', 'node');
const stageDir = path.join(repoRoot, '.build', 'artifacts', 'npm', 'cogentlm-server');
const nativeDir = path.join(stageDir, 'native');
const packageJsonPath = path.join(packageDir, 'package.json');

const packageJson = JSON.parse(await readFile(packageJsonPath, 'utf8'));
packageJson.files = [
  'index.d.ts',
  'LICENSE',
  'native',
  'README.md',
  'router.d.ts',
  'router.js',
];
packageJson.repository = {
  ...packageJson.repository,
  directory: 'lib/node',
};

await rm(stageDir, { recursive: true, force: true });
await mkdir(nativeDir, { recursive: true });
await writeFile(
  path.join(stageDir, 'package.json'),
  `${JSON.stringify(packageJson, null, 2)}\n`
);

for (const fileName of ['index.d.ts', 'router.d.ts', 'router.js']) {
  await copyFile(path.join(packageDir, fileName), path.join(stageDir, fileName));
}
await copyFile(path.join(packageDir, 'README.md'), path.join(stageDir, 'README.md'));
await copyFile(path.join(repoRoot, 'LICENSE'), path.join(stageDir, 'LICENSE'));

const nativeArtifacts = (await readdir(sourceArtifactDir))
  .filter((fileName) => fileName.startsWith('cogentlm_node_') && fileName.endsWith('.node'))
  .sort();

if (nativeArtifacts.length === 0) {
  throw new Error(`No Node native artifacts found in ${sourceArtifactDir}`);
}

for (const fileName of nativeArtifacts) {
  await copyFile(path.join(sourceArtifactDir, fileName), path.join(nativeDir, fileName));
}
