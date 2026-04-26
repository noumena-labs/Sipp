import { spawn } from 'node:child_process';
import { copyFile, lstat, mkdir, readdir, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const distDir = path.join(projectRoot, 'dist');
const tempRoot = path.join(distDir, '.ts-build');
const tempEsmDir = path.join(tempRoot, 'esm');
const tempTypesDir = path.join(tempRoot, 'types');
const esmDir = path.join(distDir, 'esm');
const typesDir = path.join(distDir, 'types');
const backupEsmDir = path.join(distDir, '.previous-esm');
const backupTypesDir = path.join(distDir, '.previous-types');

async function pathExists(targetPath) {
  try {
    await lstat(targetPath);
    return true;
  } catch (error) {
    if (error?.code === 'ENOENT') {
      return false;
    }

    throw error;
  }
}

async function pathIsDirectory(targetPath) {
  try {
    return (await lstat(targetPath)).isDirectory();
  } catch (error) {
    if (error?.code === 'ENOENT') {
      return false;
    }

    throw error;
  }
}

function runTypeScriptBuild() {
  return new Promise((resolve, reject) => {
    const childProcess = spawn(
      'bunx',
      [
        '--bun',
        'tsc',
        '-p',
        'tsconfig.json',
        '--outDir',
        path.relative(projectRoot, tempEsmDir),
        '--declarationDir',
        path.relative(projectRoot, tempTypesDir),
      ],
      {
        cwd: projectRoot,
        stdio: 'inherit',
        shell: false,
        windowsHide: true,
      }
    );

    childProcess.once('error', reject);
    childProcess.once('exit', (code, signal) => {
      if (signal) {
        reject(new Error(`TypeScript build terminated by ${signal}`));
        return;
      }

      if (code !== 0) {
        reject(new Error(`TypeScript build failed with exit code ${code}`));
        return;
      }

      resolve();
    });
  });
}

async function mirrorDirectory(sourceDir, targetDir) {
  await mkdir(targetDir, { recursive: true });

  const sourceEntries = await readdir(sourceDir, { withFileTypes: true });
  const sourceNames = new Set(sourceEntries.map((entry) => entry.name));

  for (const entry of sourceEntries) {
    const sourcePath = path.join(sourceDir, entry.name);
    const targetPath = path.join(targetDir, entry.name);

    if (entry.isDirectory()) {
      if (!(await pathIsDirectory(targetPath))) {
        await rm(targetPath, { recursive: true, force: true });
      }

      await mirrorDirectory(sourcePath, targetPath);
      continue;
    }

    if (await pathIsDirectory(targetPath)) {
      await rm(targetPath, { recursive: true, force: true });
    }

    await copyFile(sourcePath, targetPath);
  }

  const targetEntries = await readdir(targetDir, { withFileTypes: true });
  for (const entry of targetEntries) {
    if (!sourceNames.has(entry.name)) {
      await rm(path.join(targetDir, entry.name), { recursive: true, force: true });
    }
  }
}

async function restoreInterruptedSwap() {
  if ((await pathExists(backupEsmDir)) && !(await pathExists(esmDir))) {
    await mirrorDirectory(backupEsmDir, esmDir);
  }

  if ((await pathExists(backupTypesDir)) && !(await pathExists(typesDir))) {
    await mirrorDirectory(backupTypesDir, typesDir);
  }
}

async function syncTypeScriptDist() {
  await mirrorDirectory(tempEsmDir, esmDir);
  await mirrorDirectory(tempTypesDir, typesDir);
  await rm(backupEsmDir, { recursive: true, force: true });
  await rm(backupTypesDir, { recursive: true, force: true });
}

await restoreInterruptedSwap();
await rm(tempRoot, { recursive: true, force: true });
await mkdir(tempRoot, { recursive: true });

try {
  await runTypeScriptBuild();
  await syncTypeScriptDist();
  console.log('[build-ts] updated dist/esm and dist/types');
} finally {
  await rm(tempRoot, { recursive: true, force: true });
}
