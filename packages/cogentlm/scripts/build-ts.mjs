import { spawn } from 'node:child_process';
import { copyFile, lstat, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
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
const packageRootEnvVar = 'COGENTLM_PACKAGE_ROOT';
const internalPackageRoot = 'node_modules/@noumena-labs/cogentlm';
const publicPackageRoot = 'node_modules/cogentlm';
const supportedPackageRoots = new Set([internalPackageRoot, publicPackageRoot]);

function getPackageRoot() {
  const packageRoot = process.env[packageRootEnvVar]?.trim() || internalPackageRoot;

  if (!supportedPackageRoots.has(packageRoot)) {
    throw new Error(
      `${packageRootEnvVar} must be one of: ${Array.from(supportedPackageRoots).join(', ')}`
    );
  }

  return packageRoot;
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

async function syncTypeScriptDist() {
  await mirrorDirectory(tempEsmDir, esmDir);
  await mirrorDirectory(tempTypesDir, typesDir);
}

async function preserveBundlerDirectives() {
  const runtimePath = path.join(tempEsmDir, 'runtime', 'engine-runtime-main-thread.js');
  const runtimeText = await readFile(runtimePath, 'utf8');
  // Stack ignore comments so every major bundler skips static analysis of the
  // dynamic Emscripten module URL:
  //   - @vite-ignore       -> Vite / Rollup
  //   - webpackIgnore      -> webpack (>=2)
  //   - turbopackIgnore    -> Turbopack (Next.js)
  // esbuild, Bun, and native ESM ignore unknown comments and pass through.
  const patchedRuntimeText = runtimeText.replace(
    'import(moduleUrl)',
    'import(/* @vite-ignore */ /* webpackIgnore: true */ /* turbopackIgnore: true */ moduleUrl)'
  );

  if (patchedRuntimeText === runtimeText) {
    throw new Error(
      'Could not preserve bundler ignore directives in engine-runtime-main-thread.js.'
    );
  }

  await writeFile(runtimePath, patchedRuntimeText);
}

async function applyPackageRootOverride() {
  const packageRoot = getPackageRoot();

  if (packageRoot === internalPackageRoot) {
    return;
  }

  const packageAssetsPath = path.join(tempEsmDir, 'runtime', 'package-assets.js');
  const packageAssetsText = await readFile(packageAssetsPath, 'utf8');
  const internalPackageRootStatement = `const PACKAGE_ROOT = '${internalPackageRoot}';`;

  if (!packageAssetsText.includes(internalPackageRootStatement)) {
    throw new Error('Could not find the default PACKAGE_ROOT statement in runtime/package-assets.js.');
  }

  await writeFile(
    packageAssetsPath,
    packageAssetsText.replace(
      internalPackageRootStatement,
      `const PACKAGE_ROOT = '${packageRoot}';`
    )
  );

  console.log(`[build-ts] package asset root: ${packageRoot}`);
}

await rm(tempRoot, { recursive: true, force: true });
await mkdir(tempRoot, { recursive: true });

try {
  await runTypeScriptBuild();
  await preserveBundlerDirectives();
  await applyPackageRootOverride();
  await syncTypeScriptDist();
  console.log('[build-ts] updated dist/esm and dist/types');
} finally {
  await rm(tempRoot, { recursive: true, force: true });
}
