import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const packageDir = path.resolve(scriptDir, '..');
const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const requiredPackPaths = [
  'dist/esm/index.js',
  'dist/esm/runtime/package-assets.js',
  'dist/esm/runtime-assets.js',
  'dist/esm/worker/model-service-client.js',
  'dist/esm/worker/model-service-entry.js',
  'dist/esm/character/index.js',
  'dist/esm/orchestrator/index.js',
  'dist/types/index.d.ts',
  'dist/types/runtime/package-assets.d.ts',
  'dist/types/runtime-assets.d.ts',
  'dist/types/character/index.d.ts',
  'dist/types/orchestrator/index.d.ts',
  'dist/wasm/cogentlm-wasm.js',
  'dist/wasm/cogentlm-wasm.wasm'
];

function fail(message) {
  console.error(`[pack:validate] ${message}`);
  process.exit(1);
}

const packResult = spawnSync(npmCommand, ['pack', '--dry-run', '--json'], {
  cwd: packageDir,
  encoding: 'utf8',
  stdio: ['ignore', 'pipe', 'pipe']
});

if (packResult.status !== 0) {
  fail(packResult.stderr.trim() || 'npm pack --dry-run failed.');
}

let packEntries;

try {
  packEntries = JSON.parse(packResult.stdout);
} catch (error) {
  fail(`Could not parse npm pack output as JSON.\n${packResult.stdout.trim()}`);
}

if (!Array.isArray(packEntries) || packEntries.length === 0) {
  fail('npm pack --dry-run returned no tarball metadata.');
}

const [packEntry] = packEntries;
const packedPaths = new Set((packEntry.files ?? []).map((file) => file.path));
const missingPaths = requiredPackPaths.filter((requiredPath) => !packedPaths.has(requiredPath));

if (missingPaths.length > 0) {
  fail(`Missing required release artifacts in tarball:\n- ${missingPaths.join('\n- ')}`);
}

const workerClientPath = path.join(packageDir, 'dist', 'esm', 'worker', 'model-service-client.js');
const workerEntryPath = path.join(packageDir, 'dist', 'esm', 'worker', 'model-service-entry.js');
const runtimePath = path.join(packageDir, 'dist', 'esm', 'runtime', 'engine-runtime-main-thread.js');
const workerClientText = readFileSync(workerClientPath, 'utf8');
const workerEntryText = readFileSync(workerEntryPath, 'utf8');
const runtimeText = readFileSync(runtimePath, 'utf8');

if (!workerClientText.includes("new Worker(new URL('./model-service-entry.js', import.meta.url)")) {
  fail(
    'Default worker construction must use new Worker(new URL(..., import.meta.url), ...) so bundlers include the worker graph.'
  );
}

if (!workerClientText.includes("resolveOptimizedPackageAssetUrl('dist/esm/worker/model-service-entry.js'")) {
  fail(
    'Worker client must preserve Vite optimized-deps fallback for package-hosted worker assets.'
  );
}

if (workerEntryText.includes('../../wasm/cogentlm-wasm')) {
  fail(
    'Worker entry must not hard-code ../../wasm runtime asset URLs; use getDefaultRuntimeUrls() so bundlers rewrite assets consistently.'
  );
}

if (!runtimeText.includes('import(/* @vite-ignore */ moduleUrl)')) {
  fail(
    'Runtime module import must preserve /* @vite-ignore */ in dist so Vite does not try to statically analyze user-configurable moduleUrl.'
  );
}

console.log(
  `[pack:validate] ${packEntry.filename} includes ${requiredPackPaths.length} required release artifacts.`
);
