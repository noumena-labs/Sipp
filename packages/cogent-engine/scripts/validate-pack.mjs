import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const packageDir = path.resolve(scriptDir, '..');
const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const requiredPackPaths = [
  'dist/esm/index.js',
  'dist/esm/runtime-assets.js',
  'dist/esm/character/index.js',
  'dist/esm/orchestrator/index.js',
  'dist/types/index.d.ts',
  'dist/types/runtime-assets.d.ts',
  'dist/types/character/index.d.ts',
  'dist/types/orchestrator/index.d.ts',
  'dist/wasm/cogent-engine-wasm.js',
  'dist/wasm/cogent-engine-wasm.wasm'
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

console.log(
  `[pack:validate] ${packEntry.filename} includes ${requiredPackPaths.length} required release artifacts.`
);