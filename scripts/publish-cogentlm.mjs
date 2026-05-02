import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '..');
const packageDir = path.join(repoRoot, 'packages', 'cogentlm');
const packageJsonPath = path.join(packageDir, 'package.json');
const projectNpmrcPath = path.join(repoRoot, '.npmrc');
const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const bunCommand = process.versions.bun ? process.execPath : 'bun';
const registryHost = 'npm.pkg.github.com';
const registryUrl = `https://${registryHost}`;
const expectedPackageName = '@noumena-labs/cogentlm';
const supportedFlags = new Set(['--dry-run', '--help']);
const rawFlags = process.argv.slice(2);

for (const flag of rawFlags) {
  if (!supportedFlags.has(flag)) {
    throw new Error(`Unsupported flag: ${flag}`);
  }
}

if (rawFlags.includes('--help')) {
  console.log(`Usage: bun ./scripts/publish-cogentlm.mjs [--dry-run]\n\nRuns release:prepare for packages/cogentlm, validates npm auth configuration, and publishes with npm publish.`);
  process.exit(0);
}

const dryRun = rawFlags.includes('--dry-run');
const packageJson = JSON.parse(readFileSync(packageJsonPath, 'utf8'));

function run(command, args, cwd, label) {
  const result = spawnSync(command, args, {
    cwd,
    stdio: 'inherit',
    env: process.env
  });

  if (result.status !== 0) {
    throw new Error(`${label} failed with exit code ${result.status ?? 'unknown'}.`);
  }
}

function readNpmrcCandidates() {
  const candidatePaths = [
    projectNpmrcPath,
    process.env.NPM_CONFIG_USERCONFIG,
    process.env.npm_config_userconfig,
    path.join(os.homedir(), '.npmrc')
  ].filter(Boolean);

  return candidatePaths.filter((candidatePath, index) => candidatePaths.indexOf(candidatePath) === index);
}

function npmrcHasUsableAuth(filePath) {
  if (!existsSync(filePath)) {
    return false;
  }

  return readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .some((line) => {
      const trimmedLine = line.trim();
      if (!trimmedLine.startsWith(`//${registryHost}/:_authToken=`)) {
        return false;
      }

      const tokenValue = trimmedLine.slice(`//${registryHost}/:_authToken=`.length).trim();
      if (!tokenValue) {
        return false;
      }

      const envReference = tokenValue.match(/^\$\{(.+)\}$/);
      if (envReference) {
        return Boolean(process.env[envReference[1]]?.trim());
      }

      return true;
    });
}

function ensurePublishConfig() {
  if (packageJson.name !== expectedPackageName) {
    throw new Error(`Expected ${packageJsonPath} to declare ${expectedPackageName}, found ${packageJson.name}.`);
  }
}

function ensureAuthConfigured() {
  const hasNodeAuthToken = Boolean(process.env.NODE_AUTH_TOKEN?.trim());
  if (hasNodeAuthToken) {
    return;
  }

  const hasUserNpmLogin = readNpmrcCandidates().some((candidatePath) => npmrcHasUsableAuth(candidatePath));
  if (hasUserNpmLogin) {
    return;
  }

  throw new Error(
    `No npm credentials found for ${registryUrl}. Set NODE_AUTH_TOKEN or run npm login --registry=${registryUrl}.`
  );
}

ensurePublishConfig();
ensureAuthConfigured();

console.log(`[publish-cogentlm] preparing ${packageJson.name}@${packageJson.version}`);
run(bunCommand, ['run', 'release:prepare'], packageDir, 'release preparation');

const publishArgs = ['publish', '--registry', registryUrl];
if (dryRun) {
  publishArgs.push('--dry-run');
}

console.log(
  `[publish-cogentlm] ${dryRun ? 'running dry-run publish' : 'publishing'} to ${registryUrl}`
);
run(npmCommand, publishArgs, packageDir, 'npm publish');
