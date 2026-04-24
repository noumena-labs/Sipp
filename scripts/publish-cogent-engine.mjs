import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '..');
const packageDir = path.join(repoRoot, 'packages', 'cogent-engine');
const packageJsonPath = path.join(packageDir, 'package.json');
const projectNpmrcPath = path.join(repoRoot, '.npmrc');
const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const bunCommand = process.versions.bun ? process.execPath : 'bun';
const registryHost = 'npm.pkg.github.com';
const registryUrl = `https://${registryHost}`;
const expectedScope = '@noumena-labs';
const expectedPackageName = '@noumena-labs/cogent-engine';
const supportedFlags = new Set(['--dry-run', '--help']);
const rawFlags = process.argv.slice(2);

for (const flag of rawFlags) {
  if (!supportedFlags.has(flag)) {
    throw new Error(`Unsupported flag: ${flag}`);
  }
}

if (rawFlags.includes('--help')) {
  console.log(`Usage: bun ./scripts/publish-cogent-engine.mjs [--dry-run]\n\nRuns release:prepare for packages/cogent-engine, validates GitHub Packages auth configuration, and publishes with npm publish.`);
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

function npmrcContainsScopeRegistry(filePath) {
  if (!existsSync(filePath)) {
    return false;
  }

  return readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .some((line) => line.trim() === `${expectedScope}:registry=${registryUrl}`);
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

  if (packageJson.publishConfig?.registry !== registryUrl) {
    throw new Error(`Expected publishConfig.registry to be ${registryUrl}.`);
  }

  const hasScopeMapping = readNpmrcCandidates().some((candidatePath) => npmrcContainsScopeRegistry(candidatePath));
  if (!hasScopeMapping) {
    throw new Error(
      `Missing ${expectedScope}:registry=${registryUrl} in project or user npm config. Add it to ${projectNpmrcPath}.`
    );
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
    `No GitHub Packages credentials found for ${registryUrl}. Set NODE_AUTH_TOKEN or run npm login --scope=${expectedScope} --registry=${registryUrl}.`
  );
}

ensurePublishConfig();
ensureAuthConfigured();

console.log(`[publish-cogent-engine] preparing ${packageJson.name}@${packageJson.version}`);
run(bunCommand, ['run', 'release:prepare'], packageDir, 'release preparation');

const publishArgs = ['publish', '--registry', registryUrl];
if (dryRun) {
  publishArgs.push('--dry-run');
}

console.log(
  `[publish-cogent-engine] ${dryRun ? 'running dry-run publish' : 'publishing'} to ${registryUrl}`
);
run(npmCommand, publishArgs, packageDir, 'npm publish');