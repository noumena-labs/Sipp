import { spawn, spawnSync } from 'node:child_process';
import { access, copyFile, mkdir, readdir, rm } from 'node:fs/promises';
import { existsSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const wasmTargetName = 'CogentEngine';
const isWindows = process.platform === 'win32';
const supportedGenerators = new Set(['Ninja', 'NMake Makefiles', 'Unix Makefiles']);
const buildLabel = '[build-wasm]';
const buildDirName = 'build';
const artifactPrefix = 'cogent-engine-wasm';
const buildDir = path.join(projectRoot, buildDirName);
const buildDistDir = path.join(buildDir, 'dist');
const packageWasmDir = path.join(projectRoot, 'dist', 'wasm');
const enableJspi = true;
const emscriptenEnvironment = 'web,worker';

let activeChildProcess = null;
let signalHandlersInstalled = false;
let activeMakeProgramDir = null;
let cachedCmakeExecutable = null;

function normalizeHostPath(inputPath) {
  if (!inputPath || !isWindows) {
    return inputPath;
  }

  return path.win32.normalize(inputPath.replace(/\//g, '\\'));
}

function findProgramsOnPath(command) {
  const locator = isWindows ? 'where.exe' : 'which';
  const result = spawnSync(locator, [command], {
    cwd: projectRoot,
    stdio: ['ignore', 'pipe', 'ignore'],
    shell: false,
    windowsHide: true,
    encoding: 'utf8'
  });

  if (result.error || result.status !== 0 || !result.stdout) {
    return [];
  }

  return result.stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function commandAvailable(command, args = ['--version']) {
  const result = spawnSync(command, args, {
    cwd: projectRoot,
    stdio: 'ignore',
    shell: false,
    windowsHide: true
  });

  return !result.error && result.status === 0;
}

// Resolve cmake once up front. On Windows, Node's raw spawn lookup can be less
// reliable than resolving the executable path explicitly.
function resolveCmakeExecutable() {
  if (cachedCmakeExecutable) {
    return cachedCmakeExecutable;
  }

  const candidates = [
    ...findProgramsOnPath(isWindows ? 'cmake.exe' : 'cmake'),
    isWindows
      ? path.join(process.env.ProgramFiles ?? 'C:\\Program Files', 'CMake', 'bin', 'cmake.exe')
      : null
  ]
    .map((candidate) => normalizeHostPath(candidate))
    .filter(Boolean);

  const cmakeExecutable = candidates.find((candidate) => existsSync(candidate));
  if (!cmakeExecutable) {
    throw new Error('CMake executable not found. Install CMake or add it to PATH.');
  }

  cachedCmakeExecutable = cmakeExecutable;
  return cmakeExecutable;
}

function validateGenerator(generatorName) {
  if (!supportedGenerators.has(generatorName)) {
    throw new Error(
      `Unsupported CMAKE_GENERATOR "${generatorName}". Supported generators: ${Array.from(supportedGenerators).join(', ')}.`
    );
  }
}

function inferGeneratorFromMakeProgram(makeProgramPath) {
  const lowerPath = makeProgramPath.toLowerCase();

  if (lowerPath.includes('ninja')) {
    return 'Ninja';
  }

  if (lowerPath.includes('nmake')) {
    return 'NMake Makefiles';
  }

  if (lowerPath.endsWith('make') || lowerPath.endsWith('make.exe')) {
    return 'Unix Makefiles';
  }

  return null;
}

function collectEmsdkRoots() {
  const roots = [];

  if (process.env.EMSDK?.trim()) {
    roots.push(process.env.EMSDK.trim());
  }

  if (process.env.USERPROFILE?.trim()) {
    roots.push(path.join(process.env.USERPROFILE, 'emsdk'));
    roots.push(path.join(process.env.USERPROFILE, 'Documents', 'emsdk'));
  }

  if (process.env.HOME?.trim()) {
    roots.push(path.join(process.env.HOME, 'emsdk'));
  }

  if (isWindows) {
    roots.push('C:\\emsdk', 'D:\\emsdk');
  }

  return roots
    .map((candidate) => normalizeHostPath(candidate))
    .filter((candidate, index, allCandidates) => candidate && allCandidates.indexOf(candidate) === index)
    .filter((candidate) => existsSync(candidate));
}

function resolveEmsdkRoot() {
  const envRoot = normalizeHostPath(process.env.EMSDK?.trim());
  if (envRoot && existsSync(envRoot)) {
    return envRoot;
  }

  return collectEmsdkRoots()[0] ?? null;
}

// Match the legacy standalone build by reusing the cached Dawn WebGPU port when it exists.
function resolveEmdawnwebgpuDir(emsdkRoot) {
  const candidates = [process.env.EMDAWNWEBGPU_DIR?.trim()];

  if (emsdkRoot) {
    candidates.push(
      path.join(emsdkRoot, 'upstream', 'emscripten', 'cache', 'ports', 'emdawnwebgpu', 'emdawnwebgpu_pkg')
    );
  }

  for (const candidate of candidates) {
    const portDir = normalizeHostPath(candidate);
    if (portDir && existsSync(path.join(portDir, 'emdawnwebgpu.port.py'))) {
      return portDir;
    }
  }

  return null;
}

function detectBundledNinjaMakeProgram() {
  const ninjaExecutableName = isWindows ? 'ninja.exe' : 'ninja';

  for (const emsdkRoot of collectEmsdkRoots()) {
    const ninjaRoot = path.join(emsdkRoot, 'ninja');
    if (!existsSync(ninjaRoot)) {
      continue;
    }

    const versionDirectories = readdirSync(ninjaRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .sort((left, right) => right.localeCompare(left, undefined, { numeric: true }));

    for (const versionDirectory of versionDirectories) {
      const makeProgram = path.join(ninjaRoot, versionDirectory, ninjaExecutableName);
      if (existsSync(makeProgram)) {
        return normalizeHostPath(makeProgram);
      }
    }
  }

  return null;
}

function detectNinjaFromCmakeInstall() {
  const ninjaExecutableName = isWindows ? 'ninja.exe' : 'ninja';

  for (const cmakePath of findProgramsOnPath(isWindows ? 'cmake.exe' : 'cmake')) {
    const ninjaPath = path.join(path.dirname(cmakePath), ninjaExecutableName);
    if (existsSync(ninjaPath)) {
      return normalizeHostPath(ninjaPath);
    }
  }

  return null;
}

function detectNinjaFromVisualStudio() {
  if (!isWindows) {
    return null;
  }

  const visualStudioRoot = path.join(process.env.ProgramFiles ?? 'C:\\Program Files', 'Microsoft Visual Studio');
  if (!existsSync(visualStudioRoot)) {
    return null;
  }

  for (const year of ['2026', '2022', '2019', '2017']) {
    const yearDir = path.join(visualStudioRoot, year);
    if (!existsSync(yearDir)) {
      continue;
    }

    const editions = readdirSync(yearDir, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name);

    for (const edition of editions) {
      const ninjaPath = path.join(
        yearDir,
        edition,
        'Common7',
        'IDE',
        'CommonExtensions',
        'Microsoft',
        'CMake',
        'Ninja',
        'ninja.exe'
      );

      if (existsSync(ninjaPath)) {
        return normalizeHostPath(ninjaPath);
      }
    }
  }

  return null;
}

// Prefer explicit environment configuration first, then fall back to the common
// Ninja locations used by EMSDK, CMake, and Visual Studio.
function resolveBuildConfiguration() {
  const generatorFromEnv = process.env.CMAKE_GENERATOR?.trim();
  const makeProgramFromEnv = normalizeHostPath(process.env.CMAKE_MAKE_PROGRAM?.trim());

  if (generatorFromEnv) {
    validateGenerator(generatorFromEnv);
    return { generator: generatorFromEnv, makeProgram: makeProgramFromEnv || null };
  }

  if (makeProgramFromEnv) {
    const inferredGenerator = inferGeneratorFromMakeProgram(makeProgramFromEnv);
    if (!inferredGenerator) {
      throw new Error(
        'CMAKE_MAKE_PROGRAM is set but CMAKE_GENERATOR is missing and could not be inferred. Set both explicitly.'
      );
    }

    validateGenerator(inferredGenerator);
    return { generator: inferredGenerator, makeProgram: makeProgramFromEnv };
  }

  const detectedNinja =
    detectBundledNinjaMakeProgram() ?? detectNinjaFromCmakeInstall() ?? detectNinjaFromVisualStudio();

  if (detectedNinja) {
    return { generator: 'Ninja', makeProgram: detectedNinja };
  }

  if (commandAvailable('ninja')) {
    return { generator: 'Ninja', makeProgram: null };
  }

  if (isWindows && commandAvailable('nmake', ['/?'])) {
    return { generator: 'NMake Makefiles', makeProgram: null };
  }

  if (commandAvailable('make')) {
    return { generator: 'Unix Makefiles', makeProgram: null };
  }

  throw new Error(
    'No supported CMake generator found. Install Ninja, ensure EMSDK or Visual Studio Ninja is available, or set CMAKE_GENERATOR/CMAKE_MAKE_PROGRAM explicitly.'
  );
}

function getCacheEntry(cacheText, key) {
  const match = cacheText.match(new RegExp(`^${key}:[^=]*=(.*)$`, 'm'));
  return match ? match[1].trim() : null;
}

function hasIncompleteBuildDirectory() {
  if (!existsSync(buildDir)) {
    return false;
  }

  if (existsSync(path.join(buildDir, 'CMakeCache.txt'))) {
    return false;
  }

  return existsSync(path.join(buildDir, 'CMakeFiles')) || existsSync(path.join(buildDir, 'build.ninja'));
}

// Interrupted configure runs leave behind a partial build directory. Reusing it on the
// next run causes misleading follow-up failures, so start from a clean build directory.
function removeInvalidBuildDirectory(expectedGenerator) {
  if (hasIncompleteBuildDirectory()) {
    console.log(`${buildLabel} removing incomplete build directory`);
    rmSync(buildDir, { recursive: true, force: true });
    return;
  }

  const cachePath = path.join(buildDir, 'CMakeCache.txt');
  if (!existsSync(cachePath)) {
    return;
  }

  const cacheText = readFileSync(cachePath, 'utf8');
  const cachedGenerator = getCacheEntry(cacheText, 'CMAKE_GENERATOR');
  const reasons = [];

  if (cacheText.includes('CMAKE_MAKE_PROGRAM:FILEPATH=CMAKE_MAKE_PROGRAM-NOTFOUND')) {
    reasons.push('CMAKE_MAKE_PROGRAM-NOTFOUND');
  }

  if (cacheText.includes('LLAMA_WASM_MEM64:BOOL=ON')) {
    reasons.push('LLAMA_WASM_MEM64=ON');
  }

  if (expectedGenerator && cachedGenerator && cachedGenerator !== expectedGenerator) {
    reasons.push(`generator=${cachedGenerator}`);
  }

  if (reasons.length > 0) {
    console.log(`${buildLabel} removing stale build directory (${reasons.join(', ')})`);
    rmSync(buildDir, { recursive: true, force: true });
  }
}

function prependPathEntry(env, pathEntry) {
  if (!pathEntry) {
    return;
  }

  const pathKey = Object.keys(env).find((key) => key.toLowerCase() === 'path') ?? 'PATH';
  const delimiter = isWindows ? ';' : ':';
  env[pathKey] = env[pathKey] ? `${pathEntry}${delimiter}${env[pathKey]}` : pathEntry;
}

function terminateProcessTree(pid) {
  if (!pid) {
    return;
  }

  // On Windows, killing the direct child is not enough. CMake can leave Ninja alive
  // unless the full process tree is terminated.
  if (isWindows) {
    spawnSync('taskkill.exe', ['/T', '/F', '/PID', String(pid)], {
      stdio: 'ignore',
      shell: false,
      windowsHide: true
    });
    return;
  }

  try {
    process.kill(-pid, 'SIGTERM');
  } catch {}

  try {
    process.kill(pid, 'SIGTERM');
  } catch {}
}

function installSignalHandlers() {
  if (signalHandlersInstalled) {
    return;
  }

  const exitForSignal = (signal) => {
    if (activeChildProcess?.pid) {
      terminateProcessTree(activeChildProcess.pid);
    }

    process.exit(signal === 'SIGINT' ? 130 : 143);
  };

  process.on('SIGINT', () => exitForSignal('SIGINT'));
  process.on('SIGTERM', () => exitForSignal('SIGTERM'));
  signalHandlersInstalled = true;
}

// Only one build subprocess runs at a time, so keep the lifecycle logic local and
// forward interrupts to the child process tree instead of leaving stale workers.
async function runCommand(executable, args) {
  const env = { ...process.env };
  prependPathEntry(env, activeMakeProgramDir);

  console.log(`${buildLabel} run: ${executable} ${args.join(' ')}`);

  installSignalHandlers();

  const childProcess = spawn(executable, args, {
    cwd: projectRoot,
    stdio: 'inherit',
    shell: false,
    windowsHide: true,
    env,
    detached: !isWindows
  });

  activeChildProcess = childProcess;

  try {
    await new Promise((resolve, reject) => {
      childProcess.once('error', reject);
      childProcess.once('exit', (code, signal) => {
        if (signal) {
          reject(new Error(`Command terminated by signal ${signal}: ${executable} ${args.join(' ')}`));
          return;
        }

        if (code !== 0) {
          reject(new Error(`Command failed: ${executable} ${args.join(' ')}`));
          return;
        }

        resolve();
      });
    });
  } catch (error) {
    if (error instanceof Error && error.message.startsWith('Command failed')) {
      throw error;
    }

    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Command failed to start: ${executable} ${args.join(' ')}\n${message}`);
  } finally {
    if (activeChildProcess === childProcess) {
      activeChildProcess = null;
    }
  }
}

// Package consumers import the renamed wasm artifacts from dist/wasm rather than
// the raw CMake output names produced in the build directory.
async function copyWasmArtifacts() {
  try {
    await access(buildDistDir);
  } catch {
    throw new Error(`Missing build output directory: ${buildDistDir}`);
  }

  await rm(packageWasmDir, { recursive: true, force: true });
  await mkdir(packageWasmDir, { recursive: true });

  for (const artifactName of await readdir(buildDistDir)) {
    if (!artifactName.startsWith('CogentEngine')) {
      continue;
    }

    const sourcePath = path.join(buildDistDir, artifactName);
    const targetPath = path.join(
      packageWasmDir,
      `${artifactPrefix}${artifactName.slice('CogentEngine'.length)}`
    );

    await copyFile(sourcePath, targetPath);
  }
}

const buildConfig = resolveBuildConfiguration();
removeInvalidBuildDirectory(buildConfig.generator);

activeMakeProgramDir = buildConfig.makeProgram ? path.dirname(buildConfig.makeProgram) : null;

const cmakeExecutable = resolveCmakeExecutable();
const toolchainPath = normalizeHostPath(path.resolve(projectRoot, 'cmake', 'toolchains', 'EmscriptenAuto.cmake'));
const cmakeConfigureArgs = [
  '-S',
  '.',
  '-B',
  buildDirName,
  '-G',
  buildConfig.generator,
  '-DCMAKE_BUILD_TYPE=Release',
  `-DCMAKE_TOOLCHAIN_FILE=${toolchainPath}`,
  '-DCE_WASM_ES_MODULE=ON',
  '-DCE_WASM_AGGRESSIVE_OPT=ON',
  `-DCE_WASM_USE_JSPI=${enableJspi ? 'ON' : 'OFF'}`,
  `-DCE_WASM_ENVIRONMENT=${emscriptenEnvironment}`,
  '-DLLAMA_WASM_MEM64=OFF',
  '-DLLAMA_BUILD_HTML=OFF'
];

const emsdkRoot = resolveEmsdkRoot();
if (emsdkRoot) {
  cmakeConfigureArgs.push(`-DEMSDK=${emsdkRoot}`);
}

const emdawnwebgpuDir = resolveEmdawnwebgpuDir(emsdkRoot);
if (emdawnwebgpuDir) {
  cmakeConfigureArgs.push(`-DEMDAWNWEBGPU_DIR=${emdawnwebgpuDir}`);
}

console.log(
  `${buildLabel} generator=${buildConfig.generator}` +
    (buildConfig.makeProgram ? ` make_program=${buildConfig.makeProgram}` : '')
);

await runCommand(cmakeExecutable, cmakeConfigureArgs);
await runCommand(cmakeExecutable, ['--build', buildDirName, '--config', 'Release', '--target', wasmTargetName]);
await copyWasmArtifacts();
