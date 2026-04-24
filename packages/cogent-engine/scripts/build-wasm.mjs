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
const buildLabel = process.env.CE_WASM_BUILD_LABEL?.trim() || '[build-wasm]';
const buildDirName = process.env.CE_WASM_BUILD_DIR_NAME?.trim() || 'build';
const artifactPrefix = 'cogent-engine-wasm';
const buildDir = path.join(projectRoot, buildDirName);
const buildDistDir = path.join(buildDir, 'dist');
const packageWasmSubdir = process.env.CE_WASM_OUTPUT_SUBDIR?.trim() || 'wasm';
const packageWasmDir = path.join(projectRoot, 'dist', packageWasmSubdir);
const enableDebug = false;
const enableEsModule = true;
const enableFilesystem = true;
const enableJspi = true;
const enableAggressiveOpt = true;
const enablePthreads = false;
const suppressLlamaLogs = true;
const emscriptenEnvironment = 'web,worker';
const enableMemory64 = readBooleanEnv('CE_WASM_MEM64', true);
const ltoMode = process.env.CE_WASM_LTO_MODE?.trim().toUpperCase() || (isWindows ? 'THIN' : 'FULL');
const initialMemory = process.env.CE_WASM_INITIAL_MEMORY?.trim() || '512MB';
const maximumMemory =
  process.env.CE_WASM_MAXIMUM_MEMORY?.trim() || (enableMemory64 ? '16384MB' : '4096MB');
const stackSize = process.env.CE_WASM_STACK_SIZE?.trim() || '16MB';
const buildParallelLevel = process.env.CE_WASM_BUILD_PARALLEL_LEVEL?.trim() || process.env.CMAKE_BUILD_PARALLEL_LEVEL?.trim();

let activeChildProcess = null;
let signalHandlersInstalled = false;
let activeMakeProgramDir = null;
let cachedCmakeExecutable = null;

function readBooleanEnv(name, fallback = false) {
  const rawValue = process.env[name]?.trim().toLowerCase();
  if (!rawValue) {
    return fallback;
  }

  if (['1', 'true', 'yes', 'on'].includes(rawValue)) {
    return true;
  }

  if (['0', 'false', 'no', 'off'].includes(rawValue)) {
    return false;
  }

  throw new Error(`Invalid boolean value for ${name}: ${process.env[name]}`);
}

function parseMemorySizeBytes(rawValue) {
  const normalizedValue = rawValue.trim().toUpperCase();
  const match = normalizedValue.match(/^(\d+)(B|KB|MB|GB|TB)?$/);
  if (!match) {
    throw new Error(`Invalid memory size "${rawValue}". Use values like 4096MB or 4GB.`);
  }

  const amount = Number.parseInt(match[1], 10);
  const unit = match[2] ?? 'B';
  const multiplierByUnit = {
    B: 1,
    KB: 1024,
    MB: 1024 ** 2,
    GB: 1024 ** 3,
    TB: 1024 ** 4,
  };

  return amount * multiplierByUnit[unit];
}

function validateMaximumMemorySetting(rawValue) {
  if (enableMemory64) {
    return;
  }

  const maximumAllowedBytes = 4 * 1024 ** 3;
  const configuredBytes = parseMemorySizeBytes(rawValue);
  if (configuredBytes > maximumAllowedBytes) {
    throw new Error(
      `CE_WASM_MAXIMUM_MEMORY=${rawValue} exceeds the wasm32 limit of 4GB. ` +
        'Use 4096MB or enable CE_WASM_MEM64.'
    );
  }
}

function validateLtoMode(rawValue) {
  if (!['OFF', 'THIN', 'FULL'].includes(rawValue)) {
    throw new Error(`Invalid CE_WASM_LTO_MODE=${rawValue}. Use OFF, THIN, or FULL.`);
  }
}

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

function shouldRunConfigure(expectedGenerator) {
  const cachePath = path.join(buildDir, 'CMakeCache.txt');
  if (!existsSync(cachePath)) {
    return true;
  }

  const cacheText = readFileSync(cachePath, 'utf8');
  const cachedGenerator = getCacheEntry(cacheText, 'CMAKE_GENERATOR');
  const cachedBuildType = getCacheEntry(cacheText, 'CMAKE_BUILD_TYPE');
  const cachedDebug = getCacheEntry(cacheText, 'CE_WASM_DEBUG');
  const cachedEsModule = getCacheEntry(cacheText, 'CE_WASM_ES_MODULE');
  const cachedAggressiveOpt = getCacheEntry(cacheText, 'CE_WASM_AGGRESSIVE_OPT');
  const cachedSuppressLlamaLogs = getCacheEntry(cacheText, 'CE_SUPPRESS_LLAMA_LOGS');
  const cachedPthreads = getCacheEntry(cacheText, 'CE_WASM_PTHREADS');
  const cachedJspi = getCacheEntry(cacheText, 'CE_WASM_USE_JSPI');
  const cachedEnvironment = getCacheEntry(cacheText, 'CE_WASM_ENVIRONMENT');
  const cachedFilesystem = getCacheEntry(cacheText, 'CE_WASM_FILESYSTEM');
  const cachedMemory64 = getCacheEntry(cacheText, 'CE_WASM_MEM64') ?? getCacheEntry(cacheText, 'LLAMA_WASM_MEM64');
  const cachedInitialMemory = getCacheEntry(cacheText, 'CE_WASM_INITIAL_MEMORY');
  const cachedMaximumMemory = getCacheEntry(cacheText, 'CE_WASM_MAXIMUM_MEMORY');
  const cachedStackSize = getCacheEntry(cacheText, 'CE_WASM_STACK_SIZE');
  const cachedLtoMode = getCacheEntry(cacheText, 'CE_WASM_LTO_MODE');

  if (expectedGenerator && cachedGenerator && cachedGenerator !== expectedGenerator) {
    return true;
  }

  if (cachedBuildType && cachedBuildType !== 'Release') {
    return true;
  }

  if (cachedDebug && cachedDebug !== (enableDebug ? 'ON' : 'OFF')) {
    return true;
  }

  if (cachedEsModule && cachedEsModule !== (enableEsModule ? 'ON' : 'OFF')) {
    return true;
  }

  if (cachedAggressiveOpt && cachedAggressiveOpt !== (enableAggressiveOpt ? 'ON' : 'OFF')) {
    return true;
  }

  if (cachedSuppressLlamaLogs && cachedSuppressLlamaLogs !== (suppressLlamaLogs ? 'ON' : 'OFF')) {
    return true;
  }

  if (cachedPthreads && cachedPthreads !== (enablePthreads ? 'ON' : 'OFF')) {
    return true;
  }

  if (cachedJspi && cachedJspi !== (enableJspi ? 'ON' : 'OFF')) {
    return true;
  }

  if (cachedEnvironment && cachedEnvironment !== emscriptenEnvironment) {
    return true;
  }

  if (cachedFilesystem && cachedFilesystem !== (enableFilesystem ? 'ON' : 'OFF')) {
    return true;
  }

  if (cachedMemory64 && cachedMemory64 !== (enableMemory64 ? 'ON' : 'OFF')) {
    return true;
  }

  if (cachedInitialMemory && cachedInitialMemory !== initialMemory) {
    return true;
  }

  if (cachedMaximumMemory && cachedMaximumMemory !== maximumMemory) {
    return true;
  }

  if (cachedStackSize && cachedStackSize !== stackSize) {
    return true;
  }

  if (cachedLtoMode && cachedLtoMode.toUpperCase() !== ltoMode) {
    return true;
  }

  return false;
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

  const cachedMemory64 = getCacheEntry(cacheText, 'CE_WASM_MEM64') ?? getCacheEntry(cacheText, 'LLAMA_WASM_MEM64');
  if (cachedMemory64 && (cachedMemory64 === 'ON') !== enableMemory64) {
    reasons.push(`LLAMA_WASM_MEM64=${cachedMemory64}`);
  }

  const cachedMaximumMemory = getCacheEntry(cacheText, 'CE_WASM_MAXIMUM_MEMORY');
  if (cachedMaximumMemory && cachedMaximumMemory !== maximumMemory) {
    reasons.push(`CE_WASM_MAXIMUM_MEMORY=${cachedMaximumMemory}`);
  }

  const cachedLtoMode = getCacheEntry(cacheText, 'CE_WASM_LTO_MODE');
  if (cachedLtoMode && cachedLtoMode.toUpperCase() !== ltoMode) {
    reasons.push(`CE_WASM_LTO_MODE=${cachedLtoMode}`);
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
validateMaximumMemorySetting(maximumMemory);
validateLtoMode(ltoMode);
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
  `-DCE_WASM_DEBUG=${enableDebug ? 'ON' : 'OFF'}`,
  `-DCE_WASM_ES_MODULE=${enableEsModule ? 'ON' : 'OFF'}`,
  `-DCE_WASM_FILESYSTEM=${enableFilesystem ? 'ON' : 'OFF'}`,
  `-DCE_WASM_AGGRESSIVE_OPT=${enableAggressiveOpt ? 'ON' : 'OFF'}`,
  `-DCE_SUPPRESS_LLAMA_LOGS=${suppressLlamaLogs ? 'ON' : 'OFF'}`,
  `-DCE_WASM_PTHREADS=${enablePthreads ? 'ON' : 'OFF'}`,
  `-DCE_WASM_USE_JSPI=${enableJspi ? 'ON' : 'OFF'}`,
  `-DCE_WASM_ENVIRONMENT=${emscriptenEnvironment}`,
  `-DCE_WASM_MEM64=${enableMemory64 ? 'ON' : 'OFF'}`,
  `-DCE_WASM_LTO_MODE=${ltoMode}`,
  `-DCE_WASM_INITIAL_MEMORY=${initialMemory}`,
  `-DCE_WASM_STACK_SIZE=${stackSize}`,
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

if (maximumMemory) {
  cmakeConfigureArgs.push(`-DCE_WASM_MAXIMUM_MEMORY=${maximumMemory}`);
}

console.log(
  `${buildLabel} generator=${buildConfig.generator} mem64=${enableMemory64 ? 'on' : 'off'} lto=${ltoMode.toLowerCase()} output=dist/${packageWasmSubdir}` +
    (buildConfig.makeProgram ? ` make_program=${buildConfig.makeProgram}` : '') +
    (buildParallelLevel ? ` jobs=${buildParallelLevel}` : '')
);

if (shouldRunConfigure(buildConfig.generator)) {
  await runCommand(cmakeExecutable, cmakeConfigureArgs);
} else {
  console.log(`${buildLabel} reusing existing CMake configure state`);
}

const cmakeBuildArgs = ['--build', buildDirName, '--config', 'Release', '--target', wasmTargetName];
if (buildParallelLevel) {
  cmakeBuildArgs.push('--parallel', buildParallelLevel);
}

await runCommand(cmakeExecutable, cmakeBuildArgs);
await copyWasmArtifacts();
