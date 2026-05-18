import { spawn, spawnSync } from 'node:child_process';
import { access, copyFile, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { existsSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Parse arguments to configure default environment variables
const isDev = process.argv.includes('--dev');
if (isDev) {
  process.env.CE_WASM_BUILD_LABEL ??= '[build-wasm:dev]';
  process.env.CE_WASM_BUILD_DIR_NAME ??= 'build-wasm-dev';
  process.env.CE_WASM_LTO_MODE ??= 'OFF';
  process.env.CE_WASM_BUILD_PARALLEL_LEVEL ??= '8';
} else {
  // Default unified browser release settings
  process.env.CE_WASM_BUILD_LABEL ??= '[build-wasm:browser]';
  process.env.CE_WASM_BUILD_DIR_NAME ??= 'build-browser';
  process.env.CE_WASM_OUTPUT_SUBDIR ??= 'wasm';
  process.env.CE_WASM_MEM64 ??= '0';
  process.env.CE_WASM_MAXIMUM_MEMORY ??= '4096MB';
  process.env.CE_WASM_PTHREADS ??= '0';
}

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const wasmTargetName = 'CogentLM';
const isWindows = process.platform === 'win32';
const supportedGenerators = new Set(['Ninja', 'NMake Makefiles', 'Unix Makefiles']);
const buildLabel = process.env.CE_WASM_BUILD_LABEL?.trim() || '[build-wasm]';
const buildDirName = process.env.CE_WASM_BUILD_DIR_NAME?.trim() || 'build';
const artifactPrefix = 'cogentlm-wasm';
const buildDir = path.join(projectRoot, buildDirName);
const buildDistDir = path.join(buildDir, 'dist');
const packageWasmSubdir = process.env.CE_WASM_OUTPUT_SUBDIR?.trim() || 'wasm';
const packageWasmDir = path.join(projectRoot, 'dist', packageWasmSubdir);
const llamaCppRoot = path.join(projectRoot, '..', 'third_party', 'llama.cpp');
const enableDebug = false;
const enableFilesystem = true;
const enableJspi = true;
const enableAggressiveOpt = true;
const enablePthreads = readBooleanEnv('CE_WASM_PTHREADS', false);
const suppressLlamaLogs = true;
const suppressMtmdLogs = true;
const emscriptenEnvironment = 'web,worker';
const enableMemory64 = readBooleanEnv('CE_WASM_MEM64', false);
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
let rustBrowserStaticlib = null;

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

function rustBuildEnv() {
  if (!enablePthreads) {
    return process.env;
  }

  return {
    ...process.env,
    RUSTFLAGS: [
      process.env.RUSTFLAGS,
      '-C',
      'target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128,+nontrapping-fptoint',
    ]
      .filter(Boolean)
      .join(' '),
  };
}

function rustCargoBuildPrefix() {
  return enablePthreads
    ? ['+nightly', 'build', '-Zbuild-std=std,panic_unwind']
    : ['build'];
}

function buildRustBrowserStaticlib() {
  if (enableMemory64) {
    throw new Error('Browser engine requires CE_WASM_MEM64=0.');
  }

  const rustRoot = path.resolve(projectRoot, '..', 'cogentlm-rs');
  const manifestPath = path.join(rustRoot, 'Cargo.toml');
  if (!existsSync(manifestPath)) {
    throw new Error(`Missing Rust workspace for browser engine: ${manifestPath}`);
  }

  console.log(`${buildLabel} building Rust browser staticlib`);
  const result = spawnSync(
    'cargo',
    [
      ...rustCargoBuildPrefix(),
      '-p',
      'cogentlm-browser',
      '--target',
      'wasm32-unknown-emscripten',
      '--release',
    ],
    {
      cwd: rustRoot,
      stdio: 'inherit',
      shell: false,
      windowsHide: true,
      env: rustBuildEnv(),
    }
  );

  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`Rust browser staticlib build failed with exit code ${result.status}`);
  }

  const staticlibPath = path.join(
    rustRoot,
    'target',
    'wasm32-unknown-emscripten',
    'release',
    'libcogentlm_browser.a'
  );
  if (!existsSync(staticlibPath)) {
    throw new Error(`Rust browser staticlib was not produced: ${staticlibPath}`);
  }

  return normalizeCmakePath(staticlibPath);
}

function normalizeHostPath(inputPath) {
  if (!inputPath || !isWindows) {
    return inputPath;
  }

  return path.win32.normalize(inputPath.replace(/\//g, '\\'));
}

function normalizeCmakePath(inputPath) {
  return inputPath?.replace(/\\/g, '/') ?? inputPath;
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
  const cachedAggressiveOpt = getCacheEntry(cacheText, 'CE_WASM_AGGRESSIVE_OPT');
  const cachedSuppressLlamaLogs = getCacheEntry(cacheText, 'CE_SUPPRESS_LLAMA_LOGS');
  const cachedSuppressMtmdLogs = getCacheEntry(cacheText, 'MTMD_NO_LOGGING');
  const cachedPthreads = getCacheEntry(cacheText, 'CE_WASM_PTHREADS');
  const cachedJspi = getCacheEntry(cacheText, 'CE_WASM_USE_JSPI');
  const cachedEnvironment = getCacheEntry(cacheText, 'CE_WASM_ENVIRONMENT');
  const cachedFilesystem = getCacheEntry(cacheText, 'CE_WASM_FILESYSTEM');
  const cachedMemory64 = getCacheEntry(cacheText, 'CE_WASM_MEM64') ?? getCacheEntry(cacheText, 'LLAMA_WASM_MEM64');
  const cachedInitialMemory = getCacheEntry(cacheText, 'CE_WASM_INITIAL_MEMORY');
  const cachedMaximumMemory = getCacheEntry(cacheText, 'CE_WASM_MAXIMUM_MEMORY');
  const cachedStackSize = getCacheEntry(cacheText, 'CE_WASM_STACK_SIZE');
  const cachedLtoMode = getCacheEntry(cacheText, 'CE_WASM_LTO_MODE');
  const cachedRustBrowserLib = getCacheEntry(cacheText, 'CE_WASM_RUST_BROWSER_LIB') ?? '';
  const expectedRustBrowserLib = rustBrowserStaticlib ?? '';

  if (expectedGenerator && cachedGenerator && cachedGenerator !== expectedGenerator) {
    return true;
  }

  if (cachedBuildType && cachedBuildType !== 'Release') {
    return true;
  }

  if (cachedDebug && cachedDebug !== (enableDebug ? 'ON' : 'OFF')) {
    return true;
  }

  if (cachedAggressiveOpt && cachedAggressiveOpt !== (enableAggressiveOpt ? 'ON' : 'OFF')) {
    return true;
  }

  if (cachedSuppressLlamaLogs && cachedSuppressLlamaLogs !== (suppressLlamaLogs ? 'ON' : 'OFF')) {
    return true;
  }

  if (cachedSuppressMtmdLogs && cachedSuppressMtmdLogs !== (suppressMtmdLogs ? 'ON' : 'OFF')) {
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

  if (cachedRustBrowserLib !== expectedRustBrowserLib) {
    return true;
  }

  return false;
}

function hasIncompleteBuildDirectory() {
  if (!existsSync(buildDir)) {
    return false;
  }

  const cachePath = path.join(buildDir, 'CMakeCache.txt');
  if (existsSync(cachePath)) {
    return false;
  }

  return (
    existsSync(path.join(buildDir, 'CMakeFiles')) ||
    existsSync(path.join(buildDir, 'build.ninja')) ||
    existsSync(path.join(buildDir, 'Makefile'))
  );
}

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

  const cachedRustBrowserLib = getCacheEntry(cacheText, 'CE_WASM_RUST_BROWSER_LIB') ?? '';
  const expectedRustBrowserLib = rustBrowserStaticlib ?? '';
  if (cachedRustBrowserLib !== expectedRustBrowserLib) {
    reasons.push(`CE_WASM_RUST_BROWSER_LIB=${cachedRustBrowserLib || 'OFF'}`);
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
    process.kill(pid, 'SIGKILL');
  } catch {}
}

function installSignalHandlers() {
  if (signalHandlersInstalled) {
    return;
  }

  const handleInterrupt = () => {
    if (activeChildProcess) {
      console.log(`${buildLabel} forwarding SIGINT to build process tree`);
      terminateProcessTree(activeChildProcess.pid);
    }
    process.exit(130);
  };

  process.on('SIGINT', handleInterrupt);
  process.on('SIGTERM', handleInterrupt);
  signalHandlersInstalled = true;
}

function runCommand(command, args) {
  installSignalHandlers();

  const makeProgramEnv = {};
  if (activeMakeProgramDir) {
    prependPathEntry(makeProgramEnv, activeMakeProgramDir);
  }

  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: projectRoot,
      stdio: 'inherit',
      shell: false,
      windowsHide: true,
      env: { ...process.env, ...makeProgramEnv }
    });

    activeChildProcess = child;

    child.on('error', (err) => {
      activeChildProcess = null;
      reject(err);
    });

    child.on('exit', (code) => {
      activeChildProcess = null;
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Command failed with exit code ${code}: ${command} ${args.join(' ')}`));
      }
    });
  });
}

async function copyWasmArtifacts() {
  await mkdir(packageWasmDir, { recursive: true });

  const artifactMap = {
    'CogentLM.js': `${artifactPrefix}.js`,
    'CogentLM.wasm': `${artifactPrefix}.wasm`
  };

  for (const [sourceName, targetName] of Object.entries(artifactMap)) {
    const sourcePath = path.join(buildDistDir, sourceName);
    const targetPath = path.join(packageWasmDir, targetName);

    console.log(`${buildLabel} copying artifact to dist/${packageWasmSubdir}/${targetName}`);
    await copyFile(sourcePath, targetPath);
  }
}

async function getWasmOptPath() {
  const binaryName = isWindows ? 'wasm-opt.exe' : 'wasm-opt';
  const emsdkRoot = resolveEmsdkRoot();
  if (emsdkRoot) {
    const candidate = path.join(
      emsdkRoot,
      'upstream',
      'emscripten',
      'bin',
      binaryName
    );
    if (existsSync(candidate)) {
      return normalizeHostPath(candidate);
    }
  }

  if (commandAvailable('wasm-opt')) {
    return 'wasm-opt';
  }

  return null;
}

async function optimizeWasmWithBinaryen() {
  const wasmOptPath = await getWasmOptPath();
  if (!wasmOptPath) {
    console.log(`${buildLabel} wasm-opt not found, skipping extra binaryen optimization`);
    return;
  }

  const wasmPath = path.join(packageWasmDir, `${artifactPrefix}.wasm`);
  console.log(`${buildLabel} optimizing ${artifactPrefix}.wasm with wasm-opt (-O3)`);

  const tempWasmPath = `${wasmPath}.opt`;
  try {
    await runCommand(wasmOptPath, ['-O3', '--asyncify', wasmPath, '-o', tempWasmPath]);
    await copyFile(tempWasmPath, wasmPath);
  } finally {
    if (existsSync(tempWasmPath)) {
      await rm(tempWasmPath, { force: true });
    }
  }
}

function parseWasmExportsAndImports(wasmBuffer) {
  // Simple binary scanner for WebAssembly imports & exports.
  let offset = 8;
  const functionImports = [];
  const functionExports = [];
  const functionTypes = [];
  const importedFunctionCount = 0;

  function readVarUint32() {
    let result = 0;
    let shift = 0;
    while (offset < wasmBuffer.length) {
      const byte = wasmBuffer[offset++];
      result |= (byte & 0x7f) << shift;
      if ((byte & 0x80) === 0) {
        break;
      }
      shift += 7;
    }
    return result;
  }

  function readString() {
    const length = readVarUint32();
    const str = wasmBuffer.toString('utf8', offset, offset + length);
    offset += length;
    return str;
  }

  while (offset < wasmBuffer.length) {
    const sectionId = wasmBuffer[offset++];
    const sectionSize = readVarUint32();
    const sectionEnd = offset + sectionSize;

    if (sectionId === 2) { // Import Section
      const count = readVarUint32();
      for (let i = 0; i < count; i++) {
        const moduleName = readString();
        const fieldName = readString();
        const kind = wasmBuffer[offset++];
        if (kind === 0) { // Function import
          const typeIndex = readVarUint32();
          functionImports.push({ module: moduleName, name: fieldName, typeIndex });
        } else if (kind === 1) { // Table
          offset++; // element type
          const limitsKind = wasmBuffer[offset++];
          readVarUint32(); // initial limit
          if (limitsKind === 1) readVarUint32(); // maximum limit
        } else if (kind === 2) { // Memory
          const limitsKind = wasmBuffer[offset++];
          readVarUint32(); // initial limit
          if (limitsKind === 1) readVarUint32(); // maximum limit
        } else if (kind === 3) { // Global
          offset += 2; // type and mutability
        }
      }
    } else if (sectionId === 7) { // Export Section
      const count = readVarUint32();
      for (let i = 0; i < count; i++) {
        const name = readString();
        const kind = wasmBuffer[offset++];
        const index = readVarUint32();
        if (kind === 0) {
          functionExports.push({ name, index });
        }
      }
    }

    offset = sectionEnd;
  }

  return { functionImports, functionExports };
}

function hasCallableSyscallImplementation(moduleText, name) {
  // Match standard Emscripten syscall assignments.
  const escapedName = name.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
  const patterns = [
    new RegExp(`_${escapedName}\\s*:\\s*`),
    new RegExp(`function\\s+_${escapedName}\\b`),
    new RegExp(`var\\s+_${escapedName}\\b`)
  ];

  return patterns.some((pattern) => pattern.test(moduleText));
}

function validateCopiedWasmGlue() {
  const wasmPath = path.join(packageWasmDir, `${artifactPrefix}.wasm`);
  const jsPath = path.join(packageWasmDir, `${artifactPrefix}.js`);

  if (!existsSync(wasmPath) || !existsSync(jsPath)) {
    return;
  }

  const wasmBuffer = readFileSync(wasmPath);
  const moduleText = readFileSync(jsPath, 'utf8');

  const { functionImports, functionExports } = parseWasmExportsAndImports(wasmBuffer);
  const envImports = functionImports.filter((item) => item.module === 'env');

  const missingBindings = envImports
    .map((item) => item.name)
    .filter((name) => !name.startsWith('__syscall_'))
    .filter((name) => !moduleText.includes(`_${name}`))
    .filter((name) => !moduleText.includes(name));

  if (missingBindings.length > 0) {
    throw new Error(
      `Copied Emscripten JS glue is missing wasm import bindings: ${missingBindings.join(', ')}.`
    );
  }

  const missingSyscallImplementations = functionImports
    .map((item) => item.name)
    .filter((name) => name.startsWith('__syscall_'))
    .filter((name) => !hasCallableSyscallImplementation(moduleText, name));

  if (missingSyscallImplementations.length > 0) {
    throw new Error(
      `Copied Emscripten JS glue is missing syscall implementations: ${missingSyscallImplementations.join(', ')}.`
    );
  }

  console.log(
    `${buildLabel} validated wasm JS glue for ${functionImports.length} env function imports`
  );
}

const buildConfig = resolveBuildConfiguration();
validateMaximumMemorySetting(maximumMemory);
validateLtoMode(ltoMode);

rustBrowserStaticlib = buildRustBrowserStaticlib();

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
  `-DCE_WASM_FILESYSTEM=${enableFilesystem ? 'ON' : 'OFF'}`,
  `-DCE_WASM_AGGRESSIVE_OPT=${enableAggressiveOpt ? 'ON' : 'OFF'}`,
  `-DCE_SUPPRESS_LLAMA_LOGS=${suppressLlamaLogs ? 'ON' : 'OFF'}`,
  `-DMTMD_NO_LOGGING=${suppressMtmdLogs ? 'ON' : 'OFF'}`,
  `-DCE_WASM_PTHREADS=${enablePthreads ? 'ON' : 'OFF'}`,
  `-DCE_WASM_USE_JSPI=${enableJspi ? 'ON' : 'OFF'}`,
  `-DCE_WASM_ENVIRONMENT=${emscriptenEnvironment}`,
  `-DCE_WASM_MEM64=${enableMemory64 ? 'ON' : 'OFF'}`,
  `-DCE_WASM_LTO_MODE=${ltoMode}`,
  `-DCE_WASM_INITIAL_MEMORY=${initialMemory}`,
  `-DCE_WASM_STACK_SIZE=${stackSize}`,
  '-DLLAMA_BUILD_HTML=OFF'
];

cmakeConfigureArgs.push(`-DCE_WASM_RUST_BROWSER_LIB=${rustBrowserStaticlib}`);

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
  `${buildLabel} generator=${buildConfig.generator} mem64=${enableMemory64 ? 'on' : 'off'} pthreads=${enablePthreads ? 'on' : 'off'} lto=${ltoMode.toLowerCase()} output=dist/${packageWasmSubdir}` +
    ' rust_browser=on' +
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
await optimizeWasmWithBinaryen();
validateCopiedWasmGlue();
