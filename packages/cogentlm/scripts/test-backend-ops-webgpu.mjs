import http from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { access, readFile } from 'node:fs/promises';
import { createReadStream, existsSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const defaultBuildLabel = '[test-backend-ops:webgpu]';
const buildLabel = process.env.CE_TEST_BACKEND_OPS_BUILD_LABEL?.trim() || defaultBuildLabel;
const buildType = process.env.CE_TEST_BACKEND_OPS_BUILD_TYPE?.trim() || 'Release';
const isDebugBuild = buildType.toLowerCase() === 'debug';
const buildDirName =
  process.env.CE_TEST_BACKEND_OPS_BUILD_DIR_NAME?.trim() ||
  (isDebugBuild ? 'build-test-backend-ops-webgpu-debug' : 'build-test-backend-ops-webgpu');
const buildDir = path.join(projectRoot, buildDirName);
const buildOutputDir = path.join(buildDir, 'bin');
const llamaCppRoot = path.join(projectRoot, 'third_party', 'llama.cpp');
const runnerDir = path.join(scriptDir, 'webgpu-test-runner');
const browserHarnessScript = path.join(scriptDir, 'run-webgpu-browser-harness.mjs');
const testTargetName = 'test-backend-ops';
const moduleFileName = `${testTargetName}.js`;
const wasmFileName = `${testTargetName}.wasm`;
const isWindows = process.platform === 'win32';
const supportedGenerators = new Set(['Ninja', 'NMake Makefiles', 'Unix Makefiles']);
const enableJspi = true;
const emscriptenEnvironment = 'web,worker';
const enableAggressiveOpt = !isDebugBuild && readBooleanEnv('CE_TEST_BACKEND_OPS_AGGRESSIVE_OPT', true);
const pauseBeforeRun = readBooleanEnv('CE_WEBGPU_PAUSE_BEFORE_RUN', false);
const maximumMemory = process.env.CE_WASM_MAXIMUM_MEMORY?.trim() || '4096MB';

let activeChildProcess = null;
let signalHandlersInstalled = false;
let activeMakeProgramDir = null;
let cachedCmakeExecutable = null;
let cachedNodeExecutable = null;

function readBooleanEnv(name, fallback = false) {
  const value = process.env[name]?.trim().toLowerCase();
  if (!value) {
    return fallback;
  }

  return value === '1' || value === 'true' || value === 'yes' || value === 'on';
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
  const maximumAllowedBytes = 4 * 1024 ** 3;
  const configuredBytes = parseMemorySizeBytes(rawValue);
  if (configuredBytes > maximumAllowedBytes) {
    throw new Error(
      `CE_WASM_MAXIMUM_MEMORY=${rawValue} exceeds the wasm32 limit of 4GB. Use 4096MB for test-backend-ops WebGPU builds.`
    );
  }
}

function printHelp() {
  console.log(`Usage: bun ./scripts/test-backend-ops-webgpu.mjs [test-backend-ops args]

Examples:
  bun run test:backend-ops:webgpu -- --list-ops
  bun run test:backend-ops:webgpu -- support --output csv
  bun run test:backend-ops:webgpu -- test -o MUL_MAT

Notes:
  - The runner injects -b WebGPU unless you pass -b yourself.
  - Chromium must be installed for Playwright. Run "bunx playwright install chromium" if needed.
  - Wrapper scripts can override build behavior with CE_TEST_BACKEND_OPS_BUILD_TYPE and related env vars.
`);
}

function parseForwardedArgs(argv) {
  if (argv.includes('--help') || argv.includes('-h')) {
    printHelp();
    process.exit(0);
  }

  const args = [...argv];
  const hasBackendSelection = args.includes('-b');
  if (!hasBackendSelection) {
    args.unshift('WebGPU');
    args.unshift('-b');
  }

  return args;
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
    encoding: 'utf8',
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
    windowsHide: true,
  });

  return !result.error && result.status === 0;
}

function resolveCmakeExecutable() {
  if (cachedCmakeExecutable) {
    return cachedCmakeExecutable;
  }

  const candidates = [
    ...findProgramsOnPath(isWindows ? 'cmake.exe' : 'cmake'),
    isWindows ? path.join(process.env.ProgramFiles ?? 'C:\\Program Files', 'CMake', 'bin', 'cmake.exe') : null,
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

function resolveNodeExecutable() {
  if (cachedNodeExecutable) {
    return cachedNodeExecutable;
  }

  const candidates = [
    process.env.NODE?.trim(),
    ...findProgramsOnPath(isWindows ? 'node.exe' : 'node'),
  ]
    .map((candidate) => normalizeHostPath(candidate))
    .filter(Boolean);

  const nodeExecutable = candidates.find((candidate) => existsSync(candidate));
  if (!nodeExecutable) {
    throw new Error('Node.js executable not found. Install Node.js or add it to PATH.');
  }

  cachedNodeExecutable = nodeExecutable;
  return nodeExecutable;
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
    candidates.push(path.join(emsdkRoot, 'upstream', 'emscripten', 'cache', 'ports', 'emdawnwebgpu', 'emdawnwebgpu_pkg'));
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

  const detectedNinja = detectBundledNinjaMakeProgram() ?? detectNinjaFromCmakeInstall() ?? detectNinjaFromVisualStudio();
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
  const cachedBuildType = getCacheEntry(cacheText, 'CMAKE_BUILD_TYPE');
  const cachedDebug = getCacheEntry(cacheText, 'CE_WASM_DEBUG');
  const reasons = [];

  if (cacheText.includes('CMAKE_MAKE_PROGRAM:FILEPATH=CMAKE_MAKE_PROGRAM-NOTFOUND')) {
    reasons.push('CMAKE_MAKE_PROGRAM-NOTFOUND');
  }

  const cachedMemory64 = getCacheEntry(cacheText, 'CE_WASM_MEM64') ?? getCacheEntry(cacheText, 'LLAMA_WASM_MEM64');
  if (cachedMemory64 === 'ON') {
    reasons.push(`LLAMA_WASM_MEM64=${cachedMemory64}`);
  }

  const cachedMaximumMemory = getCacheEntry(cacheText, 'CE_WASM_MAXIMUM_MEMORY');
  if (cachedMaximumMemory && cachedMaximumMemory !== maximumMemory) {
    reasons.push(`CE_WASM_MAXIMUM_MEMORY=${cachedMaximumMemory}`);
  }

  if (expectedGenerator && cachedGenerator && cachedGenerator !== expectedGenerator) {
    reasons.push(`generator=${cachedGenerator}`);
  }

  if (cachedBuildType && cachedBuildType !== buildType) {
    reasons.push(`build_type=${cachedBuildType}`);
  }

  if (isDebugBuild && cachedDebug !== 'ON') {
    reasons.push(`CE_WASM_DEBUG=${cachedDebug ?? 'OFF'}`);
  }

  if (cachedDebug === 'ON' && !isDebugBuild) {
    reasons.push('CE_WASM_DEBUG=ON');
  }

  const cachedSourceDir = getCacheEntry(cacheText, 'CogentLM_SOURCE_DIR') || 
                          getCacheEntry(cacheText, 'CMAKE_HOME_DIRECTORY');
  if (cachedSourceDir && path.resolve(cachedSourceDir) !== path.resolve(projectRoot)) {
    reasons.push(`source_dir=${cachedSourceDir}`);
  }

  if (!cacheText.includes('CE_BUILD_WEBGPU_TEST_BACKEND_OPS:BOOL=ON')) {
    reasons.push('CE_BUILD_WEBGPU_TEST_BACKEND_OPS=OFF');
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
      windowsHide: true,
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

async function runCommand(executable, args, envOverrides = {}) {
  const env = { ...process.env, ...envOverrides };
  prependPathEntry(env, activeMakeProgramDir);

  console.log(`${buildLabel} run: ${executable} ${args.join(' ')}`);

  installSignalHandlers();

  const childProcess = spawn(executable, args, {
    cwd: projectRoot,
    stdio: 'inherit',
    shell: false,
    windowsHide: true,
    env,
    detached: !isWindows,
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

async function runCommandWithExitCode(executable, args, envOverrides = {}) {
  const env = { ...process.env, ...envOverrides };
  prependPathEntry(env, activeMakeProgramDir);

  console.log(`${buildLabel} run: ${executable} ${args.join(' ')}`);

  installSignalHandlers();

  const childProcess = spawn(executable, args, {
    cwd: projectRoot,
    stdio: 'inherit',
    shell: false,
    windowsHide: true,
    env,
    detached: !isWindows,
  });

  activeChildProcess = childProcess;

  try {
    return await new Promise((resolve, reject) => {
      childProcess.once('error', reject);
      childProcess.once('exit', (code, signal) => {
        if (signal) {
          reject(new Error(`Command terminated by signal ${signal}: ${executable} ${args.join(' ')}`));
          return;
        }

        resolve(code ?? 0);
      });
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Command failed to start: ${executable} ${args.join(' ')}\n${message}`);
  } finally {
    if (activeChildProcess === childProcess) {
      activeChildProcess = null;
    }
  }
}

function getMimeType(filePath) {
  switch (path.extname(filePath).toLowerCase()) {
    case '.html':
      return 'text/html; charset=utf-8';
    case '.js':
    case '.mjs':
      return 'text/javascript; charset=utf-8';
    case '.wasm':
      return 'application/wasm';
    case '.map':
      return 'application/json; charset=utf-8';
    case '.debug':
      return 'application/wasm';
    case '.json':
      return 'application/json; charset=utf-8';
    case '.css':
      return 'text/css; charset=utf-8';
    case '.c':
    case '.cc':
    case '.cpp':
    case '.cxx':
    case '.h':
    case '.hh':
    case '.hpp':
    case '.hxx':
    case '.inc':
    case '.inl':
      return 'text/plain; charset=utf-8';
    default:
      return 'application/octet-stream';
  }
}

function resolveStaticFile(rootDir, requestPath) {
  const normalizedRoot = path.resolve(rootDir);
  const normalizedRequest = requestPath.split('/').filter(Boolean);
  const candidatePath = path.resolve(normalizedRoot, ...normalizedRequest);
  const relativePath = path.relative(normalizedRoot, candidatePath);

  if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
    return null;
  }

  return candidatePath;
}

function resolveDebugRequestFile(requestPath) {
  if (requestPath.startsWith('/__runner__/')) {
    return resolveStaticFile(runnerDir, requestPath.slice('/__runner__/'.length));
  }

  const candidatePaths = [];

  if (emsdkRoot && requestPath.startsWith('/emsdk/upstream/')) {
    candidatePaths.push(resolveStaticFile(emsdkRoot, requestPath.slice('/emsdk/'.length)));
  }

  if (emsdkRoot && requestPath.startsWith('/emsdk/emscripten/')) {
    candidatePaths.push(resolveStaticFile(path.join(emsdkRoot, 'upstream'), requestPath.slice('/emsdk/'.length)));
  }

  candidatePaths.push(resolveStaticFile(buildOutputDir, requestPath));
  candidatePaths.push(resolveStaticFile(projectRoot, requestPath));

  return candidatePaths.find((candidatePath) => candidatePath && existsSync(candidatePath)) ?? null;
}

async function startStaticServer() {
  const server = http.createServer(async (request, response) => {
    try {
      const requestUrl = new URL(request.url ?? '/', 'http://127.0.0.1');
      const requestPath = decodeURIComponent(requestUrl.pathname);

      response.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      response.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
      response.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
      response.setHeader('Pragma', 'no-cache');
      response.setHeader('Expires', '0');

      const filePath = resolveDebugRequestFile(requestPath);

      if (!filePath || !existsSync(filePath)) {
        response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
        response.end('404 Not Found');
        return;
      }

      response.writeHead(200, { 'Content-Type': getMimeType(filePath) });
      createReadStream(filePath).pipe(response);
    } catch (error) {
      response.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
      response.end(`500 Internal Server Error\n${error instanceof Error ? error.message : String(error)}`);
    }
  });

  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });

  const address = server.address();
  if (!address || typeof address === 'string') {
    throw new Error('Failed to start the local static server.');
  }

  return {
    server,
    origin: `http://127.0.0.1:${address.port}`,
  };
}

async function stopStaticServer(server) {
  await new Promise((resolve, reject) => {
    server.close((error) => {
      if (error) {
        reject(error);
        return;
      }

      resolve();
    });
  });
}

async function ensureBuildArtifacts() {
  try {
    await access(path.join(buildOutputDir, moduleFileName));
    await access(path.join(buildOutputDir, wasmFileName));
  } catch {
    throw new Error(`Missing build artifacts in ${buildOutputDir}. Expected ${moduleFileName} and ${wasmFileName}.`);
  }

  await readFile(path.join(runnerDir, 'runner.html'));
  await readFile(path.join(runnerDir, 'runner.mjs'));
}

async function runBrowserHarness(forwardedArgs) {
  const { server, origin } = await startStaticServer();

  try {
    const runnerUrl = new URL('/__runner__/runner.html', origin);
    runnerUrl.searchParams.set('module', `/${moduleFileName}`);
    runnerUrl.searchParams.set('args', JSON.stringify(forwardedArgs));
    runnerUrl.searchParams.set('pauseBeforeRun', pauseBeforeRun ? '1' : '0');

    const nodeExecutable = resolveNodeExecutable();
    const browserHarnessEnv = {};

    if (process.env.CE_WEBGPU_BROWSER_MODE?.trim()) {
      browserHarnessEnv.CE_WEBGPU_BROWSER_MODE = process.env.CE_WEBGPU_BROWSER_MODE.trim();
    }

    if (process.env.CE_WEBGPU_REMOTE_DEBUG_PORT?.trim()) {
      browserHarnessEnv.CE_WEBGPU_REMOTE_DEBUG_PORT = process.env.CE_WEBGPU_REMOTE_DEBUG_PORT.trim();
    }

    return await runCommandWithExitCode(nodeExecutable, [browserHarnessScript, runnerUrl.href], browserHarnessEnv);
  } finally {
    await stopStaticServer(server).catch(() => {});
  }
}

const forwardedArgs = parseForwardedArgs(Bun.argv.slice(2));
const buildConfig = resolveBuildConfiguration();
validateMaximumMemorySetting(maximumMemory);

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
  `-DCMAKE_BUILD_TYPE=${buildType}`,
  `-DCMAKE_TOOLCHAIN_FILE=${toolchainPath}`,
  '-DCE_BUILD_WEBGPU_TEST_BACKEND_OPS=ON',
  `-DCE_WASM_DEBUG=${isDebugBuild ? 'ON' : 'OFF'}`,
  `-DCE_WASM_AGGRESSIVE_OPT=${enableAggressiveOpt ? 'ON' : 'OFF'}`,
  `-DCE_WASM_USE_JSPI=${enableJspi ? 'ON' : 'OFF'}`,
  `-DCE_WASM_ENVIRONMENT=${emscriptenEnvironment}`,
  `-DCE_WASM_MAXIMUM_MEMORY=${maximumMemory}`,
  '-DGGML_WEBGPU=ON',
  '-DLLAMA_OPENSSL=OFF',
  '-DCE_WASM_MEM64=OFF',
  '-DLLAMA_BUILD_HTML=OFF',
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
  `${buildLabel} build_type=${buildType} generator=${buildConfig.generator} max_memory=${maximumMemory}` +
    (buildConfig.makeProgram ? ` make_program=${buildConfig.makeProgram}` : '')
);

await runCommand(cmakeExecutable, cmakeConfigureArgs);
await runCommand(cmakeExecutable, ['--build', buildDirName, '--config', buildType, '--target', testTargetName]);
await ensureBuildArtifacts();

const exitCode = await runBrowserHarness(forwardedArgs);
process.exit(exitCode);
