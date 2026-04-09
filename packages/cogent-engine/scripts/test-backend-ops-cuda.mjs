import { spawn, spawnSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, rmSync, writeFileSync, unlinkSync, mkdtempSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// ── Constants ────────────────────────────────────────────────────────────────

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const llamaCppRoot = path.join(projectRoot, 'third_party', 'llama.cpp');
const buildLabel = process.env.CE_TEST_BACKEND_OPS_BUILD_LABEL?.trim() || '[test-backend-ops:cuda]';
const buildType = process.env.CE_TEST_BACKEND_OPS_BUILD_TYPE?.trim() || 'Release';
const isDebugBuild = buildType.toLowerCase() === 'debug';
const buildDirName =
  process.env.CE_TEST_BACKEND_OPS_BUILD_DIR_NAME?.trim() ||
  (isDebugBuild ? 'build-test-backend-ops-cuda-debug' : 'build-test-backend-ops-cuda');
const buildDir = path.join(projectRoot, buildDirName);
const buildOutputDir = path.join(buildDir, 'bin');
const testTargetName = 'test-backend-ops';
const isWindows = process.platform === 'win32';
const testExecutableName = isWindows ? `${testTargetName}.exe` : testTargetName;
const supportedGenerators = new Set(['Ninja', 'NMake Makefiles', 'Unix Makefiles']);
const supportedVisualStudioReleases = [
  { year: '2022', versionRange: '[17.0,18.0)', directoryTokens: ['2022', '17'] },
  { year: '2019', versionRange: '[16.0,17.0)', directoryTokens: ['2019', '16'] },
  { year: '2017', versionRange: '[15.0,16.0)', directoryTokens: ['2017', '15'] },
];
const supportedVisualStudioYears = new Set(supportedVisualStudioReleases.map((release) => release.year));
const visualStudioTokenToYear = new Map([
  ['2026', '2026'],
  ['18', '2026'],
  ['2022', '2022'],
  ['17', '2022'],
  ['2019', '2019'],
  ['16', '2019'],
  ['2017', '2017'],
  ['15', '2017'],
]);

let activeChildProcess = null;
let signalHandlersInstalled = false;

// ── Helpers ──────────────────────────────────────────────────────────────────

function log(message) {
  console.log(`${buildLabel} ${message}`);
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
  return result.stdout.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
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
  const candidates = [
    ...findProgramsOnPath(isWindows ? 'cmake.exe' : 'cmake'),
    isWindows ? path.join(process.env.ProgramFiles ?? 'C:\\Program Files', 'CMake', 'bin', 'cmake.exe') : null,
  ].map((c) => normalizeHostPath(c)).filter(Boolean);

  const cmake = candidates.find((c) => existsSync(c));
  if (!cmake) {
    throw new Error('CMake executable not found. Install CMake or add it to PATH.');
  }
  return cmake;
}

function getCacheEntry(cacheText, key) {
  const match = cacheText.match(new RegExp(`^${key}:[^=]*=(.*)$`, 'm'));
  return match ? match[1].trim() : null;
}

function normalizeComparisonPath(inputPath) {
  if (!inputPath) {
    return null;
  }

  const normalizedPath = normalizeHostPath(String(inputPath).trim());
  return isWindows ? normalizedPath.toLowerCase() : normalizedPath;
}

function pathsMatch(leftPath, rightPath) {
  return normalizeComparisonPath(leftPath) === normalizeComparisonPath(rightPath);
}

function getEnvKey(env, name) {
  return Object.keys(env).find((key) => key.toLowerCase() === name.toLowerCase()) ?? null;
}

function getEnvValue(env, name) {
  const key = getEnvKey(env, name);
  return key ? env[key] : undefined;
}

function setEnvValue(env, name, value) {
  const matchingKeys = Object.keys(env).filter((key) => key.toLowerCase() === name.toLowerCase());
  const targetKey = matchingKeys[0] ?? name;

  for (const key of matchingKeys) {
    if (key !== targetKey) {
      delete env[key];
    }
  }

  if (value === undefined || value === null) {
    delete env[targetKey];
    return;
  }

  env[targetKey] = value;
}

function mergeEnvironment(targetEnv, sourceEnv) {
  for (const [key, value] of Object.entries(sourceEnv)) {
    setEnvValue(targetEnv, key, value);
  }

  return targetEnv;
}

function splitPathEntries(pathValue) {
  if (!pathValue) {
    return [];
  }

  const separator = isWindows ? ';' : ':';
  return pathValue
    .split(separator)
    .map((entry) => entry.trim().replace(/^"(.*)"$/, '$1'))
    .map((entry) => normalizeHostPath(entry))
    .filter(Boolean);
}

function findProgramInPathEntries(programName, pathValue) {
  for (const entry of splitPathEntries(pathValue)) {
    const candidate = normalizeHostPath(path.join(entry, programName));
    if (existsSync(candidate)) {
      return candidate;
    }
  }

  return null;
}

function resolveVswhereExecutable() {
  if (!isWindows) {
    return null;
  }

  const vswherePath = path.join(
    process.env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)',
    'Microsoft Visual Studio',
    'Installer',
    'vswhere.exe'
  );

  return existsSync(vswherePath) ? normalizeHostPath(vswherePath) : null;
}

function getVisualStudioMetadataFromPath(inputPath) {
  if (!isWindows || !inputPath) {
    return null;
  }

  const normalizedPath = normalizeHostPath(inputPath);
  const parts = normalizedPath.split('\\');
  const visualStudioIndex = parts.findIndex((part) => part.toLowerCase() === 'microsoft visual studio');

  if (visualStudioIndex < 0 || visualStudioIndex + 1 >= parts.length) {
    return null;
  }

  const versionToken = parts[visualStudioIndex + 1];
  return {
    versionToken,
    year: visualStudioTokenToYear.get(versionToken) ?? versionToken,
    edition: parts[visualStudioIndex + 2] ?? null,
  };
}

function isSupportedVisualStudioYear(year) {
  return Boolean(year && supportedVisualStudioYears.has(year));
}

function describeVisualStudioInstall(installation) {
  const editionSuffix = installation.edition ? ` ${installation.edition}` : '';
  return `Visual Studio ${installation.year}${editionSuffix}`;
}

function getVisualStudioInstallationPathFromCompilerPath(compilerPath) {
  if (!isWindows || !compilerPath) {
    return null;
  }

  const normalizedPath = normalizeHostPath(compilerPath);
  const marker = `${path.win32.sep}VC${path.win32.sep}Tools${path.win32.sep}MSVC${path.win32.sep}`;
  const markerIndex = normalizedPath.toLowerCase().indexOf(marker.toLowerCase());
  if (markerIndex < 0) {
    return null;
  }

  return normalizedPath.slice(0, markerIndex);
}

function toVisualStudioInstallationInfo(installationPath) {
  const normalizedPath = normalizeHostPath(installationPath);
  const metadata = getVisualStudioMetadataFromPath(normalizedPath);
  if (!normalizedPath || !metadata) {
    return null;
  }

  const vcvarsallPath = normalizeHostPath(path.join(normalizedPath, 'VC', 'Auxiliary', 'Build', 'vcvarsall.bat'));
  if (!existsSync(vcvarsallPath)) {
    return null;
  }

  return {
    installationPath: normalizedPath,
    vcvarsallPath,
    year: metadata.year,
    edition: metadata.edition,
  };
}

// ── Help ─────────────────────────────────────────────────────────────────────

function printHelp() {
  console.log(`Usage: bun ./scripts/test-backend-ops-cuda.mjs [test-backend-ops args]

Examples:
  bun run test:backend-ops:cuda -- --list-ops
  bun run test:backend-ops:cuda -- support --output csv
  bun run test:backend-ops:cuda -- test -o MUL_MAT

Notes:
  - Leaves backend selection unset by default so the CUDA build can run its discovered CUDA devices.
  - If you pass -b explicitly, use the exact device name reported by test-backend-ops, e.g. CUDA0.
  - Requires: CUDA toolkit, Visual Studio C++ tools (Windows), CMake, Ninja.
  - Override build type: CE_TEST_BACKEND_OPS_BUILD_TYPE=Debug
`);
}

function normalizeBackendAlias(args) {
  const normalizedArgs = [];

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];

    if (arg === '-b' && index + 1 < args.length) {
      const backendName = args[index + 1];
      index += 1;

      if (backendName.trim().toUpperCase() === 'CUDA') {
        log('ignoring generic backend filter "CUDA"; use an exact device name such as CUDA0 or leave -b unset');
        continue;
      }

      normalizedArgs.push('-b', backendName);
      continue;
    }

    normalizedArgs.push(arg);
  }

  return normalizedArgs;
}

function parseForwardedArgs(argv) {
  if (argv.includes('--help') || argv.includes('-h')) {
    printHelp();
    process.exit(0);
  }

  return normalizeBackendAlias([...argv]);
}

// ── MSVC developer environment (Windows only) ───────────────────────────────
//
// CUDA 12.9 supports Visual Studio 2017-2022 only. We therefore prefer a
// supported Visual Studio install, with VS 2022 first, and avoid relying on
// whatever host compiler might happen to be on PATH.

function findSupportedVisualStudioInstallation() {
  if (!isWindows) {
    return null;
  }

  const vswherePath = resolveVswhereExecutable();
  if (vswherePath) {
    for (const release of supportedVisualStudioReleases) {
      const result = spawnSync(vswherePath, [
        '-latest',
        '-products', '*',
        '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
        '-version', release.versionRange,
        '-format', 'json',
      ], {
        cwd: projectRoot,
        stdio: ['ignore', 'pipe', 'ignore'],
        shell: false,
        windowsHide: true,
        encoding: 'utf8',
      });

      if (result.error || result.status !== 0 || !result.stdout?.trim()) {
        continue;
      }

      try {
        const matches = JSON.parse(result.stdout);
        const installation = toVisualStudioInstallationInfo(matches[0]?.installationPath?.trim());
        if (installation) {
          return installation;
        }
      } catch {
        // Ignore malformed output and fall back to filesystem scanning.
      }
    }
  }

  const visualStudioRoot = path.join(process.env.ProgramFiles ?? 'C:\\Program Files', 'Microsoft Visual Studio');
  if (!existsSync(visualStudioRoot)) {
    return null;
  }

  const visitedInstallations = new Set();

  for (const release of supportedVisualStudioReleases) {
    for (const directoryToken of release.directoryTokens) {
      const yearDir = path.join(visualStudioRoot, directoryToken);
      if (!existsSync(yearDir)) {
        continue;
      }

      const editions = readdirSync(yearDir, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .map((entry) => entry.name)
        .sort();

      for (const edition of editions) {
        const installationPath = normalizeHostPath(path.join(yearDir, edition));
        const installationKey = normalizeComparisonPath(installationPath);
        if (visitedInstallations.has(installationKey)) {
          continue;
        }

        visitedInstallations.add(installationKey);
        const installation = toVisualStudioInstallationInfo(installationPath);
        if (installation && installation.year === release.year) {
          return installation;
        }
      }
    }
  }

  return null;
}

function captureVcvarsallEnvironment(vcvarsallPath) {
  log(`sourcing VS environment: ${vcvarsallPath}`);

  // Write a temp batch file to avoid Windows cmd.exe quoting issues.
  // cmd.exe expects "" for escaping, but Node/Bun use \" via CreateProcessW,
  // causing paths-with-spaces to break when passed as arguments.
  const tmpDir = mkdtempSync(path.join(os.tmpdir(), 'ce-vcvars-'));
  const tmpBat = path.join(tmpDir, 'capture_env.bat');
  writeFileSync(tmpBat, `@call "${vcvarsallPath}" x64\r\n@if errorlevel 1 exit /b 1\r\n@set\r\n`);

  const result = spawnSync('cmd.exe', ['/C', tmpBat], {
    cwd: projectRoot,
    stdio: ['ignore', 'pipe', 'pipe'],
    shell: false,
    windowsHide: true,
    encoding: 'utf8',
  });

  try {
    unlinkSync(tmpBat);
    rmSync(tmpDir, { recursive: true, force: true });
  } catch {}

  if (result.error || result.status !== 0) {
    const stderr = result.stderr?.trim() || '';
    throw new Error(`Failed to source vcvarsall.bat.\n${stderr || 'Run from a supported Developer Command Prompt instead.'}`);
  }

  const env = {};
  for (const line of result.stdout.split(/\r?\n/)) {
    const eq = line.indexOf('=');
    if (eq > 0) {
      env[line.substring(0, eq)] = line.substring(eq + 1);
    }
  }

  if (!getEnvValue(env, 'INCLUDE') || !getEnvValue(env, 'LIB')) {
    throw new Error(
      'vcvarsall.bat ran but INCLUDE/LIB were not set. ' +
      'Ensure the "Desktop development with C++" workload is installed.'
    );
  }

  return env;
}

function resolveAmbientVisualStudioCompiler() {
  if (!isWindows) {
    return null;
  }

  const clPath = findProgramsOnPath('cl.exe').map((candidate) => normalizeHostPath(candidate)).find(Boolean);
  if (!clPath) {
    return null;
  }

  const metadata = getVisualStudioMetadataFromPath(clPath);
  return {
    clPath,
    installationPath: normalizeHostPath(getVisualStudioInstallationPathFromCompilerPath(clPath)),
    year: metadata?.year ?? null,
    edition: metadata?.edition ?? null,
  };
}

function captureMsvcEnvironment() {
  if (!isWindows) {
    return { environment: {}, clPath: null, installationPath: null, year: null, edition: null };
  }

  const ambientCompiler = resolveAmbientVisualStudioCompiler();
  if (ambientCompiler?.clPath && ambientCompiler.year && !isSupportedVisualStudioYear(ambientCompiler.year)) {
    log(`ignoring ambient MSVC from unsupported Visual Studio ${ambientCompiler.year}: ${ambientCompiler.clPath}`);
  }

  const installation = findSupportedVisualStudioInstallation();
  if (!installation) {
    const includeSet = Boolean(getEnvValue(process.env, 'INCLUDE')?.trim());
    const libSet = Boolean(getEnvValue(process.env, 'LIB')?.trim());
    if (ambientCompiler?.clPath && isSupportedVisualStudioYear(ambientCompiler.year) && includeSet && libSet) {
      log(`using MSVC from current environment: ${ambientCompiler.clPath}`);
      return { environment: {}, ...ambientCompiler };
    }

    throw new Error(
      'No supported Visual Studio installation found. CUDA 12.9 supports Visual Studio 2017-2022; ' +
      'install Visual Studio 2022 C++ tools or run from a supported Developer Command Prompt.'
    );
  }

  const environment = captureVcvarsallEnvironment(installation.vcvarsallPath);
  const clPath = findProgramInPathEntries('cl.exe', getEnvValue(environment, 'PATH'));
  if (!clPath) {
    throw new Error(`Failed to locate cl.exe after sourcing ${installation.vcvarsallPath}.`);
  }

  log(`VS developer environment captured from ${describeVisualStudioInstall(installation)} (${clPath})`);
  return {
    environment,
    clPath,
    installationPath: installation.installationPath,
    year: installation.year,
    edition: installation.edition,
  };
}

// ── CUDA detection ──────────────────────────────────────────────────────────

function resolveCudaToolkit() {
  const nvccName = isWindows ? 'nvcc.exe' : 'nvcc';
  const explicitNvcc = normalizeHostPath(process.env.CUDACXX?.trim());
  if (explicitNvcc) {
    if (!existsSync(explicitNvcc)) {
      throw new Error(`CUDACXX is set but does not exist: ${explicitNvcc}`);
    }

    const binDir = normalizeHostPath(path.dirname(explicitNvcc));
    const toolkitRoot = normalizeHostPath(path.dirname(binDir));
    log(`CUDA toolkit selected via CUDACXX: root=${toolkitRoot} nvcc=${explicitNvcc}`);
    return { source: 'CUDACXX', nvccPath: explicitNvcc, binDir, toolkitRoot };
  }

  const cudaPath = normalizeHostPath(process.env.CUDA_PATH?.trim());
  if (cudaPath) {
    const nvccPath = normalizeHostPath(path.join(cudaPath, 'bin', nvccName));
    if (existsSync(nvccPath)) {
      log(`CUDA toolkit selected via CUDA_PATH: root=${cudaPath} nvcc=${nvccPath}`);
      return { source: 'CUDA_PATH', nvccPath, binDir: normalizeHostPath(path.join(cudaPath, 'bin')), toolkitRoot: cudaPath };
    }
  }

  const nvccPath = findProgramsOnPath(nvccName)
    .map((candidate) => normalizeHostPath(candidate))
    .find((candidate) => existsSync(candidate));

  if (nvccPath) {
    const binDir = normalizeHostPath(path.dirname(nvccPath));
    const toolkitRoot = normalizeHostPath(path.dirname(binDir));
    log(`CUDA toolkit selected from PATH: root=${toolkitRoot} nvcc=${nvccPath}`);
    return { source: 'PATH', nvccPath, binDir, toolkitRoot };
  }

  throw new Error(
    'CUDA toolkit not found. Install the CUDA toolkit and ensure nvcc is on PATH, or set CUDA_PATH/CUDACXX.'
  );
}

// ── Build configuration (generator + ninja detection) ───────────────────────

function inferGeneratorFromMakeProgram(makeProgramPath) {
  const lower = makeProgramPath.toLowerCase();
  if (lower.includes('ninja')) return 'Ninja';
  if (lower.includes('nmake')) return 'NMake Makefiles';
  if (lower.endsWith('make') || lower.endsWith('make.exe')) return 'Unix Makefiles';
  return null;
}

function detectNinja() {
  const ninjaName = isWindows ? 'ninja.exe' : 'ninja';

  // From CMake install directory
  for (const cmakePath of findProgramsOnPath(isWindows ? 'cmake.exe' : 'cmake')) {
    const candidate = path.join(path.dirname(cmakePath), ninjaName);
    if (existsSync(candidate)) return normalizeHostPath(candidate);
  }

  // From Visual Studio (Windows only)
  if (isWindows) {
    const vsRoot = path.join(process.env.ProgramFiles ?? 'C:\\Program Files', 'Microsoft Visual Studio');
    if (existsSync(vsRoot)) {
      for (const year of ['2026', '2022', '2019', '2017']) {
        const yearDir = path.join(vsRoot, year);
        if (!existsSync(yearDir)) continue;
        for (const entry of readdirSync(yearDir, { withFileTypes: true })) {
          if (!entry.isDirectory()) continue;
          const candidate = path.join(yearDir, entry.name,
            'Common7', 'IDE', 'CommonExtensions', 'Microsoft', 'CMake', 'Ninja', ninjaName);
          if (existsSync(candidate)) return normalizeHostPath(candidate);
        }
      }
    }
  }

  return null;
}

function resolveBuildConfiguration() {
  const generatorFromEnv = process.env.CMAKE_GENERATOR?.trim();
  const makeProgramFromEnv = normalizeHostPath(process.env.CMAKE_MAKE_PROGRAM?.trim());

  if (generatorFromEnv) {
    if (!supportedGenerators.has(generatorFromEnv)) {
      throw new Error(`Unsupported CMAKE_GENERATOR "${generatorFromEnv}".`);
    }
    return { generator: generatorFromEnv, makeProgram: makeProgramFromEnv || null };
  }

  if (makeProgramFromEnv) {
    const gen = inferGeneratorFromMakeProgram(makeProgramFromEnv);
    if (!gen) throw new Error('CMAKE_MAKE_PROGRAM set but CMAKE_GENERATOR could not be inferred.');
    return { generator: gen, makeProgram: makeProgramFromEnv };
  }

  const ninja = detectNinja();
  if (ninja) return { generator: 'Ninja', makeProgram: ninja };
  if (commandAvailable('ninja')) return { generator: 'Ninja', makeProgram: null };
  if (isWindows && commandAvailable('nmake', ['/?'])) return { generator: 'NMake Makefiles', makeProgram: null };
  if (commandAvailable('make')) return { generator: 'Unix Makefiles', makeProgram: null };

  throw new Error('No supported CMake generator found. Install Ninja or set CMAKE_GENERATOR.');
}

// ── Build cache invalidation ────────────────────────────────────────────────

function removeInvalidBuildDirectory(expectedGenerator, cudaToolkit, msvcDetails) {
  if (!existsSync(buildDir)) return;

  if (!existsSync(path.join(buildDir, 'CMakeCache.txt'))) {
    if (existsSync(path.join(buildDir, 'CMakeFiles')) || existsSync(path.join(buildDir, 'build.ninja'))) {
      log('removing incomplete build directory');
      rmSync(buildDir, { recursive: true, force: true });
    }
    return;
  }

  const cacheText = readFileSync(path.join(buildDir, 'CMakeCache.txt'), 'utf8');
  const reasons = [];

  if (cacheText.includes('CMAKE_MAKE_PROGRAM-NOTFOUND')) reasons.push('CMAKE_MAKE_PROGRAM-NOTFOUND');
  const cachedGen = getCacheEntry(cacheText, 'CMAKE_GENERATOR');
  if (expectedGenerator && cachedGen && cachedGen !== expectedGenerator) reasons.push(`generator=${cachedGen}`);
  const cachedBt = getCacheEntry(cacheText, 'CMAKE_BUILD_TYPE');
  if (cachedBt && cachedBt !== buildType) reasons.push(`build_type=${cachedBt}`);
  if (getCacheEntry(cacheText, 'GGML_CUDA') !== 'ON') reasons.push('GGML_CUDA!=ON');

  const cachedCudaCompiler = normalizeHostPath(getCacheEntry(cacheText, 'CMAKE_CUDA_COMPILER'));
  if (cachedCudaCompiler && !existsSync(cachedCudaCompiler)) reasons.push('CMAKE_CUDA_COMPILER missing');
  if (cudaToolkit && cachedCudaCompiler && !pathsMatch(cachedCudaCompiler, cudaToolkit.nvccPath)) {
    reasons.push(`cuda_compiler=${cachedCudaCompiler}`);
  }

  const cachedToolkitNvcc = normalizeHostPath(getCacheEntry(cacheText, 'CUDAToolkit_NVCC_EXECUTABLE'));
  if (cudaToolkit && cachedToolkitNvcc && !pathsMatch(cachedToolkitNvcc, cudaToolkit.nvccPath)) {
    reasons.push(`toolkit_nvcc=${cachedToolkitNvcc}`);
  }

  const cachedToolkitBinDir = normalizeHostPath(getCacheEntry(cacheText, 'CUDAToolkit_BIN_DIR'));
  if (cudaToolkit && cachedToolkitBinDir && !pathsMatch(cachedToolkitBinDir, cudaToolkit.binDir)) {
    reasons.push(`cuda_bin=${cachedToolkitBinDir}`);
  }

  const cachedCxxCompiler = normalizeHostPath(getCacheEntry(cacheText, 'CMAKE_CXX_COMPILER'));
  if (cachedCxxCompiler && !existsSync(cachedCxxCompiler)) reasons.push('CMAKE_CXX_COMPILER missing');
  const cachedCxxVisualStudio = getVisualStudioMetadataFromPath(cachedCxxCompiler);
  if (cachedCxxVisualStudio?.year && !isSupportedVisualStudioYear(cachedCxxVisualStudio.year)) {
    reasons.push(`unsupported_msvc=${cachedCxxVisualStudio.year}`);
  }
  if (isWindows && msvcDetails?.clPath && cachedCxxCompiler && !pathsMatch(cachedCxxCompiler, msvcDetails.clPath)) {
    reasons.push(`cxx_compiler=${cachedCxxCompiler}`);
  }

  const cachedCudaHostCompiler = normalizeHostPath(getCacheEntry(cacheText, 'CMAKE_CUDA_HOST_COMPILER'));
  if (isWindows && msvcDetails?.clPath && cachedCudaHostCompiler && !pathsMatch(cachedCudaHostCompiler, msvcDetails.clPath)) {
    reasons.push(`cuda_host_compiler=${cachedCudaHostCompiler}`);
  }

  if (reasons.length > 0) {
    log(`removing stale build directory (${reasons.join(', ')})`);
    rmSync(buildDir, { recursive: true, force: true });
  }
}

// ── Process management ──────────────────────────────────────────────────────

function terminateProcessTree(pid) {
  if (!pid) return;
  if (isWindows) {
    spawnSync('taskkill.exe', ['/T', '/F', '/PID', String(pid)], { stdio: 'ignore', shell: false, windowsHide: true });
  } else {
    try { process.kill(-pid, 'SIGTERM'); } catch {}
    try { process.kill(pid, 'SIGTERM'); } catch {}
  }
}

function installSignalHandlers() {
  if (signalHandlersInstalled) return;
  const exit = (sig) => {
    if (activeChildProcess?.pid) terminateProcessTree(activeChildProcess.pid);
    process.exit(sig === 'SIGINT' ? 130 : 143);
  };
  process.on('SIGINT', () => exit('SIGINT'));
  process.on('SIGTERM', () => exit('SIGTERM'));
  signalHandlersInstalled = true;
}

function prependPathEntry(env, pathEntry) {
  if (!pathEntry) return;
  const sep = isWindows ? ';' : ':';
  const currentPath = getEnvValue(env, 'PATH');
  if (splitPathEntries(currentPath).some((entry) => pathsMatch(entry, pathEntry))) {
    return;
  }

  setEnvValue(env, 'PATH', currentPath ? `${pathEntry}${sep}${currentPath}` : pathEntry);
}

async function runCommand(executable, args, env) {
  log(`run: ${executable} ${args.join(' ')}`);
  installSignalHandlers();

  const child = spawn(executable, args, {
    cwd: projectRoot, stdio: 'inherit', shell: false, windowsHide: true, env, detached: !isWindows,
  });
  activeChildProcess = child;

  try {
    await new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('exit', (code, signal) => {
        if (signal) return reject(new Error(`Terminated by ${signal}: ${executable}`));
        if (code !== 0) return reject(new Error(`Exit code ${code}: ${executable}`));
        resolve();
      });
    });
  } finally {
    if (activeChildProcess === child) activeChildProcess = null;
  }
}

async function runCommandWithExitCode(executable, args, env) {
  log(`run: ${executable} ${args.join(' ')}`);
  installSignalHandlers();

  const child = spawn(executable, args, {
    cwd: projectRoot, stdio: 'inherit', shell: false, windowsHide: true, env, detached: !isWindows,
  });
  activeChildProcess = child;

  try {
    return await new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('exit', (code, signal) => {
        if (signal) return reject(new Error(`Terminated by ${signal}: ${executable}`));
        resolve(code ?? 0);
      });
    });
  } finally {
    if (activeChildProcess === child) activeChildProcess = null;
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────

const forwardedArgs = parseForwardedArgs(Bun.argv.slice(2));

// 1. Verify prerequisites and resolve toolchains
const cudaToolkit = resolveCudaToolkit();
const msvcDetails = captureMsvcEnvironment();

// 2. Resolve build tools
const buildConfig = resolveBuildConfiguration();
log(`build_type=${buildType} generator=${buildConfig.generator}${buildConfig.makeProgram ? ` make_program=${buildConfig.makeProgram}` : ''}`);

// 3. Invalidate stale build cache
removeInvalidBuildDirectory(buildConfig.generator, cudaToolkit, msvcDetails);

// 4. Build environment: current env + MSVC dev env (INCLUDE, LIB, PATH for cl/link)
const buildEnv = mergeEnvironment({ ...process.env }, msvcDetails.environment);
prependPathEntry(buildEnv, buildConfig.makeProgram ? path.dirname(buildConfig.makeProgram) : null);

// Ensure CUDA runtime DLLs (cudart64, cublas64, cublasLt64, etc.) are on PATH
setEnvValue(buildEnv, 'CUDA_PATH', cudaToolkit.toolkitRoot);
setEnvValue(buildEnv, 'CUDACXX', cudaToolkit.nvccPath);
prependPathEntry(buildEnv, cudaToolkit.binDir);

// 5. CMake configure
const cmake = resolveCmakeExecutable();
const configureArgs = [
  '-S', normalizeHostPath(llamaCppRoot),
  '-B', buildDirName,
  '-G', buildConfig.generator,
  `-DCMAKE_BUILD_TYPE=${buildType}`,
  `-DCUDAToolkit_ROOT=${normalizeHostPath(cudaToolkit.toolkitRoot)}`,
  `-DCMAKE_CUDA_COMPILER=${normalizeHostPath(cudaToolkit.nvccPath)}`,
  '-DGGML_CUDA=ON',
  '-DBUILD_TESTING=ON',
  '-DLLAMA_BUILD_TESTS=ON',
  '-DLLAMA_BUILD_EXAMPLES=OFF',
  '-DLLAMA_BUILD_SERVER=OFF',
  `-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${normalizeHostPath(buildOutputDir)}`,
  `-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE=${normalizeHostPath(buildOutputDir)}`,
  `-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG=${normalizeHostPath(buildOutputDir)}`,
];
if (msvcDetails.clPath) {
  configureArgs.push(`-DCMAKE_CUDA_HOST_COMPILER=${normalizeHostPath(msvcDetails.clPath)}`);
}
if (buildConfig.makeProgram) {
  configureArgs.push(`-DCMAKE_MAKE_PROGRAM=${buildConfig.makeProgram}`);
}

await runCommand(cmake, configureArgs, buildEnv);

// 6. Build
await runCommand(cmake, ['--build', buildDirName, '--config', buildType, '--target', testTargetName], buildEnv);

// 7. Run
const executablePath = path.join(buildOutputDir, testExecutableName);
if (!existsSync(executablePath)) {
  throw new Error(`Missing build artifact: ${executablePath}`);
}

const exitCode = await runCommandWithExitCode(executablePath, forwardedArgs, buildEnv);
process.exit(exitCode);
