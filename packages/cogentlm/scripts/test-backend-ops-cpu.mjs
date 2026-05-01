import { spawn, spawnSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, rmSync, writeFileSync, unlinkSync, mkdtempSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// ── Constants ────────────────────────────────────────────────────────────────

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const llamaCppRoot = path.join(projectRoot, 'third_party', 'llama.cpp');
const buildLabel = process.env.CE_TEST_BACKEND_OPS_BUILD_LABEL?.trim() || '[test-backend-ops:cpu]';
const buildType = process.env.CE_TEST_BACKEND_OPS_BUILD_TYPE?.trim() || 'Release';
const isDebugBuild = buildType.toLowerCase() === 'debug';
const buildDirName =
  process.env.CE_TEST_BACKEND_OPS_BUILD_DIR_NAME?.trim() ||
  (isDebugBuild ? 'build-test-backend-ops-cpu-debug' : 'build-test-backend-ops-cpu');
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
  if (!pathValue) return [];
  const separator = isWindows ? ';' : ':';
  return pathValue.split(separator).map((e) => e.trim().replace(/^"(.*)"$/, '$1')).map((e) => normalizeHostPath(e)).filter(Boolean);
}

function findProgramInPathEntries(programName, pathValue) {
  for (const entry of splitPathEntries(pathValue)) {
    const candidate = normalizeHostPath(path.join(entry, programName));
    if (existsSync(candidate)) return candidate;
  }
  return null;
}

// ── MSVC Environment Detection (Faithfully from CUDA script) ─────────────────

function resolveVswhereExecutable() {
  if (!isWindows) return null;
  const vswherePath = path.join(process.env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)', 'Microsoft Visual Studio', 'Installer', 'vswhere.exe');
  return existsSync(vswherePath) ? normalizeHostPath(vswherePath) : null;
}

function getVisualStudioMetadataFromPath(inputPath) {
  if (!isWindows || !inputPath) return null;
  const normalizedPath = normalizeHostPath(inputPath);
  const parts = normalizedPath.split('\\');
  const vsIndex = parts.findIndex((p) => p.toLowerCase() === 'microsoft visual studio');
  if (vsIndex < 0 || vsIndex + 1 >= parts.length) return null;
  const versionToken = parts[vsIndex + 1];
  return {
    versionToken,
    year: visualStudioTokenToYear.get(versionToken) ?? versionToken,
    edition: parts[vsIndex + 2] ?? null,
  };
}

function getVisualStudioInstallationPathFromCompilerPath(compilerPath) {
  if (!isWindows || !compilerPath) return null;
  const normalizedPath = normalizeHostPath(compilerPath);
  const marker = `${path.win32.sep}VC${path.win32.sep}Tools${path.win32.sep}MSVC${path.win32.sep}`;
  const markerIndex = normalizedPath.toLowerCase().indexOf(marker.toLowerCase());
  if (markerIndex < 0) return null;
  return normalizedPath.slice(0, markerIndex);
}

function toVisualStudioInstallationInfo(installationPath) {
  const normalizedPath = normalizeHostPath(installationPath);
  const metadata = getVisualStudioMetadataFromPath(normalizedPath);
  if (!normalizedPath || !metadata) return null;
  const vcvarsallPath = normalizeHostPath(path.join(normalizedPath, 'VC', 'Auxiliary', 'Build', 'vcvarsall.bat'));
  if (!existsSync(vcvarsallPath)) return null;
  return { installationPath: normalizedPath, vcvarsallPath, year: metadata.year, edition: metadata.edition };
}

function findSupportedVisualStudioInstallation() {
  if (!isWindows) return null;
  const vswherePath = resolveVswhereExecutable();
  if (vswherePath) {
    for (const release of supportedVisualStudioReleases) {
      const result = spawnSync(vswherePath, ['-latest', '-products', '*', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-version', release.versionRange, '-format', 'json'], { encoding: 'utf8', windowsHide: true });
      if (result.status === 0 && result.stdout?.trim()) {
        try {
          const matches = JSON.parse(result.stdout);
          const installation = toVisualStudioInstallationInfo(matches[0]?.installationPath?.trim());
          if (installation) return installation;
        } catch {}
      }
    }
  }
  return null;
}

function captureVcvarsallEnvironment(vcvarsallPath) {
  log(`sourcing VS environment: ${vcvarsallPath}`);
  const tmpDir = mkdtempSync(path.join(os.tmpdir(), 'ce-vcvars-'));
  const tmpBat = path.join(tmpDir, 'capture_env.bat');
  writeFileSync(tmpBat, `@call "${vcvarsallPath}" x64\r\n@if errorlevel 1 exit /b 1\r\n@set\r\n`);
  const result = spawnSync('cmd.exe', ['/C', tmpBat], { encoding: 'utf8', windowsHide: true });
  try { unlinkSync(tmpBat); rmSync(tmpDir, { recursive: true, force: true }); } catch {}
  if (result.status !== 0) throw new Error('Failed to source vcvarsall.bat.');
  const env = {};
  for (const line of result.stdout.split(/\r?\n/)) {
    const eq = line.indexOf('=');
    if (eq > 0) env[line.substring(0, eq)] = line.substring(eq + 1);
  }
  return env;
}

function resolveAmbientVisualStudioCompiler() {
  if (!isWindows) return null;
  const clPath = findProgramsOnPath('cl.exe').map((c) => normalizeHostPath(c)).find(Boolean);
  if (!clPath) return null;
  const metadata = getVisualStudioMetadataFromPath(clPath);
  return { clPath, installationPath: normalizeHostPath(getVisualStudioInstallationPathFromCompilerPath(clPath)), year: metadata?.year ?? null, edition: metadata?.edition ?? null };
}

function captureMsvcEnvironment() {
  if (!isWindows) return { environment: {}, clPath: null };
  const installation = findSupportedVisualStudioInstallation();
  if (!installation) {
    const ambient = resolveAmbientVisualStudioCompiler();
    if (ambient?.clPath) return { environment: {}, ...ambient };
    throw new Error('No supported Visual Studio installation found.');
  }
  const environment = captureVcvarsallEnvironment(installation.vcvarsallPath);
  const clPath = findProgramInPathEntries('cl.exe', getEnvValue(environment, 'PATH'));
  return { environment, clPath };
}

// ── Build Configuration ─────────────────────────────────────────────────────

function inferGeneratorFromMakeProgram(makeProgramPath) {
  const lower = makeProgramPath.toLowerCase();
  if (lower.includes('ninja')) return 'Ninja';
  if (lower.includes('nmake')) return 'NMake Makefiles';
  if (lower.endsWith('make') || lower.endsWith('make.exe')) return 'Unix Makefiles';
  return null;
}

function detectNinja() {
  const ninjaName = isWindows ? 'ninja.exe' : 'ninja';
  for (const cmakePath of findProgramsOnPath(isWindows ? 'cmake.exe' : 'cmake')) {
    const candidate = path.join(path.dirname(cmakePath), ninjaName);
    if (existsSync(candidate)) return normalizeHostPath(candidate);
  }
  if (isWindows) {
    const vsRoot = path.join(process.env.ProgramFiles ?? 'C:\\Program Files', 'Microsoft Visual Studio');
    if (existsSync(vsRoot)) {
      for (const year of ['2022', '2019', '2017']) {
        const yearDir = path.join(vsRoot, year);
        if (!existsSync(yearDir)) continue;
        for (const entry of readdirSync(yearDir, { withFileTypes: true })) {
          if (!entry.isDirectory()) continue;
          const candidate = path.join(yearDir, entry.name, 'Common7', 'IDE', 'CommonExtensions', 'Microsoft', 'CMake', 'Ninja', ninjaName);
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
  if (generatorFromEnv) return { generator: generatorFromEnv, makeProgram: makeProgramFromEnv || null };
  if (makeProgramFromEnv) {
    const gen = inferGeneratorFromMakeProgram(makeProgramFromEnv);
    if (gen) return { generator: gen, makeProgram: makeProgramFromEnv };
  }
  const ninja = detectNinja();
  if (ninja) return { generator: 'Ninja', makeProgram: ninja };
  if (commandAvailable('ninja')) return { generator: 'Ninja', makeProgram: null };
  if (isWindows && commandAvailable('nmake', ['/?'])) return { generator: 'NMake Makefiles', makeProgram: null };
  if (commandAvailable('make')) return { generator: 'Unix Makefiles', makeProgram: null };
  throw new Error('No supported CMake generator found.');
}

function removeInvalidBuildDirectory(expectedGenerator) {
  if (!existsSync(buildDir)) return;
  if (!existsSync(path.join(buildDir, 'CMakeCache.txt'))) {
    rmSync(buildDir, { recursive: true, force: true });
    return;
  }
  const cacheText = readFileSync(path.join(buildDir, 'CMakeCache.txt'), 'utf8');
  const reasons = [];
  if (getCacheEntry(cacheText, 'CMAKE_GENERATOR') !== expectedGenerator) reasons.push('generator mismatch');
  if (getCacheEntry(cacheText, 'CMAKE_BUILD_TYPE') !== buildType) reasons.push('build type mismatch');
  if (reasons.length > 0) {
    log(`Removing stale build directory (${reasons.join(', ')})`);
    rmSync(buildDir, { recursive: true, force: true });
  }
}

// ── Process management ──────────────────────────────────────────────────────

function installSignalHandlers() {
  if (signalHandlersInstalled) return;
  const exit = (sig) => {
    if (activeChildProcess?.pid) {
      if (isWindows) spawnSync('taskkill.exe', ['/T', '/F', '/PID', String(activeChildProcess.pid)], { windowsHide: true });
      else activeChildProcess.kill(sig);
    }
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
  setEnvValue(env, 'PATH', currentPath ? `${pathEntry}${sep}${currentPath}` : pathEntry);
}

async function runCommand(executable, args, env) {
  log(`run: ${executable} ${args.join(' ')}`);
  installSignalHandlers();
  const child = spawn(executable, args, { cwd: projectRoot, stdio: 'inherit', env, windowsHide: true });
  activeChildProcess = child;
  return new Promise((resolve, reject) => {
    child.on('exit', (code) => { activeChildProcess = null; if (code === 0) resolve(); else reject(new Error(`Command failed with code ${code}`)); });
    child.on('error', reject);
  });
}

async function runCommandWithExitCode(executable, args, env) {
  log(`run: ${executable} ${args.join(' ')}`);
  installSignalHandlers();
  const child = spawn(executable, args, { cwd: projectRoot, stdio: 'inherit', env, windowsHide: true });
  activeChildProcess = child;
  return new Promise((resolve) => {
    child.on('exit', (code) => { activeChildProcess = null; resolve(code ?? 0); });
  });
}

// ── Main ─────────────────────────────────────────────────────────────────────

const forwardedArgs = Bun.argv.slice(2);
if (forwardedArgs.includes('--help') || forwardedArgs.includes('-h')) {
  console.log(`Usage: bun ./scripts/test-backend-ops-cpu.mjs [test-backend-ops args]`);
  process.exit(0);
}

// Ensure -b CPU is present if no backend is selected
if (!forwardedArgs.includes('-b')) {
  forwardedArgs.unshift('CPU');
  forwardedArgs.unshift('-b');
}

const msvcDetails = captureMsvcEnvironment();
const buildConfig = resolveBuildConfiguration();
const buildEnv = mergeEnvironment({ ...process.env }, msvcDetails.environment);
prependPathEntry(buildEnv, buildConfig.makeProgram ? path.dirname(buildConfig.makeProgram) : null);

removeInvalidBuildDirectory(buildConfig.generator);

const cmake = resolveCmakeExecutable();
const configureArgs = [
  '-S', normalizeHostPath(llamaCppRoot),
  '-B', buildDirName,
  '-G', buildConfig.generator,
  `-DCMAKE_BUILD_TYPE=${buildType}`,
  '-DGGML_CUDA=OFF',
  '-DGGML_WEBGPU=OFF',
  '-DBUILD_TESTING=ON',
  '-DLLAMA_BUILD_TESTS=ON',
  '-DLLAMA_BUILD_EXAMPLES=OFF',
  '-DLLAMA_BUILD_SERVER=OFF',
  `-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${normalizeHostPath(buildOutputDir)}`,
];

if (msvcDetails.clPath) {
  configureArgs.push(`-DCMAKE_ASM_COMPILER=${normalizeHostPath(msvcDetails.clPath)}`);
}
if (buildConfig.makeProgram) {
  configureArgs.push(`-DCMAKE_MAKE_PROGRAM=${buildConfig.makeProgram}`);
}

await runCommand(cmake, configureArgs, buildEnv);
await runCommand(cmake, ['--build', buildDirName, '--config', buildType, '--target', testTargetName], buildEnv);

const executablePath = path.join(buildOutputDir, testExecutableName);
if (!existsSync(executablePath)) throw new Error(`Missing build artifact: ${executablePath}`);

const exitCode = await runCommandWithExitCode(executablePath, forwardedArgs, buildEnv);
process.exit(exitCode);
