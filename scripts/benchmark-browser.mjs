import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawn } from 'node:child_process';

import { chromium } from 'playwright-core';

const repoRoot = process.cwd();
const defaultHost = '127.0.0.1';
const defaultPort = 4173;
const defaultPrompt = 'Describe how to benchmark browser-hosted inference.';
const defaultTokens = 64;
const defaultWarmup = 1;
const defaultRuns = 3;

function printHelp() {
  console.log(`Usage: bun run bench:browser --model <path> [options]

Options:
  --model <path>              Path to local GGUF model file
  --browser <name>            chrome | msedge (default: chrome)
  --executable <path>         Explicit browser executable path
  --host <host>               Preview host (default: ${defaultHost})
  --port <port>               Preview port (default: ${defaultPort})
  --prompt <text>             Prompt text
  --tokens <n>                Max generation tokens (default: ${defaultTokens})
  --warmup <n>                Warmup runs (default: ${defaultWarmup})
  --runs <n>                  Measured runs (default: ${defaultRuns})
  --output <path>             JSON output path
  --skip-build <true|false>   Skip package/app build (default: false)
  --headed <true|false>       Launch headed browser (default: true)
  --require-adapter <text>    Fail unless adapter label/vendor/description contains this text
  --force-high-performance-gpu <true|false>
                              Pass Chromium --force-high-performance-gpu (default: false)
  --help                      Show this message
`);
}

function parseBoolean(flagName, rawValue, fallback) {
  if (rawValue == null) {
    return fallback;
  }
  if (rawValue === 'true') {
    return true;
  }
  if (rawValue === 'false') {
    return false;
  }
  throw new Error(`Expected "true" or "false" for ${flagName}, got "${rawValue}".`);
}

function parsePositiveInt(flagName, rawValue, fallback) {
  if (rawValue == null) {
    return fallback;
  }
  const value = Number.parseInt(rawValue, 10);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Expected a positive integer for ${flagName}, got "${rawValue}".`);
  }
  return value;
}

function parseArgs(argv) {
  const values = new Map();

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith('--')) {
      throw new Error(`Unexpected positional argument "${arg}".`);
    }
    const key = arg.slice(2);
    if (key === 'help') {
      printHelp();
      process.exit(0);
    }
    const value = argv[i + 1];
    if (!value || value.startsWith('--')) {
      throw new Error(`Missing value for --${key}.`);
    }
    values.set(key, value);
    i += 1;
  }

  const modelPath = values.get('model');
  if (!modelPath) {
    throw new Error('Missing required --model <path>.');
  }

  const resolvedModelPath = path.isAbsolute(modelPath) ? modelPath : path.resolve(repoRoot, modelPath);
  if (!existsSync(resolvedModelPath)) {
    throw new Error(`Model file does not exist: ${resolvedModelPath}`);
  }

  const browser = (values.get('browser') ?? 'chrome').trim();
  if (browser !== 'chrome' && browser !== 'msedge') {
    throw new Error(`Unsupported browser "${browser}". Use "chrome" or "msedge".`);
  }

  return {
    modelPath: resolvedModelPath,
    browser,
    executablePath: values.get('executable') ?? null,
    host: values.get('host') ?? defaultHost,
    port: parsePositiveInt('--port', values.get('port'), defaultPort),
    prompt: values.get('prompt') ?? defaultPrompt,
    tokenCount: parsePositiveInt('--tokens', values.get('tokens'), defaultTokens),
    warmupRuns: parsePositiveInt('--warmup', values.get('warmup'), defaultWarmup),
    measuredRuns: parsePositiveInt('--runs', values.get('runs'), defaultRuns),
    outputPath: values.get('output')
      ? (path.isAbsolute(values.get('output')) ? values.get('output') : path.resolve(repoRoot, values.get('output')))
      : path.resolve(repoRoot, 'benchmarks', 'browser', 'latest-browser.json'),
    skipBuild: parseBoolean('--skip-build', values.get('skip-build'), false),
    headed: parseBoolean('--headed', values.get('headed'), true),
    requireAdapter: values.get('require-adapter') ?? null,
    forceHighPerformanceGpu: parseBoolean(
      '--force-high-performance-gpu',
      values.get('force-high-performance-gpu'),
      false
    ),
  };
}

function detectBrowserExecutable(browser) {
  const candidates =
    browser === 'msedge'
      ? [
          path.join(process.env['ProgramFiles(x86)'] ?? '', 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
          path.join(process.env.ProgramFiles ?? '', 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
        ]
      : [
          path.join(process.env.ProgramFiles ?? '', 'Google', 'Chrome', 'Application', 'chrome.exe'),
          path.join(process.env['ProgramFiles(x86)'] ?? '', 'Google', 'Chrome', 'Application', 'chrome.exe'),
        ];

  for (const candidate of candidates) {
    if (candidate && existsSync(candidate)) {
      return candidate;
    }
  }

  throw new Error(`Could not find a ${browser} executable. Pass --executable <path>.`);
}

function runCommand(command, args, workdir) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: workdir,
      stdio: 'inherit',
      shell: false,
    });

    child.on('error', reject);
    child.on('exit', (code, signal) => {
      if (signal) {
        reject(new Error(`Command terminated by signal ${signal}: ${command} ${args.join(' ')}`));
        return;
      }
      if (code !== 0) {
        reject(new Error(`Command failed with code ${code}: ${command} ${args.join(' ')}`));
        return;
      }
      resolve();
    });
  });
}

async function waitForHttpReady(url, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return;
      }
    } catch {
      // server not ready yet
    }
    await Bun.sleep(250);
  }
  throw new Error(`Timed out waiting for preview server at ${url}`);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const previewUrl = `http://${options.host}:${options.port}`;

  if (!options.skipBuild) {
    await runCommand('bun', ['run', 'build:package'], repoRoot);
    await runCommand('bun', ['run', '--filter=cogent-engine-benchmark-app', 'build'], repoRoot);
  }

  const preview = spawn(
    'bun',
    [
      'run',
      '--filter=cogent-engine-benchmark-app',
      'preview',
      '--',
      '--host',
      options.host,
      '--port',
      String(options.port),
      '--strictPort',
    ],
    {
      cwd: repoRoot,
      stdio: 'inherit',
      shell: false,
    }
  );

  let userDataDir = null;
  let context = null;

  try {
    await waitForHttpReady(previewUrl);

    userDataDir = await mkdtemp(path.join(os.tmpdir(), 'cogent-browser-bench-'));
    const executablePath = options.executablePath ?? detectBrowserExecutable(options.browser);
    const launchArgs = [];
    if (options.forceHighPerformanceGpu) {
      launchArgs.push('--force-high-performance-gpu');
    }

    context = await chromium.launchPersistentContext(userDataDir, {
      executablePath,
      headless: !options.headed,
      args: launchArgs,
      viewport: { width: 1440, height: 1200 },
    });

    const page = await context.newPage();
    await page.goto(previewUrl, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => typeof window.__cogentBench === 'object');
    await page.setInputFiles('#modelFile', options.modelPath);

    const environment = await page.evaluate(async () => {
      return await window.__cogentBench.collectEnvironmentInfo(true);
    });

    if (options.requireAdapter) {
      const haystack = [
        environment?.adapterLabel,
        environment?.adapterVendor,
        environment?.adapterDescription,
        environment?.adapterArchitecture,
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
      if (!haystack.includes(options.requireAdapter.toLowerCase())) {
        throw new Error(`Selected adapter does not match "${options.requireAdapter}". Actual: ${haystack || 'unknown'}`);
      }
    }

    const report = await page.evaluate(async (config) => {
      return await window.__cogentBench.runBenchmark(config);
    }, {
      prompt: options.prompt,
      tokenCount: options.tokenCount,
      warmupRuns: options.warmupRuns,
      measuredRuns: options.measuredRuns,
    });

    const output = {
      ...report,
      automation: {
        runner: 'scripts/benchmark-browser.mjs',
        browser: options.browser,
        executablePath,
        headed: options.headed,
        previewUrl,
        launchArgs,
        modelFile: path.basename(options.modelPath),
        requireAdapter: options.requireAdapter,
      },
    };

    await mkdir(path.dirname(options.outputPath), { recursive: true });
    await Bun.write(options.outputPath, `${JSON.stringify(output, null, 2)}\n`);

    console.log('\nBrowser benchmark complete');
    console.log(`  output          ${options.outputPath}`);
    console.log(`  adapter         ${environment?.adapterLabel ?? 'unknown'}`);
    console.log(`  runtime backend ${report?.backend?.runtimeBackendStatus ?? 'unknown'}`);
    console.log(`  execution       ${report?.backend?.inferredExecutionBackend ?? 'unknown'}`);
  } finally {
    await context?.close();
    if (preview.pid) {
      preview.kill('SIGTERM');
    }
    if (userDataDir) {
      await rm(userDataDir, { recursive: true, force: true });
    }
  }
}

await main();
