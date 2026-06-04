import http from 'node:http';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5173;
const DEFAULT_TIMEOUT_MS = 30_000;

function parseArgs(argv) {
  const options = {
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    requireRustEngine: false,
    requireGgufIngest: false,
    requireWebgpu: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--host') {
      options.host = readValue(argv, index, arg);
      index += 1;
    } else if (arg === '--port') {
      options.port = parsePort(readValue(argv, index, arg));
      index += 1;
    } else if (arg === '--timeout-ms') {
      options.timeoutMs = parsePositiveInt(readValue(argv, index, arg), arg);
      index += 1;
    } else if (arg === '--require-rust-engine' || arg === '--require-rust-browser-engine') {
      options.requireRustEngine = true;
    } else if (arg === '--require-gguf-ingest') {
      options.requireGgufIngest = true;
    } else if (arg === '--require-webgpu') {
      options.requireWebgpu = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function readValue(argv, index, flag) {
  const value = argv[index + 1];
  if (value == null || value.startsWith('--')) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
}

function parsePort(value) {
  const port = parsePositiveInt(value, '--port');
  if (port > 65_535) {
    throw new Error(`--port must be <= 65535, got ${value}`);
  }
  return port;
}

function parsePositiveInt(value, flag) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed <= 0 || String(parsed) !== value) {
    throw new Error(`${flag} must be a positive integer, got ${value}`);
  }
  return parsed;
}

function benchmarkDir() {
  const scriptDir = path.dirname(fileURLToPath(import.meta.url));
  return path.resolve(scriptDir, '..');
}

function serverUrl(options) {
  return `http://${options.host}:${options.port}`;
}

async function waitForServer(url, timeoutMs) {
  const started = Date.now();
  let lastError = null;
  while (Date.now() - started < timeoutMs) {
    try {
      const status = await httpStatus(url);
      if (status >= 200 && status < 500) {
        return true;
      }
    } catch (error) {
      lastError = error;
    }
    await sleep(250);
  }
  if (lastError != null) {
    throw lastError;
  }
  return false;
}

function httpStatus(url) {
  return new Promise((resolve, reject) => {
    const request = http.get(url, (response) => {
      response.resume();
      response.on('end', () => resolve(response.statusCode ?? 0));
    });
    request.setTimeout(1_000, () => {
      request.destroy(new Error(`Timed out connecting to ${url}`));
    });
    request.on('error', reject);
  });
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function withTimeout(promise, timeoutMs, label) {
  let timer = null;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error(`${label} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => {
    if (timer != null) {
      clearTimeout(timer);
    }
  });
}

function bunxCommand() {
  if (process.platform !== 'win32') {
    return 'bunx';
  }

  const home = process.env.USERPROFILE;
  if (home != null) {
    const bunx = path.join(home, '.bun', 'bin', 'bunx.exe');
    if (existsSync(bunx)) {
      return bunx;
    }
  }
  return 'bunx.exe';
}

function startVite(options) {
  const command = bunxCommand();
  const child = spawn(
    command,
    ['--bun', 'vite', '--host', options.host, '--port', String(options.port), '--strictPort'],
    {
      cwd: benchmarkDir(),
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    }
  );

  child.stdout.on('data', (chunk) => {
    process.stderr.write(chunk);
  });
  child.stderr.on('data', (chunk) => {
    process.stderr.write(chunk);
  });

  return child;
}

async function closeServer(child) {
  if (child == null || child.exitCode != null) {
    return;
  }

  child.kill();
  await new Promise((resolve) => {
    const timer = setTimeout(resolve, 3_000);
    child.once('exit', () => {
      clearTimeout(timer);
      resolve();
    });
  });
}

async function ensureServer(options) {
  const url = serverUrl(options);
  try {
    if (await waitForServer(url, 1_000)) {
      return { url, child: null };
    }
  } catch {
    // No existing server; start a local Vite process below.
  }

  const child = startVite(options);
  try {
    await waitForServer(url, options.timeoutMs);
    return { url, child };
  } catch (error) {
    await closeServer(child);
    throw new Error(`Benchmark server did not start at ${url}: ${error.message}`);
  }
}

function validateSmoke(result, options) {
  const failures = [];
  if (options.requireRustEngine && !result?.rustEngine?.available) {
    failures.push(`Rust browser engine unavailable: ${result?.rustEngine?.error ?? 'unknown error'}`);
  }
  if (options.requireGgufIngest && !result?.ggufIngest?.available) {
    failures.push(`GGUF ingest unavailable: ${result?.ggufIngest?.error ?? 'unknown error'}`);
  }
  if (options.requireWebgpu && !result?.webgpuReady) {
    failures.push('WebGPU backend is not ready');
  }
  if (failures.length > 0) {
    throw new Error(failures.join('; '));
  }
}

async function runBrowserSmoke(options) {
  const { url, child } = await ensureServer(options);
  let browser = null;
  try {
    browser = await withTimeout(
      chromium.launch({ headless: true }),
      options.timeoutMs,
      'Chromium launch'
    );
    const page = await browser.newPage();
    page.setDefaultTimeout(options.timeoutMs);
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: options.timeoutMs });
    await page.waitForFunction(() => window.__cogentBench != null, null, {
      timeout: options.timeoutMs,
    });
    const result = await withTimeout(
      page.evaluate(() => window.__cogentBench.runBrowserRuntimeSmoke()),
      options.timeoutMs,
      'Browser runtime smoke'
    );
    validateSmoke(result, options);
    return {
      url,
      result,
    };
  } finally {
    await browser?.close();
    await closeServer(child);
  }
}

try {
  const options = parseArgs(process.argv.slice(2));
  const report = await runBrowserSmoke(options);
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
