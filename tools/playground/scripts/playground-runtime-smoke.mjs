import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import {
  closeViteServer,
  ensureViteServer,
  withTimeout,
} from '../../browser-smoke/vite-server.mjs';

const PLAYGROUND_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5173;
const DEFAULT_TIMEOUT_MS = 30_000;

function parseArgs(argv) {
  const options = {
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
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

function validateProbe(result, options) {
  const failures = [];
  if (options.requireWebgpu) {
    if (!result.environment?.hasNavigatorGpu) {
      failures.push('navigator.gpu is unavailable');
    } else if (!result.environment?.adapterAvailable) {
      failures.push('WebGPU adapter is unavailable');
    }
  }
  if (failures.length > 0) {
    throw new Error(failures.join('; '));
  }
}

async function runBrowserProbe(options) {
  const { url, child } = await ensureViteServer({
    rootDir: PLAYGROUND_DIR,
    host: options.host,
    port: options.port,
    timeoutMs: options.timeoutMs,
    label: 'Playground',
  });
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
    await page.waitForFunction(() => window.__sippPlayground != null, null, {
      timeout: options.timeoutMs,
    });
    const result = await withTimeout(
      page.evaluate(async () => {
        const api = window.__sippPlayground;
        return {
          environment: await api.getEnvironment(),
          observability: api.getRuntimeObservability(),
          backend: api.getBackendObservability(),
        };
      }),
      options.timeoutMs,
      'Playground browser probe'
    );
    validateProbe(result, options);
    return {
      url,
      result,
    };
  } finally {
    await browser?.close();
    await closeViteServer(child);
  }
}

try {
  const options = parseArgs(process.argv.slice(2));
  const report = await runBrowserProbe(options);
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
