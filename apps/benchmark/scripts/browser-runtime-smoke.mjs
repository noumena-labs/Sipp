#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_TIMEOUT_MS = 60_000;

const appDir = fileURLToPath(new URL('..', import.meta.url));
const args = parseArgs(process.argv.slice(2));
const host = args.host ?? DEFAULT_HOST;
const port = args.port ?? await findOpenPort(host);
const timeoutMs = args.timeoutMs ?? DEFAULT_TIMEOUT_MS;
const url = `http://${host}:${port}`;

let serverProcess = null;

try {
  serverProcess = startViteServer(host, port);
  await waitForServerOrExit(serverProcess, url, timeoutMs);

  const { chromium } = await loadPlaywright();
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    page.on('console', (message) => {
      console.log(`[browser:${message.type()}] ${message.text()}`);
    });
    page.on('pageerror', (error) => {
      console.error(`[browser:error] ${error.message}`);
    });

    await page.goto(url, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => typeof window.__cogentBench?.runBrowserRuntimeSmoke === 'function',
      undefined,
      { timeout: timeoutMs }
    );

    const result = await page.evaluate(async () => {
      return await window.__cogentBench.runBrowserRuntimeSmoke();
    });

    console.log(JSON.stringify(result, null, 2));
    assertSmokeResult(result, args);
  } finally {
    await browser.close();
  }
} finally {
  await stopServer(serverProcess);
}

function parseArgs(values) {
  const parsed = {
    requireRustBrowserEngine: false,
    requireGgufIngest: false,
    host: null,
    port: null,
    timeoutMs: null,
  };

  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value === '--require-rust-browser-engine') {
      parsed.requireRustBrowserEngine = true;
    } else if (value === '--require-gguf-ingest') {
      parsed.requireGgufIngest = true;
    } else if (value === '--host') {
      parsed.host = readValue(values, ++index, value);
    } else if (value === '--port') {
      parsed.port = readNumber(values, ++index, value);
    } else if (value === '--timeout-ms') {
      parsed.timeoutMs = readNumber(values, ++index, value);
    } else {
      throw new Error(`Unknown argument: ${value}`);
    }
  }

  return parsed;
}

function readValue(values, index, flag) {
  const value = values[index];
  if (value == null || value.startsWith('--')) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
}

function readNumber(values, index, flag) {
  const value = Number(readValue(values, index, flag));
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${flag} must be a positive integer`);
  }
  return value;
}

function startViteServer(host, port) {
  const child = spawn(
    'bunx',
    ['--bun', 'vite', '--host', host, '--port', String(port), '--strictPort'],
    {
      cwd: appDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    }
  );

  child.stdout.on('data', (chunk) => {
    process.stdout.write(`[vite] ${chunk}`);
  });
  child.stderr.on('data', (chunk) => {
    process.stderr.write(`[vite] ${chunk}`);
  });
  child.once('exit', (code, signal) => {
    if (code !== 0 && code != null) {
      console.error(`[vite] exited with code ${code}`);
    }
    if (signal != null) {
      console.error(`[vite] exited with signal ${signal}`);
    }
  });

  return child;
}

async function waitForServer(targetUrl, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(targetUrl);
      if (response.ok) {
        return;
      }
    } catch {
      // Vite is still starting.
    }
    await delay(250);
  }

  throw new Error(`Timed out waiting for Vite at ${targetUrl}`);
}

function waitForServerOrExit(child, targetUrl, timeoutMs) {
  return Promise.race([
    waitForServer(targetUrl, timeoutMs),
    new Promise((_, reject) => {
      child.once('error', reject);
      child.once('exit', (code, signal) => {
        reject(
          new Error(`Vite exited before ${targetUrl} was ready: code=${code} signal=${signal}`)
        );
      });
    }),
  ]);
}

async function loadPlaywright() {
  try {
    return await import('playwright');
  } catch {
    try {
      return await import('playwright-core');
    } catch (error) {
      throw new Error(
        'Playwright is required for browser runtime smoke tests. Run `bun install` at the workspace root.',
        { cause: error }
      );
    }
  }
}

function assertSmokeResult(result, options) {
  if (options.requireRustBrowserEngine && !result?.rustEngine?.available) {
    const error = result?.rustEngine?.error ?? 'Rust browser engine smoke was unavailable';
    throw new Error(error);
  }

  if (options.requireGgufIngest && !result?.ggufIngest?.available) {
    const error = result?.ggufIngest?.error ?? 'GGUF ingest smoke was unavailable';
    throw new Error(error);
  }
}

async function stopServer(child) {
  if (child == null || child.exitCode != null) {
    return;
  }

  child.kill();
  await Promise.race([
    new Promise((resolve) => child.once('exit', resolve)),
    delay(2_000),
  ]);
  if (child.exitCode == null) {
    child.kill('SIGKILL');
  }
}

function findOpenPort(host) {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.once('error', reject);
    server.listen(0, host, () => {
      const address = server.address();
      server.close(() => {
        if (address == null || typeof address === 'string') {
          reject(new Error('Failed to reserve a local port'));
        } else {
          resolve(address.port);
        }
      });
    });
  });
}
