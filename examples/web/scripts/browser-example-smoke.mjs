import http from 'node:http';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5174;
const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_TOKENS = 64;
const DEFAULT_PROMPT = 'Describe browser LLM inference.';
const DEFAULT_CASES = ['query', 'chat', 'embed'];
const CASE_PAGES = new Map([
  ['query', 'query.html'],
  ['chat', 'chat.html'],
  ['embed', 'embed.html'],
]);

function parseArgs(argv) {
  const options = {
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    model: null,
    prompt: DEFAULT_PROMPT,
    maxTokens: DEFAULT_MAX_TOKENS,
    cases: [],
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
    } else if (arg === '--model') {
      options.model = readValue(argv, index, arg);
      index += 1;
    } else if (arg === '--prompt') {
      options.prompt = readValue(argv, index, arg);
      index += 1;
    } else if (arg === '--max-tokens') {
      options.maxTokens = parsePositiveInt(readValue(argv, index, arg), arg);
      index += 1;
    } else if (arg === '--case') {
      options.cases.push(parseCase(readValue(argv, index, arg)));
      index += 1;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (options.model == null) {
    throw new Error('--model requires a GGUF file path');
  }
  if (!existsSync(options.model)) {
    throw new Error(`model file does not exist: ${options.model}`);
  }
  if (options.cases.length === 0) {
    options.cases = [...DEFAULT_CASES];
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

function parseCase(value) {
  if (!CASE_PAGES.has(value)) {
    throw new Error(`--case must be one of ${[...CASE_PAGES.keys()].join(', ')}, got ${value}`);
  }
  return value;
}

function exampleDir() {
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
      cwd: exampleDir(),
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
    throw new Error(`Example server did not start at ${url}: ${error.message}`);
  }
}

async function runCase(page, url, caseName, options) {
  const pageName = CASE_PAGES.get(caseName);
  await page.goto(`${url}/${pageName}`, { waitUntil: 'domcontentloaded', timeout: options.timeoutMs });
  await page.setInputFiles('#model-file', options.model);
  await page.locator('#model-form button[type=submit]').click();
  await page.waitForFunction(
    () => document.querySelector('#output')?.textContent?.includes('Loaded ') === true,
    null,
    { timeout: options.timeoutMs }
  );
  await page.fill('#prompt', options.prompt);
  if (await page.locator('#max-tokens').count() > 0) {
    await page.fill('#max-tokens', String(options.maxTokens));
  }
  await page.locator('#run-form button[type=submit]').click();
  await page.waitForFunction(
    (activeCase) => {
      const text = document.querySelector('#output')?.textContent ?? '';
      if (activeCase === 'embed') {
        return text.includes('dimensions=') && text.includes('preview=');
      }
      return text.includes('finish_reason=') && text.includes('text=') && text.includes('metrics=');
    },
    caseName,
    { timeout: options.timeoutMs }
  );
  return {
    case: caseName,
    output: await page.locator('#output').textContent(),
  };
}

async function runBrowserExampleSmoke(options) {
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
    const cases = [];
    for (const caseName of options.cases) {
      cases.push(await runCase(page, url, caseName, options));
    }
    return { url, cases };
  } finally {
    await browser?.close();
    await closeServer(child);
  }
}

try {
  const options = parseArgs(process.argv.slice(2));
  const report = await runBrowserExampleSmoke(options);
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
