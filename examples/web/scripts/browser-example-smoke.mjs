import path from 'node:path';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import {
  closeViteServer,
  ensureViteServer,
  withTimeout,
} from '../../../tools/browser-smoke/vite-server.mjs';

const EXAMPLE_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
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
  const { url, child } = await ensureViteServer({
    rootDir: EXAMPLE_DIR,
    host: options.host,
    port: options.port,
    timeoutMs: options.timeoutMs,
    label: 'Example',
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
    const cases = [];
    for (const caseName of options.cases) {
      cases.push(await runCase(page, url, caseName, options));
    }
    return { url, cases };
  } finally {
    await browser?.close();
    await closeViteServer(child);
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
