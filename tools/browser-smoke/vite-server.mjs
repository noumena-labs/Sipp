import http from 'node:http';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';

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

async function waitForServer(url, timeoutMs) {
  const started = Date.now();
  let lastError = null;
  while (Date.now() - started < timeoutMs) {
    try {
      const status = await httpStatus(url);
      if (status >= 200 && status < 500) {
        return;
      }
    } catch (error) {
      lastError = error;
    }
    await sleep(250);
  }
  if (lastError != null) {
    throw lastError;
  }
  throw new Error(`Server at ${url} did not become ready within ${timeoutMs}ms`);
}

function startVite(rootDir, host, port) {
  const appRoot = path.resolve(rootDir);
  const require = createRequire(path.join(appRoot, 'package.json'));
  const vitePackagePath = require.resolve('vite/package.json');
  const viteCliPath = path.join(path.dirname(vitePackagePath), 'bin', 'vite.js');
  const child = spawn(
    process.execPath,
    [viteCliPath, '--host', host, '--port', String(port), '--strictPort'],
    {
      cwd: appRoot,
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

/** Rejects an operation when it exceeds the smoke test timeout. */
export function withTimeout(promise, timeoutMs, label) {
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

/** Stops a Vite process started by `ensureViteServer`. */
export async function closeViteServer(child) {
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

/** Reuses a reachable server or starts the workspace-local Vite installation. */
export async function ensureViteServer({ rootDir, host, port, timeoutMs, label }) {
  const url = `http://${host}:${port}`;
  try {
    await waitForServer(url, 1_000);
    return { url, child: null };
  } catch {
    // No existing server; start a local Vite process below.
  }

  const child = startVite(rootDir, host, port);
  try {
    await waitForServer(url, timeoutMs);
    return { url, child };
  } catch (error) {
    await closeViteServer(child);
    throw new Error(`${label} server did not start at ${url}: ${error.message}`);
  }
}
