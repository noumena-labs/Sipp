import { chromium } from 'playwright';

function getRemoteDebuggingPort() {
  const rawPort = process.env.CE_WEBGPU_REMOTE_DEBUG_PORT?.trim();
  if (!rawPort) {
    return null;
  }

  const parsedPort = Number.parseInt(rawPort, 10);
  if (!Number.isInteger(parsedPort) || parsedPort < 1 || parsedPort > 65535) {
    throw new Error(`Invalid CE_WEBGPU_REMOTE_DEBUG_PORT value: ${rawPort}`);
  }

  return parsedPort;
}

function getRunnerUrl() {
  const runnerUrl = process.argv[2];
  if (!runnerUrl) {
    throw new Error('Missing runner URL argument.');
  }

  return runnerUrl;
}

function mirrorConsoleMessage(message) {
  const text = message.text();
  if (!text) {
    return;
  }

  if (message.type() === 'error' || message.type() === 'warning') {
    process.stderr.write(`${text}\n`);
    return;
  }

  process.stdout.write(`${text}\n`);
}

function handlePageError(error) {
  process.stderr.write(`${error.stack ?? error.message}\n`);
}

function getBrowserLaunchPlans() {
  const mode = (process.env.CE_WEBGPU_BROWSER_MODE ?? 'auto').trim().toLowerCase();
  const remoteDebuggingPort = getRemoteDebuggingPort();
  const browserArgs = ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'];

  if (remoteDebuggingPort) {
    browserArgs.push(`--remote-debugging-port=${remoteDebuggingPort}`, '--remote-debugging-address=127.0.0.1');
  }

  const baseOptions = {
    channel: 'chromium',
    timeout: 30000,
    args: browserArgs,
  };

  if (mode === 'headless') {
    return [{ label: 'headless Chromium', options: { ...baseOptions, headless: true } }];
  }

  if (mode === 'headed') {
    return [{ label: 'headed Chromium', options: { ...baseOptions, headless: false } }];
  }

  return [
    { label: 'headless Chromium', options: { ...baseOptions, headless: true } },
    { label: 'headed Chromium fallback', options: { ...baseOptions, headless: false } },
  ];
}

async function launchBrowser() {
  let lastError = null;

  for (const plan of getBrowserLaunchPlans()) {
    console.log(`[webgpu-browser-harness] launching ${plan.label}`);

    try {
      return await chromium.launch(plan.options);
    } catch (error) {
      lastError = error;
      const message = error instanceof Error ? error.message : String(error);
      process.stderr.write(`[webgpu-browser-harness] ${plan.label} failed: ${message}\n`);
    }
  }

  const message = lastError instanceof Error ? lastError.message : String(lastError ?? 'Unknown browser launch failure');
  throw new Error(
    `${message}\nSet CE_WEBGPU_BROWSER_MODE=headed to force a visible Chromium window if headless WebGPU startup hangs.\nInstall Chromium for Playwright with: bunx playwright install chromium`
  );
}

async function main() {
  const runnerUrl = getRunnerUrl();
  const remoteDebuggingPort = getRemoteDebuggingPort();
  const browser = await launchBrowser();
  let page = null;
  let exitCode = 0;

  try {
    page = await browser.newPage();
    page.on('console', mirrorConsoleMessage);
    page.on('pageerror', handlePageError);

    await page.goto(runnerUrl, { waitUntil: 'domcontentloaded' });

    if (remoteDebuggingPort) {
      console.log(`[webgpu-browser-harness] debugger ready on port ${remoteDebuggingPort}`);
    }

    await page.waitForFunction(() => window.__webgpuTestRunner?.done === true, null, { timeout: 0 });

    const result = await page.evaluate(() => window.__webgpuTestRunner);
    page.off('pageerror', handlePageError);

    if (!result || typeof result.exitCode !== 'number') {
      throw new Error('The browser harness did not report a valid exit code.');
    }

    if (result.error) {
      process.stderr.write(`${result.error}\n`);
    }

    exitCode = result.exitCode;
  } finally {
    if (page) {
      page.off('pageerror', handlePageError);
      await page.close().catch(() => {});
    }

    await browser.close().catch(() => {});
  }

  return exitCode;
}

main()
  .then((exitCode) => {
    process.exit(exitCode);
  })
  .catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    process.exit(1);
  });