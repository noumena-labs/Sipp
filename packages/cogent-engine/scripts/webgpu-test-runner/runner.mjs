const statusNode = document.getElementById('status');
const outputNode = document.getElementById('output');
const resumeButtonNode = document.getElementById('resume');

const state = {
  done: false,
  exitCode: 1,
  error: null,
  startedAt: Date.now(),
  finishedAt: null,
  pauseBeforeRun: false,
  waitingForResume: false,
};

window.__webgpuTestRunner = state;

let resumeRun = null;

function appendOutput(stream, chunk) {
  const text = String(chunk ?? '');
  const prefix = stream === 'stderr' ? '[stderr]' : '[stdout]';
  outputNode.textContent += `${prefix} ${text}\n`;
}

function stringifyError(error) {
  if (error == null) {
    return 'Unknown error';
  }

  if (typeof error === 'string') {
    return error;
  }

  if (typeof error === 'object' && 'stack' in error && typeof error.stack === 'string') {
    return error.stack;
  }

  if (typeof error === 'object' && 'message' in error && typeof error.message === 'string') {
    return error.message;
  }

  return String(error);
}

function finalize(exitCode, error) {
  if (state.done) {
    return;
  }

  state.done = true;
  state.exitCode = Number.isFinite(exitCode) ? Number(exitCode) : 1;
  state.error = error ? stringifyError(error) : null;
  state.finishedAt = Date.now();
  state.waitingForResume = false;

  if (resumeButtonNode) {
    resumeButtonNode.hidden = true;
  }

  if (state.error) {
    statusNode.textContent = `Run failed with exit code ${state.exitCode}.`;
  } else {
    statusNode.textContent = `Run finished with exit code ${state.exitCode}.`;
  }
}

function isExitStatus(error) {
  return Boolean(error && typeof error === 'object' && 'status' in error && Number.isFinite(error.status));
}

function readArguments() {
  const params = new URLSearchParams(window.location.search);
  const modulePath = params.get('module');
  const rawArgs = params.get('args');
  const pauseBeforeRun = params.get('pauseBeforeRun') === '1';
  const args = rawArgs ? JSON.parse(rawArgs) : [];

  if (!modulePath) {
    throw new Error('Missing module query parameter.');
  }

  if (!Array.isArray(args)) {
    throw new Error('The args query parameter must decode to a JSON array.');
  }

  return {
    modulePath,
    args,
    pauseBeforeRun,
  };
}

function resumeExecution() {
  if (!resumeRun) {
    return false;
  }

  const callback = resumeRun;
  resumeRun = null;
  state.waitingForResume = false;

  if (resumeButtonNode) {
    resumeButtonNode.hidden = true;
  }

  statusNode.textContent = 'Debugger attached. Starting wasm run...';
  console.log('[webgpu-test-runner] resuming wasm run');
  callback();
  return true;
}

window.__webgpuTestRunner.resume = resumeExecution;

if (resumeButtonNode) {
  resumeButtonNode.addEventListener('click', () => {
    resumeExecution();
  });
}

function waitForResumeIfNeeded(pauseBeforeRun) {
  if (!pauseBeforeRun) {
    return Promise.resolve();
  }

  state.pauseBeforeRun = true;
  state.waitingForResume = true;
  statusNode.textContent = 'Waiting for debugger attach. Set breakpoints, then click Resume Wasm Run.';

  if (resumeButtonNode) {
    resumeButtonNode.hidden = false;
  }

  console.log('[webgpu-test-runner] waiting for debugger attach; click Resume Wasm Run or call window.__webgpuTestRunner.resume()');

  return new Promise((resolve) => {
    resumeRun = resolve;
  });
}

window.addEventListener('error', (event) => {
  if (state.done) {
    return;
  }

  finalize(1, event.error ?? event.message);
});

window.addEventListener('unhandledrejection', (event) => {
  if (state.done) {
    return;
  }

  finalize(1, event.reason);
});

async function main() {
  const { modulePath, args, pauseBeforeRun } = readArguments();
  if (!('gpu' in navigator)) {
    throw new Error('navigator.gpu is unavailable. Launch Chromium with WebGPU enabled.');
  }

  const moduleUrl = new URL(modulePath, window.location.origin).href;
  const moduleFactoryModule = await import(moduleUrl);
  const moduleFactory = moduleFactoryModule.default ?? moduleFactoryModule;

  if (typeof moduleFactory !== 'function') {
    throw new Error('The generated wasm wrapper does not export a default module factory.');
  }

  const moduleOptions = {
    noInitialRun: true,
    print: (text) => {
      appendOutput('stdout', text);
      console.log(text);
    },
    printErr: (text) => {
      appendOutput('stderr', text);
      console.error(text);
    },
    locateFile: (assetName) => new URL(assetName, moduleUrl).href,
    onAbort: (reason) => {
      finalize(1, reason);
    },
    quit: (status, toThrow) => {
      finalize(status, Number(status) === 0 ? null : `Program exited with code ${status}.`);
      if (toThrow !== undefined) {
        throw toThrow;
      }
    },
  };

  const moduleInstance = await moduleFactory(moduleOptions);
  await waitForResumeIfNeeded(pauseBeforeRun);

  try {
    await Promise.resolve(moduleInstance.callMain(args));
    finalize(0, null);
  } catch (error) {
    if (state.done) {
      return;
    }

    if (isExitStatus(error)) {
      const exitCode = Number(error.status);
      finalize(exitCode, exitCode === 0 ? null : `Program exited with code ${exitCode}.`);
      return;
    }

    finalize(1, error);
    throw error;
  }
}

main().catch((error) => {
  if (!state.done) {
    finalize(1, error);
  }
  console.error(stringifyError(error));
});