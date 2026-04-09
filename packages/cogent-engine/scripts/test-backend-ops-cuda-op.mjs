import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const rawRunnerScript = path.join(scriptDir, 'test-backend-ops-cuda.mjs');
const validModes = new Set(['test', 'support', 'perf', 'grad']);

function printHelp() {
  console.log(`Usage: bun ./scripts/test-backend-ops-cuda-op.mjs [op selector] [options]

Examples:
  bun run test:backend-ops:cuda:op -- GET_ROWS
  bun run test:backend-ops:cuda:op -- GET_ROWS,SET_ROWS --mode support --output csv
  bun run test:backend-ops:cuda:op -- "Get Rows" --filter "type=f32"
  bun run test:backend-ops:cuda:debug -- GET_ROWS

Options:
  --mode <test|support|perf|grad>   Select the upstream test-backend-ops mode.
  --filter <regex>                  Forward a regex to -p.
  --output <console|csv|sql>        Forward the upstream --output format.
  --backend <name>                  Override backend selection with an exact device name such as CUDA0.
  --test-file <path>                Forward --test-file.
  --list-ops                        Forward --list-ops.
  --show-coverage                   Forward --show-coverage.
  --debug                           Build Debug binary with verbose output.
  --                               Forward any remaining arguments directly to test-backend-ops.

Notes:
  - Friendly selectors such as "Get Rows" and "get-rows" normalize to GET_ROWS.
  - Exact upstream test case strings containing parentheses are preserved as-is.
  - Passing --backend CUDA falls back to the runner default, which targets discovered CUDA devices in this CUDA-only build.
`);
}

function normalizeOpSelector(value) {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  if (trimmed.includes('(')) {
    return trimmed;
  }

  return trimmed.replace(/[\s-]+/g, '_').toUpperCase();
}

function buildOpSelector(opTokens) {
  const selectors = [];

  for (const token of opTokens) {
    if (token.includes('(')) {
      const exactSelector = normalizeOpSelector(token);
      if (exactSelector) {
        selectors.push(exactSelector);
      }
      continue;
    }

    for (const part of token.split(',')) {
      const selector = normalizeOpSelector(part);
      if (selector) {
        selectors.push(selector);
      }
    }
  }

  return selectors.join(',');
}

function requireValue(argv, index, flagName) {
  if (index + 1 >= argv.length) {
    throw new Error(`Missing value for ${flagName}.`);
  }

  return argv[index + 1];
}

function normalizeBackendSelection(value) {
  if (!value) {
    return null;
  }

  const trimmedValue = value.trim();
  if (!trimmedValue) {
    return null;
  }

  return trimmedValue.toUpperCase() === 'CUDA' ? null : trimmedValue;
}

function parseWrapperArgs(argv) {
  const options = {
    debug: false,
    mode: 'test',
    filter: null,
    output: null,
    backend: null,
    testFile: null,
    listOps: false,
    showCoverage: false,
    opTokens: [],
    passthrough: [],
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === '--') {
      options.passthrough = argv.slice(index + 1);
      break;
    }

    if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    }

    if (arg === '--debug') {
      options.debug = true;
      continue;
    }

    if (arg === '--list-ops') {
      options.listOps = true;
      continue;
    }

    if (arg === '--show-coverage') {
      options.showCoverage = true;
      continue;
    }

    if (arg === '--mode') {
      options.mode = requireValue(argv, index, '--mode');
      index += 1;
      continue;
    }

    if (arg.startsWith('--mode=')) {
      options.mode = arg.slice('--mode='.length);
      continue;
    }

    if (arg === '--filter') {
      options.filter = requireValue(argv, index, '--filter');
      index += 1;
      continue;
    }

    if (arg.startsWith('--filter=')) {
      options.filter = arg.slice('--filter='.length);
      continue;
    }

    if (arg === '--output') {
      options.output = requireValue(argv, index, '--output');
      index += 1;
      continue;
    }

    if (arg.startsWith('--output=')) {
      options.output = arg.slice('--output='.length);
      continue;
    }

    if (arg === '--backend') {
      options.backend = normalizeBackendSelection(requireValue(argv, index, '--backend'));
      index += 1;
      continue;
    }

    if (arg.startsWith('--backend=')) {
      options.backend = normalizeBackendSelection(arg.slice('--backend='.length));
      continue;
    }

    if (arg === '--test-file') {
      options.testFile = requireValue(argv, index, '--test-file');
      index += 1;
      continue;
    }

    if (arg.startsWith('--test-file=')) {
      options.testFile = arg.slice('--test-file='.length);
      continue;
    }

    if (arg.startsWith('-')) {
      throw new Error(`Unknown wrapper option: ${arg}`);
    }

    options.opTokens.push(arg);
  }

  if (!validModes.has(options.mode)) {
    throw new Error(`Unsupported mode "${options.mode}". Expected one of: ${Array.from(validModes).join(', ')}.`);
  }

  if (options.opTokens.length === 0 && !options.listOps && !options.showCoverage && !options.testFile) {
    throw new Error('Provide at least one op selector, or use --list-ops / --show-coverage / --test-file.');
  }

  return options;
}

function createForwardedArgs(options) {
  const forwardedArgs = [];
  const opSelector = buildOpSelector(options.opTokens);

  if (!options.listOps && !options.showCoverage) {
    forwardedArgs.push(options.mode);
  }

  if (opSelector) {
    forwardedArgs.push('-o', opSelector);
  }

  if (options.backend) {
    forwardedArgs.push('-b', options.backend);
  }

  if (options.filter) {
    forwardedArgs.push('-p', options.filter);
  }

  if (options.output) {
    forwardedArgs.push('--output', options.output);
  }

  if (options.testFile) {
    forwardedArgs.push('--test-file', options.testFile);
  }

  if (options.listOps) {
    forwardedArgs.push('--list-ops');
  }

  if (options.showCoverage) {
    forwardedArgs.push('--show-coverage');
  }

  forwardedArgs.push(...options.passthrough);
  return forwardedArgs;
}

function resolveRuntimeExecutable() {
  const executable = process.execPath?.trim();
  if (executable && existsSync(executable)) {
    return executable;
  }

  return 'bun';
}

async function runRawRunner(forwardedArgs, envOverrides) {
  const runtimeExecutable = resolveRuntimeExecutable();
  const childProcess = spawn(runtimeExecutable, [rawRunnerScript, ...forwardedArgs], {
    cwd: projectRoot,
    stdio: 'inherit',
    shell: false,
    windowsHide: true,
    env: {
      ...process.env,
      ...envOverrides,
    },
  });

  const exitForSignal = (signal) => {
    childProcess.kill(signal);
    process.exit(signal === 'SIGINT' ? 130 : 143);
  };

  process.on('SIGINT', () => exitForSignal('SIGINT'));
  process.on('SIGTERM', () => exitForSignal('SIGTERM'));

  return await new Promise((resolve, reject) => {
    childProcess.once('error', reject);
    childProcess.once('exit', (code, signal) => {
      if (signal) {
        reject(new Error(`Wrapper runner terminated by signal ${signal}.`));
        return;
      }

      resolve(code ?? 0);
    });
  });
}

const options = parseWrapperArgs(Bun.argv.slice(2));
const forwardedArgs = createForwardedArgs(options);
const envOverrides = {
  CE_TEST_BACKEND_OPS_BUILD_LABEL: options.debug ? '[test-backend-ops:cuda:debug]' : '[test-backend-ops:cuda:op]',
};

if (options.debug) {
  envOverrides.CE_TEST_BACKEND_OPS_BUILD_TYPE = 'Debug';
  envOverrides.CE_TEST_BACKEND_OPS_BUILD_DIR_NAME = 'build-test-backend-ops-cuda-debug';
}

const exitCode = await runRawRunner(forwardedArgs, envOverrides);
process.exit(exitCode);
