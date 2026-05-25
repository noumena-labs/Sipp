import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const rawRunnerScript = path.join(scriptDir, 'test-backend-ops-cpu.mjs');
const validModes = new Set(['test', 'support', 'perf', 'grad']);

function printHelp() {
  console.log(`Usage: bun ./scripts/test-backend-ops-cpu-op.mjs [op selector] [options]

Examples:
  bun run test:backend-ops:cpu:op -- GET_ROWS
  bun run test:backend-ops:cpu:op -- GET_ROWS,SET_ROWS --mode support --output csv
  bun run test:backend-ops:cpu:op -- "Get Rows" --filter "type=f32"

Options:
  --mode <test|support|perf|grad>   Select the upstream test-backend-ops mode.
  --filter <regex>                  Forward a regex to -p.
  --output <console|csv|sql>        Forward the upstream --output format.
  --backend <name>                  Override backend selection.
  --test-file <path>                Forward --test-file.
  --list-ops                        Forward --list-ops.
  --show-coverage                   Forward --show-coverage.
  --debug                           Build Debug binary.
  --                                Forward any remaining arguments directly to test-backend-ops.
`);
}

function normalizeOpSelector(value) {
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (trimmed.includes('(')) return trimmed;
  return trimmed.replace(/[\s-]+/g, '_').toUpperCase();
}

function buildOpSelector(opTokens) {
  const selectors = [];
  for (const token of opTokens) {
    if (token.includes('(')) {
      const exactSelector = normalizeOpSelector(token);
      if (exactSelector) selectors.push(exactSelector);
      continue;
    }
    for (const part of token.split(',')) {
      const selector = normalizeOpSelector(part);
      if (selector) selectors.push(selector);
    }
  }
  return selectors.join(',');
}

function requireValue(argv, index, flagName) {
  if (index + 1 >= argv.length) throw new Error(`Missing value for ${flagName}.`);
  return argv[index + 1];
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
      options.backend = requireValue(argv, index, '--backend');
      index += 1;
      continue;
    }
    if (arg.startsWith('--backend=')) {
      options.backend = arg.slice('--backend='.length);
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
    if (arg.startsWith('-')) throw new Error(`Unknown option: ${arg}`);
    options.opTokens.push(arg);
  }

  if (!validModes.has(options.mode)) throw new Error(`Unsupported mode "${options.mode}".`);
  if (options.opTokens.length === 0 && !options.listOps && !options.showCoverage && !options.testFile) {
    throw new Error('Provide at least one op selector.');
  }
  return options;
}

function createForwardedArgs(options) {
  const forwardedArgs = [];
  const opSelector = buildOpSelector(options.opTokens);
  if (!options.listOps && !options.showCoverage) forwardedArgs.push(options.mode);
  if (opSelector) forwardedArgs.push('-o', opSelector);
  if (options.backend) forwardedArgs.push('-b', options.backend);
  if (options.filter) forwardedArgs.push('-p', options.filter);
  if (options.output) forwardedArgs.push('--output', options.output);
  if (options.testFile) forwardedArgs.push('--test-file', options.testFile);
  if (options.listOps) forwardedArgs.push('--list-ops');
  if (options.showCoverage) forwardedArgs.push('--show-coverage');
  forwardedArgs.push(...options.passthrough);
  return forwardedArgs;
}

const options = parseWrapperArgs(Bun.argv.slice(2));
const forwardedArgs = createForwardedArgs(options);
const envOverrides = {
  CE_TEST_BACKEND_OPS_BUILD_LABEL: options.debug ? '[test-backend-ops:cpu:debug]' : '[test-backend-ops:cpu:op]',
};

if (options.debug) {
  envOverrides.CE_TEST_BACKEND_OPS_BUILD_TYPE = 'Debug';
  envOverrides.CE_TEST_BACKEND_OPS_BUILD_DIR_NAME = 'build-test-backend-ops-cpu-debug';
}

const childProcess = spawn('bun', [rawRunnerScript, ...forwardedArgs], {
  cwd: projectRoot,
  stdio: 'inherit',
  env: { ...process.env, ...envOverrides },
});

childProcess.on('exit', (code) => process.exit(code ?? 0));
