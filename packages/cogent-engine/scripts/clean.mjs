import { rm } from 'node:fs/promises';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const transientRemoveErrorCodes = new Set(['EBUSY', 'ENOTEMPTY', 'EPERM']);
const removeRetryLimit = 10;
const removeRetryDelayMs = 200;
const optionalCleanTargets = new Set([
  path.join(projectRoot, 'build'),
  path.join(projectRoot, 'build-wasm-dev')
]);

const cleanTargets = [
  path.join(projectRoot, 'build'),
  path.join(projectRoot, 'build-wasm-dev'),
  path.join(projectRoot, 'dist', 'esm'),
  path.join(projectRoot, 'dist', 'types'),
  path.join(projectRoot, 'dist', 'wasm'),
  path.join(projectRoot, 'a.out.js'),
  path.join(projectRoot, 'a.out.wasm')
];

async function removeTarget(targetPath) {
  for (let attempt = 0; attempt <= removeRetryLimit; attempt += 1) {
    try {
      await rm(targetPath, { recursive: true, force: true });
      return;
    } catch (error) {
      const errorCode =
        typeof error === 'object' && error !== null && 'code' in error ? String(error.code) : null;
      const canRetry = errorCode != null && transientRemoveErrorCodes.has(errorCode);
      if (!canRetry || attempt === removeRetryLimit) {
        throw error;
      }

      await delay(removeRetryDelayMs * (attempt + 1));
    }
  }
}

function getErrorCode(error) {
  return typeof error === 'object' && error !== null && 'code' in error ? String(error.code) : null;
}

for (const targetPath of cleanTargets) {
  try {
    await removeTarget(targetPath);
  } catch (error) {
    const errorCode = getErrorCode(error);
    if (errorCode != null && transientRemoveErrorCodes.has(errorCode) && optionalCleanTargets.has(targetPath)) {
      console.warn(`[clean] skipped locked ${targetPath} (${errorCode})`);
      continue;
    }

    throw error;
  }

  console.log(`[clean] removed ${targetPath}`);
}
