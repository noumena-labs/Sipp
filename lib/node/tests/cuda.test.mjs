import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { gzipSync } from 'node:zlib';

const require = createRequire(import.meta.url);
const {
  cudaDownloadBlocker,
  cudaLinuxLibraries,
  cudaPlatform,
  cudaRuntimeDll,
  installCudaBundle,
  locateCuda,
  nvidiaDriver,
  prepareCudaRuntime,
  readyCudaBundle,
} = require('../cuda.js');

const sha256 = (buffer) => createHash('sha256').update(buffer).digest('hex');

test('finds the Windows CUDA runtime on PATH, in a toolkit, or in the download', () => {
  const toolkit = String.raw`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`;
  const windows = String.raw`C:\Windows`;
  const cuda13 = String.raw`${toolkit}\v13.0\bin\x64`;
  const cuda12 = String.raw`${toolkit}\v12.4\bin`;
  const probe = {
    platform: 'win32',
    exists: (file) => file === `${cuda13}\\${cudaRuntimeDll}`,
    toolkits: () => ['v12.4', 'v13.0'],
    downloaded: undefined,
  };

  // A terminal opened before CUDA 13 was installed: the toolkit goes first on PATH.
  const stale = { PATH: `${cuda12};${windows}`, ProgramFiles: String.raw`C:\Program Files` };
  assert.equal(prepareCudaRuntime({ ...probe, env: stale }), true);
  assert.equal(stale.PATH, `${cuda13};${cuda12};${windows}`);

  // A PATH that already reaches it is left alone.
  const current = { PATH: `${cuda13};${windows}` };
  assert.deepEqual(locateCuda({ ...probe, env: current }), { kind: 'loader', directory: cuda13 });
  assert.equal(prepareCudaRuntime({ ...probe, env: current }), true);
  assert.equal(current.PATH, `${cuda13};${windows}`);

  // Only CUDA 12: the download stands in, and without one nothing is reachable.
  const only12 = { ...probe, exists: () => false, toolkits: () => ['v12.4'] };
  assert.equal(prepareCudaRuntime({ ...only12, env: { PATH: windows } }), false);
  const downloaded = String.raw`C:\Users\me\.sipp\cuda\13.0.3\win32-x64`;
  const env = { PATH: windows };
  assert.equal(prepareCudaRuntime({ ...only12, env, downloaded }), true);
  assert.equal(env.PATH, `${downloaded};${windows}`);
});

test('preloads Linux CUDA 13 libraries the loader cannot reach, dependencies first', () => {
  const cuda13 = '/usr/local/cuda-13.0/lib64';
  const present = new Set(cudaLinuxLibraries.map((library) => `${cuda13}/${library}`));
  const opened = [];
  const open = (file) => {
    opened.push(file);
    throw new Error(`Module did not self-register: '${file}'.`);
  };
  const probe = {
    platform: 'linux',
    exists: (file) => present.has(file),
    toolkits: () => ['cuda-12.4', 'cuda-13.0'],
    loaderCache: () => '',
    downloaded: undefined,
    open,
  };

  assert.equal(prepareCudaRuntime({ ...probe, env: {} }), true);
  assert.deepEqual(opened, cudaLinuxLibraries.map((library) => `${cuda13}/${library}`));

  // Libraries on LD_LIBRARY_PATH or in the ld.so cache are left to the loader.
  opened.length = 0;
  assert.equal(prepareCudaRuntime({ ...probe, env: { LD_LIBRARY_PATH: cuda13 } }), true);
  const cached = cudaLinuxLibraries.map((library) => `${library}\0`).join('');
  assert.equal(
    prepareCudaRuntime({ ...probe, env: {}, exists: () => false, loaderCache: () => cached }),
    true
  );
  assert.deepEqual(opened, []);

  // A library that fails to load for real is not reachable.
  const broken = () => {
    throw new Error('libcublasLt.so.13: cannot open shared object file');
  };
  assert.equal(prepareCudaRuntime({ ...probe, env: {}, open: broken }), false);
});

test('downloads only for a driver that runs CUDA 13 on a supported platform', () => {
  const smi = (stdout, status = 0) => () => ({ status, stdout });
  assert.deepEqual(nvidiaDriver('linux', smi('591.59\n')), { version: '591.59', major: 591 });
  assert.deepEqual(nvidiaDriver('linux', smi('', 1), () => false, () => 'libcuda.so.1\0'), {});
  assert.equal(nvidiaDriver('win32', smi('', 1), () => false), undefined);

  const bundle = {
    version: '13.0.3',
    baseUrl: 'https://cdn.example/cuda',
    platforms: { 'linux-x64': [{ name: 'a', size: 1, sha256: 'x', gzipSize: 1 }] },
  };
  assert.equal(cudaDownloadBlocker({ version: '591.59', major: 591 }, bundle, 'linux-x64'), undefined);
  assert.match(cudaDownloadBlocker({ version: '560.35', major: 560 }, bundle, 'linux-x64'), /560\.35 is too old/);
  assert.match(cudaDownloadBlocker(undefined, bundle, 'linux-x64'), /No NVIDIA driver/);
  assert.match(cudaDownloadBlocker({}, bundle, 'win32-x64'), /no CUDA 13\.0\.3 download for win32-x64/);
  assert.deepEqual(
    [cudaPlatform('darwin', 'arm64'), cudaPlatform('linux', 'arm64'), cudaPlatform('win32', 'x64')],
    [undefined, undefined, 'win32-x64']
  );
});

test('downloads and verifies the CUDA libraries once, and rejects a corrupted one', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'sipp-cuda-'));
  const previousHome = process.env.SIPP_HOME;
  process.env.SIPP_HOME = home;
  const files = {
    'libcudart.so.13': Buffer.from('cudart'.repeat(1000)),
    'NVIDIA-LICENSE.txt': Buffer.from('license'),
  };
  const served = {};
  let requests = 0;
  const server = createServer((request, response) => {
    requests += 1;
    const body = served[request.url];
    response.writeHead(body ? 200 : 404);
    response.end(body);
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  try {
    const baseUrl = `http://127.0.0.1:${server.address().port}/cuda`;
    const libraries = Object.entries(files).map(([name, bytes]) => {
      const gzip = gzipSync(bytes);
      served[`/cuda/13.0.3/linux-x64/${name}.gz`] = gzip;
      return { name, size: bytes.length, sha256: sha256(bytes), gzipSize: gzip.length };
    });
    const bundle = { version: '13.0.3', baseUrl, platforms: { 'linux-x64': libraries } };

    assert.equal(readyCudaBundle(bundle, 'linux-x64'), undefined);
    const progress = [];
    const directory = await installCudaBundle({
      bundle,
      platform: 'linux-x64',
      onProgress: ({ received }) => progress.push(received),
    });
    assert.equal(directory, path.join(home, 'cuda', '13.0.3', 'linux-x64'));
    assert.deepEqual(readFileSync(path.join(directory, 'libcudart.so.13')), files['libcudart.so.13']);
    assert.equal(readyCudaBundle(bundle, 'linux-x64'), directory);
    assert.equal(progress.at(-1), libraries.reduce((sum, library) => sum + library.gzipSize, 0));

    // A complete folder is not downloaded again.
    const before = requests;
    await installCudaBundle({ bundle, platform: 'linux-x64' });
    assert.equal(requests, before);

    // Bytes that inflate to something else are refused, and the folder stays incomplete.
    served['/cuda/13.0.4/linux-x64/libcudart.so.13.gz'] = gzipSync(Buffer.from('tampered'));
    const tampered = { ...bundle, version: '13.0.4', platforms: { 'linux-x64': [libraries[0]] } };
    await assert.rejects(
      installCudaBundle({ bundle: tampered, platform: 'linux-x64' }),
      /does not match its published sha256/
    );
    assert.equal(readyCudaBundle(tampered, 'linux-x64'), undefined);

    // A completion record that names other files is not trusted.
    writeFileSync(path.join(directory, '.complete'), '[]');
    assert.equal(readyCudaBundle(bundle, 'linux-x64'), undefined);
  } finally {
    await new Promise((resolve) => server.close(resolve));
    if (previousHome === undefined) {
      delete process.env.SIPP_HOME;
    } else {
      process.env.SIPP_HOME = previousHome;
    }
    rmSync(home, { recursive: true, force: true });
  }
});
