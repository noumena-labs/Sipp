'use strict'

// Finds the CUDA 13 libraries the CUDA binding links and makes them reachable
// before it loads. When none is installed, `installCudaBundle` downloads
// NVIDIA's redistributable libraries into `~/.sipp/cuda`, once per machine.

const { spawnSync } = require('node:child_process')
const { createHash } = require('node:crypto')
const { createReadStream, createWriteStream, existsSync, readdirSync, readFileSync } = require('node:fs')
const { mkdir, rename, rm, stat, writeFile } = require('node:fs/promises')
const os = require('node:os')
const path = require('node:path')
const { Readable, Transform } = require('node:stream')
const { pipeline } = require('node:stream/promises')
const { createGunzip } = require('node:zlib')

/** The CUDA library the Windows binding links; the CUDA runtime itself is static there. */
const cudaRuntimeDll = 'cublas64_13.dll'
/** The CUDA libraries the Linux binding links, dependencies first. */
const cudaLinuxLibraries = ['libcudart.so.13', 'libcublasLt.so.13', 'libcublas.so.13']
/** CUDA 13 runs on NVIDIA drivers from this major version on. */
const minimumDriverMajor = 580

/**
 * NVIDIA's CUDA 13 redistributable libraries, served gzipped as
 * `<baseUrl>/<version>/<platform>/<name>.gz`. Generated with the published
 * bundle; replace it as a whole rather than editing entries.
 */
const cudaBundle = {
  version: '13.0.3',
  baseUrl: 'https://d27ul77tgvshhk.cloudfront.net/cuda',
  platforms: {
    'win32-x64': [
      { name: 'cublas64_13.dll', size: 51561584, sha256: 'b787fada026a2cfe3eb07fe6f15b73b7a2aec6083a3cbc9584eb006737b980d6', gzipSize: 47561296 },
      { name: 'cublasLt64_13.dll', size: 480399472, sha256: '5d9ef9e66b68713f2b2a9cd6f0219f20f458e5d36e893fe34fe483bbcd68a744', gzipSize: 356550750 },
      { name: 'NVIDIA-LICENSE.txt', size: 67875, sha256: 'f176e5fd245a4ff1d7aee743e345e14b29c4b724f150e4858c4c6a313966d1fb', gzipSize: 16033 },
    ],
    'linux-x64': [
      { name: 'libcudart.so.13', size: 704288, sha256: '96c42e418cec19054186b9429c321603cc190bf26a18104e19408117a2a817b0', gzipSize: 208032 },
      { name: 'NVIDIA-LICENSE.txt', size: 67875, sha256: 'f176e5fd245a4ff1d7aee743e345e14b29c4b724f150e4858c4c6a313966d1fb', gzipSize: 16033 },
      { name: 'libcublas.so.13', size: 54177976, sha256: '25cde432108543626df85d887c730bf2cab96803ca16f9654e4abc8d2b45833e', gzipSize: 48040991 },
      { name: 'libcublasLt.so.13', size: 541607888, sha256: '1489aeb75d55ee55782798343ac66f57771d561cc1fe1e0491568deefa0e6515', gzipSize: 374967107 },
    ],
  },
}

function listDirectory(directory) {
  try {
    return readdirSync(directory)
  } catch {
    return []
  }
}

function readLoaderCache() {
  try {
    return readFileSync('/etc/ld.so.cache', 'latin1')
  } catch {
    return ''
  }
}

const newestFirst = (left, right) => right.localeCompare(left, undefined, { numeric: true })
const pathEntries = (env) => (env.PATH ?? env.Path ?? '').split(path.win32.delimiter).filter(Boolean)
const windowsToolkits = (env) =>
  path.win32.join(env.ProgramFiles ?? 'C:\\Program Files', 'NVIDIA GPU Computing Toolkit', 'CUDA')

function cudaPlatform(platform = process.platform, arch = process.arch) {
  if (arch !== 'x64') {
    return undefined
  }
  return platform === 'win32' ? 'win32-x64' : platform === 'linux' ? 'linux-x64' : undefined
}

/** `<SIPP_HOME or ~/.sipp>/cuda/<version>/<platform>`, shared by every Sipp install. */
function cudaBundleDir(bundle = cudaBundle, platform = cudaPlatform()) {
  const home = path.resolve(process.env.SIPP_HOME || path.join(os.homedir(), '.sipp'))
  return path.join(home, 'cuda', bundle.version, platform)
}

const completionFile = (directory) => path.join(directory, '.complete')
const completionText = (libraries) =>
  JSON.stringify(libraries.map(({ name, sha256 }) => ({ name, sha256 })))

/** The download folder once every library was verified into it. */
function readyCudaBundle(bundle = cudaBundle, platform = cudaPlatform()) {
  const libraries = platform && bundle.platforms[platform]
  if (!libraries?.length) {
    return undefined
  }
  const directory = cudaBundleDir(bundle, platform)
  try {
    const complete =
      readFileSync(completionFile(directory), 'utf8') === completionText(libraries) &&
      libraries.every((library) => existsSync(path.join(directory, library.name)))
    return complete ? directory : undefined
  } catch {
    return undefined
  }
}

/**
 * Finds CUDA 13: what the loader finds itself, then an installed toolkit
 * (CUDA_PATH first, then the newest), then the verified download.
 */
function locateCuda(probe = {}) {
  const {
    platform = process.platform,
    env = process.env,
    exists = existsSync,
    toolkits = listDirectory,
    loaderCache = readLoaderCache,
  } = probe
  const downloaded = 'downloaded' in probe ? probe.downloaded : readyCudaBundle()
  if (platform === 'win32') {
    const entry = pathEntries(env).find((directory) =>
      exists(path.win32.join(directory, cudaRuntimeDll)),
    )
    if (entry) {
      return { kind: 'loader', directory: entry }
    }
    const root = windowsToolkits(env)
    const versions = toolkits(root)
      .filter((name) => /^v\d+/.test(name))
      .sort(newestFirst)
      .map((name) => path.win32.join(root, name))
    const toolkit = [env.CUDA_PATH, ...versions]
      .filter(Boolean)
      .flatMap((home) => [path.win32.join(home, 'bin', 'x64'), path.win32.join(home, 'bin')])
      .find((directory) => exists(path.win32.join(directory, cudaRuntimeDll)))
    if (toolkit) {
      return { kind: 'toolkit', directory: toolkit }
    }
  } else if (platform === 'linux') {
    const entries = (env.LD_LIBRARY_PATH ?? '').split(path.posix.delimiter).filter(Boolean)
    const cache = loaderCache()
    const onLoaderPath = cudaLinuxLibraries.every(
      (library) =>
        cache.includes(`${library}\0`) ||
        entries.some((entry) => exists(path.posix.join(entry, library))),
    )
    if (onLoaderPath) {
      return { kind: 'loader' }
    }
    const versions = toolkits('/usr/local')
      .filter((name) => /^cuda-13(\.|$)/.test(name))
      .sort(newestFirst)
      .map((name) => path.posix.join('/usr/local', name))
    const toolkit = [env.CUDA_PATH, env.CUDA_HOME, ...versions, '/usr/local/cuda', '/opt/cuda']
      .filter(Boolean)
      .flatMap((home) => [
        path.posix.join(home, 'lib64'),
        path.posix.join(home, 'targets', 'x86_64-linux', 'lib'),
      ])
      .find((directory) =>
        cudaLinuxLibraries.every((library) => exists(path.posix.join(directory, library))),
      )
    if (toolkit) {
      return { kind: 'toolkit', directory: toolkit }
    }
  } else {
    return undefined
  }
  return downloaded ? { kind: 'download', directory: downloaded } : undefined
}

// glibc dlopen flags. RTLD_NODELETE keeps a preloaded library resident after Node closes it.
const RTLD_NOW = 0x2
const RTLD_NODELETE = 0x1000
const preload = (file) => process.dlopen({ exports: {} }, file, RTLD_NOW | RTLD_NODELETE)

/**
 * Makes CUDA 13 reachable for the binding about to load, and returns whether it
 * is. Windows finds DLLs through PATH, so the folder goes in front of it for
 * this process. Linux reads LD_LIBRARY_PATH only at startup, so the libraries
 * are loaded by full path, dependencies first, and the binding then resolves
 * them by soname.
 */
function prepareCudaRuntime(probe = {}) {
  const { platform = process.platform, env = process.env, open = preload } = probe
  if (platform !== 'win32' && platform !== 'linux') {
    return true
  }
  const location = locateCuda(probe)
  if (!location) {
    return false
  }
  if (location.kind === 'loader') {
    return true
  }
  if (platform === 'win32') {
    env.PATH = [location.directory, ...pathEntries(env)].join(path.win32.delimiter)
    return true
  }
  for (const library of cudaLinuxLibraries) {
    // Node loads the library, then rejects it as not being an addon; the load itself is what matters.
    try {
      open(path.posix.join(location.directory, library))
    } catch (error) {
      if (!/did not self-register/.test(String(error))) {
        return false
      }
    }
  }
  return true
}

/** The installed NVIDIA driver: its version where nvidia-smi reports one, `{}` when only its library shows it. */
function nvidiaDriver(platform = process.platform, run = spawnSync, exists = existsSync, loaderCache = readLoaderCache) {
  const result = run('nvidia-smi', ['--query-gpu=driver_version', '--format=csv,noheader'], {
    encoding: 'utf8',
    timeout: 10_000,
    windowsHide: true,
  })
  const version = result.status === 0 ? String(result.stdout).split(/\r?\n/)[0]?.trim() : undefined
  if (version) {
    return { version, major: Number.parseInt(version, 10) }
  }
  if (platform === 'win32') {
    const nvcuda = path.win32.join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'nvcuda.dll')
    return exists(nvcuda) ? {} : undefined
  }
  const linuxDriver = [
    '/usr/lib/wsl/lib/libcuda.so.1',
    '/usr/lib/x86_64-linux-gnu/libcuda.so.1',
    '/usr/lib64/libcuda.so.1',
  ]
  return loaderCache().includes('libcuda.so.1\0') || linuxDriver.some(exists) ? {} : undefined
}

/** Why the CUDA 13 libraries cannot be downloaded here, or undefined when they can. */
function cudaDownloadBlocker(driver, bundle = cudaBundle, platform = cudaPlatform()) {
  if (!platform) {
    return `Sipp downloads CUDA only for Windows x64 and Linux x64, not ${process.platform} ${process.arch}.`
  }
  if (!bundle.platforms[platform]?.length) {
    return `This Sipp release has no CUDA ${bundle.version} download for ${platform}.`
  }
  if (!driver) {
    return `No NVIDIA driver was found; install one (${minimumDriverMajor} or newer).`
  }
  if (driver.major !== undefined && driver.major < minimumDriverMajor) {
    return `NVIDIA driver ${driver.version} is too old for CUDA 13; update it to ${minimumDriverMajor} or newer.`
  }
  return undefined
}

/** Why the CUDA binding could not load, with how to get CUDA 13. */
function cudaMissingMessage(platform = process.platform) {
  const missing =
    platform === 'win32'
      ? `the CUDA 13 runtime (${cudaRuntimeDll}), which is neither on PATH nor in an installed CUDA toolkit`
      : `the CUDA 13 libraries (${cudaLinuxLibraries.join(', ')}), which are neither on the loader path nor in an installed CUDA toolkit`
  const download = `node "${path.join(__dirname, 'install-cuda.js')}"`
  return (
    `The CUDA backend needs ${missing}. Download them with ${download}, or install the ` +
    `CUDA 13 toolkit; both need NVIDIA driver ${minimumDriverMajor} or newer.`
  )
}

function cudaDownloadBytes(bundle = cudaBundle, platform = cudaPlatform()) {
  return ((platform && bundle.platforms[platform]) || []).reduce(
    (sum, library) => sum + library.gzipSize,
    0,
  )
}

/** Downloads one gzipped library to `target`, keeping it only when its size and sha256 match. */
async function downloadLibrary(url, target, library, onBytes) {
  const response = await fetch(url)
  if (!response.ok) {
    await response.body?.cancel()
    throw new Error(`Download of ${url} failed with HTTP ${response.status}`)
  }
  const temporary = `${target}.${process.pid}.tmp`
  const hash = createHash('sha256')
  let size = 0
  const meter = new Transform({
    transform(chunk, _encoding, callback) {
      hash.update(chunk)
      size += chunk.length
      callback(null, chunk)
    },
  })
  const counter = new Transform({
    transform(chunk, _encoding, callback) {
      onBytes(chunk.length)
      callback(null, chunk)
    },
  })
  try {
    await pipeline(Readable.fromWeb(response.body), counter, createGunzip(), meter, createWriteStream(temporary))
    const hex = hash.digest('hex')
    if (size !== library.size || hex !== library.sha256) {
      throw new Error(`${library.name} does not match its published sha256 (${hex}, expected ${library.sha256})`)
    }
    await rename(temporary, target)
  } finally {
    await rm(temporary, { force: true })
  }
}

async function verified(file, library) {
  if ((await stat(file).catch(() => undefined))?.size !== library.size) {
    return false
  }
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(file)) {
    hash.update(chunk)
  }
  return hash.digest('hex') === library.sha256
}

/**
 * Downloads this platform's CUDA libraries into `cudaBundleDir()`, verifying
 * each one. `.complete` is written last, so a partial folder is never used.
 */
async function installCudaBundle(options = {}) {
  const bundle = options.bundle ?? cudaBundle
  const platform = options.platform ?? cudaPlatform()
  const libraries = platform && bundle.platforms[platform]
  if (!libraries?.length) {
    throw new Error(cudaDownloadBlocker({}, bundle, platform))
  }
  const directory = cudaBundleDir(bundle, platform)
  if (readyCudaBundle(bundle, platform)) {
    return directory
  }
  await mkdir(directory, { recursive: true })
  const total = cudaDownloadBytes(bundle, platform)
  let received = 0
  for (const library of libraries) {
    const target = path.join(directory, library.name)
    if (!(await verified(target, library))) {
      const url = `${bundle.baseUrl}/${bundle.version}/${platform}/${library.name}.gz`
      await downloadLibrary(url, target, library, (bytes) => {
        received += bytes
        options.onProgress?.({ received, total })
      })
    } else {
      received += library.gzipSize
    }
  }
  await writeFile(completionFile(directory), completionText(libraries))
  return directory
}

module.exports = {
  cudaDownloadBlocker,
  cudaDownloadBytes,
  cudaLinuxLibraries,
  cudaMissingMessage,
  cudaPlatform,
  cudaRuntimeDll,
  installCudaBundle,
  locateCuda,
  nvidiaDriver,
  prepareCudaRuntime,
  readyCudaBundle,
}
