'use strict'

const { execSync } = require('node:child_process')
const { EventEmitter } = require('node:events')
const { readFileSync } = require('node:fs')
const path = require('node:path')
const gatewayProfile = require('./gateway-profile.js')
const packageMetadata = require('./package.json')

const BINARY_NAME = 'sipp_node'
const VALID_BACKENDS = new Set(['auto', 'cpu', 'cuda', 'metal', 'vulkan'])

let activeBackend = 'unknown'
const backendEvents = new EventEmitter()

function isFileMusl(file) {
  return file.includes('libc.musl-') || file.includes('ld-musl-')
}

function isMuslFromFilesystem() {
  try {
    return readFileSync('/usr/bin/ldd', 'utf8').includes('musl')
  } catch {
    return null
  }
}

function isMuslFromReport() {
  let report = null
  if (typeof process.report?.getReport === 'function') {
    process.report.excludeNetwork = true
    report = process.report.getReport()
  }
  if (!report) {
    return null
  }
  if (report.header && report.header.glibcVersionRuntime) {
    return false
  }
  if (Array.isArray(report.sharedObjects)) {
    return report.sharedObjects.some(isFileMusl)
  }
  return false
}

function isMuslFromChildProcess() {
  try {
    return execSync('ldd --version', { encoding: 'utf8' }).includes('musl')
  } catch {
    return false
  }
}

function isMusl() {
  if (process.platform !== 'linux') {
    return false
  }

  return isMuslFromFilesystem() ?? isMuslFromReport() ?? isMuslFromChildProcess()
}

function platformTriplet() {
  if (process.platform === 'win32' && process.arch === 'x64') {
    const isGnu =
      process.config?.variables?.shlib_suffix === 'dll.a' ||
      process.config?.variables?.node_target_type === 'shared_library'
    return isGnu ? 'win32-x64-gnu' : 'win32-x64-msvc'
  }

  if (process.platform === 'darwin') {
    if (process.arch === 'x64') {
      return 'darwin-x64'
    }
    if (process.arch === 'arm64') {
      return 'darwin-arm64'
    }
  }

  if (process.platform === 'linux' && process.arch === 'x64') {
    return isMusl() ? 'linux-x64-musl' : 'linux-x64-gnu'
  }

  throw new Error(
    `Unsupported OS/architecture for Sipp Node bindings: ${process.platform} ${process.arch}`,
  )
}

function autoBackendsForHost() {
  if (process.platform === 'darwin') {
    return ['metal', 'cpu']
  }
  return ['cuda', 'vulkan', 'cpu']
}

function requestedBackends() {
  const requested = (process.env.SIPP_NODE_BACKEND ?? 'auto').toLowerCase()
  if (!VALID_BACKENDS.has(requested)) {
    const valid = 'auto, cpu, cuda, metal, vulkan'
    throw new Error(
      `Invalid SIPP_NODE_BACKEND=${process.env.SIPP_NODE_BACKEND}. Expected one of: ${valid}`,
    )
  }

  return requested === 'auto' ? autoBackendsForHost() : [requested]
}

function backendBinaryPaths(backend, triplet) {
  const fileName = `${BINARY_NAME}_${backend}.${triplet}.node`
  const platformPackage = platformPackageName(triplet)
  return [
    { path: path.join(__dirname, 'native', fileName) },
    { specifier: `${platformPackage}/native/${fileName}` },
    { path: path.join(__dirname, '..', '..', '.build', 'artifacts', 'node', fileName) },
  ]
}

function platformPackageName(triplet) {
  return `${packageMetadata.name}-${triplet}`
}

function assertBackendUsable(binding, backend) {
  if (backend !== 'cpu' && !binding.backendIsUsable(backend)) {
    throw new Error(
      `${backend} binding loaded, but no usable ${backend} backend was reported by llama.cpp`,
    )
  }
}

function errorMessage(error) {
  return error && error.message ? error.message : String(error)
}

function backendFallbackEvents(errors, fallbackTo) {
  return errors.map(({ backend, error }) => ({
    type: 'fallback-warning',
    kind: 'backend',
    detail: `${backend} backend unavailable: ${errorMessage(error)}`,
    fallbackTo,
  }))
}

function loadCandidate(backend, triplet) {
  const errors = []
  for (const candidate of backendBinaryPaths(backend, triplet)) {
    let binding
    try {
      binding = requireBinaryCandidate(candidate)
    } catch (error) {
      errors.push(error)
      continue
    }

    assertBackendUsable(binding, backend)
    activeBackend = backend
    return binding
  }
  throw errors[errors.length - 1]
}

function requireBinaryCandidate(candidate) {
  if (candidate.path != null) {
    return require(candidate.path)
  }

  return require(candidate.specifier)
}

function loadNativeBinding() {
  if (process.env.NAPI_RS_NATIVE_LIBRARY_PATH) {
    const binding = require(process.env.NAPI_RS_NATIVE_LIBRARY_PATH)
    const requested = (process.env.SIPP_NODE_BACKEND ?? 'cpu').toLowerCase()
    activeBackend = VALID_BACKENDS.has(requested) && requested !== 'auto' ? requested : 'cpu'
    return { binding, fallbackEvents: [] }
  }

  const triplet = platformTriplet()
  const errors = []

  for (const backend of requestedBackends()) {
    try {
      const binding = loadCandidate(backend, triplet)
      return {
        binding,
        fallbackEvents: backendFallbackEvents(errors, backend),
      }
    } catch (error) {
      errors.push({ backend, error })
    }
  }

  const detail = errors
    .map(({ backend, error }) => `${backend}: ${errorMessage(error)}`)
    .join('\n')
  const message =
    `Sipp failed to load a usable Node backend for ${process.platform} ${process.arch}.\n` +
    detail
  throw new Error(message, { cause: errors[errors.length - 1]?.error })
}

const RESPONSE_PROMISE = Symbol('sipp.responsePromise')

function attachResponseGetter(Run) {
  if (typeof Run !== 'function') {
    return
  }
  if (Object.getOwnPropertyDescriptor(Run.prototype, 'response') != null) {
    return
  }
  Object.defineProperty(Run.prototype, 'response', {
    get() {
      if (this[RESPONSE_PROMISE] == null) {
        this[RESPONSE_PROMISE] = this.__response()
      }
      return this[RESPONSE_PROMISE]
    },
  })
}

const { binding, fallbackEvents } = loadNativeBinding()

queueMicrotask(() => {
  for (const event of fallbackEvents) {
    backendEvents.emit('fallback', event)
  }
})

function attachRunIterables(nativeBinding) {
  const TextRun = nativeBinding.SippTextRun
  const EmbeddingRun = nativeBinding.SippEmbeddingRun
  const AudioRun = nativeBinding.SippAudioRun

  attachResponseGetter(TextRun)
  attachResponseGetter(EmbeddingRun)
  attachResponseGetter(AudioRun)

  if (typeof TextRun === 'function' && TextRun.prototype[Symbol.asyncIterator] == null) {
    Object.defineProperty(TextRun.prototype, Symbol.asyncIterator, {
      value: async function* tokenIterator() {
        while (true) {
          const batch = await this.__nextToken()
          if (batch == null) {
            return
          }
          yield batch
        }
      },
    })
  }

  if (typeof TextRun === 'function' && Object.getOwnPropertyDescriptor(TextRun.prototype, 'tokens') == null) {
    Object.defineProperty(TextRun.prototype, 'tokens', {
      get() {
        const run = this
        return {
          [Symbol.asyncIterator]: () => run[Symbol.asyncIterator](),
        }
      },
    })
  }
}

attachRunIterables(binding)

module.exports = binding
module.exports.getActiveBackend = () => activeBackend
module.exports.onFallback = (listener) => {
  backendEvents.on('fallback', listener)
  return () => backendEvents.off('fallback', listener)
}
Object.assign(module.exports, gatewayProfile)
