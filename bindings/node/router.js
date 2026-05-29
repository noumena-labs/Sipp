'use strict'

const { execSync } = require('node:child_process')
const { readFileSync } = require('node:fs')
const path = require('node:path')

const BINARY_NAME = 'cogentlm_node'
const VALID_BACKENDS = new Set(['auto', 'cpu', 'cuda', 'metal', 'vulkan'])

let activeBackend = 'unknown'

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
    `Unsupported OS/architecture for CogentLM Node bindings: ${process.platform} ${process.arch}`,
  )
}

function autoBackendsForHost() {
  if (process.platform === 'darwin') {
    return ['metal', 'cpu']
  }
  return ['cuda', 'vulkan', 'cpu']
}

function requestedBackends() {
  const requested = (process.env.COGENTLM_NODE_BACKEND ?? 'auto').toLowerCase()
  if (!VALID_BACKENDS.has(requested)) {
    const valid = 'auto, cpu, cuda, metal, vulkan'
    throw new Error(
      `Invalid COGENTLM_NODE_BACKEND=${process.env.COGENTLM_NODE_BACKEND}. Expected one of: ${valid}`,
    )
  }

  return requested === 'auto' ? autoBackendsForHost() : [requested]
}

function backendBinaryPath(backend, triplet) {
  return path.join(
    __dirname,
    '..',
    '..',
    '.build',
    'artifacts',
    'node',
    `${BINARY_NAME}_${backend}.${triplet}.node`,
  )
}

function backendNameMatches(value, backend) {
  return String(value ?? '').toLowerCase().includes(backend)
}

function backendAvailable(info, backend) {
  const compiled = info && info.compiled && info.compiled[backend] === true
  const gpuOffloadSupported = info && info.gpuOffloadSupported === true
  const availableBackends = Array.isArray(info?.availableBackends) ? info.availableBackends : []
  const devices = Array.isArray(info?.devices) ? info.devices : []

  return (
    compiled &&
    gpuOffloadSupported &&
    (availableBackends.some((item) => backendNameMatches(item?.name, backend)) ||
      devices.some((item) => backendNameMatches(item?.backendName, backend)))
  )
}

function assertBackendUsable(binding, backend) {
  if (backend === 'cpu') {
    return
  }
  if (typeof binding.backendObservabilityJson !== 'function') {
    throw new Error(`${backend} binding does not expose backendObservabilityJson()`)
  }

  const info = JSON.parse(binding.backendObservabilityJson(true))
  if (!backendAvailable(info, backend)) {
    throw new Error(
      `${backend} binding loaded, but no usable ${backend} backend was reported by llama.cpp`,
    )
  }
}

function loadCandidate(backend, triplet) {
  const binding = require(backendBinaryPath(backend, triplet))
  assertBackendUsable(binding, backend)
  activeBackend = backend
  return binding
}

function loadNativeBinding() {
  if (process.env.NAPI_RS_NATIVE_LIBRARY_PATH) {
    const binding = require(process.env.NAPI_RS_NATIVE_LIBRARY_PATH)
    const requested = (process.env.COGENTLM_NODE_BACKEND ?? 'cpu').toLowerCase()
    activeBackend = VALID_BACKENDS.has(requested) && requested !== 'auto' ? requested : 'cpu'
    return binding
  }

  const triplet = platformTriplet()
  const errors = []

  for (const backend of requestedBackends()) {
    try {
      return loadCandidate(backend, triplet)
    } catch (error) {
      errors.push({ backend, error })
    }
  }

  const detail = errors
    .map(({ backend, error }) => {
      const message = error && error.message ? error.message : String(error)
      return `${backend}: ${message}`
    })
    .join('\n')
  const message =
    `CogentLM failed to load a usable Node backend for ${process.platform} ${process.arch}.\n` +
    detail
  throw new Error(message, { cause: errors[errors.length - 1]?.error })
}

const binding = loadNativeBinding()

module.exports = binding
module.exports.getActiveBackend = () => activeBackend
