#!/usr/bin/env node
'use strict'

// Runs after npm installs @sipphq/sipp-server, and can be run again by hand.
// When no CUDA 13 is installed and the NVIDIA driver can run it, it downloads
// NVIDIA's CUDA 13 libraries once into ~/.sipp/cuda. It never fails the
// install. SIPP_SKIP_CUDA_DOWNLOAD=1 skips it.

const log = (message) => process.stderr.write(`sipp-server: ${message}\n`)

async function main() {
  if (process.env.SIPP_SKIP_CUDA_DOWNLOAD) {
    return
  }
  const {
    cudaDownloadBlocker,
    cudaDownloadBytes,
    installCudaBundle,
    locateCuda,
    nvidiaDriver,
  } = require('./cuda.js')
  const driver = nvidiaDriver()
  // Machines without an NVIDIA GPU never use CUDA, so they get no notice either.
  if (!driver || locateCuda()) {
    return
  }
  const blocker = cudaDownloadBlocker(driver)
  if (blocker) {
    log(`not downloading CUDA 13: ${blocker}`)
    return
  }
  log(`downloading NVIDIA's CUDA 13 libraries (${Math.round(cudaDownloadBytes() / 2 ** 20)} MiB), once`)
  let shown = Date.now()
  const directory = await installCudaBundle({
    onProgress: ({ received, total }) => {
      if (Date.now() - shown < 10_000) {
        return
      }
      shown = Date.now()
      log(`CUDA 13: ${Math.floor((received * 100) / total)}%`)
    },
  })
  log(`CUDA 13 libraries ready in ${directory}`)
}

main().catch((error) => {
  log(`could not download CUDA 13 (${error instanceof Error ? error.message : String(error)})`)
})
