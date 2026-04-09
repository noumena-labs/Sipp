export interface BundledRuntimeUrls {
  moduleUrl: string;
  wasmUrl: string;
}

/**
 * Returns the runtime asset URLs shipped with this package.
 * Use this when you want the package's default wasm runtime layout.
 */
export function getBundledRuntimeUrls(): BundledRuntimeUrls {
  return {
    moduleUrl: new URL('../wasm/cogent-engine-wasm.js', import.meta.url).toString(),
    wasmUrl: new URL('../wasm/cogent-engine-wasm.wasm', import.meta.url).toString()
  };
}

/**
 * Returns the Bun-specific runtime asset URLs shipped with this package.
 * Use this when Bun needs a separate compatibility build from the browser default.
 */
export function getBundledBunRuntimeUrls(): BundledRuntimeUrls {
  return {
    moduleUrl: new URL('../wasm-bun/cogent-engine-wasm.js', import.meta.url).toString(),
    wasmUrl: new URL('../wasm-bun/cogent-engine-wasm.wasm', import.meta.url).toString()
  };
}
