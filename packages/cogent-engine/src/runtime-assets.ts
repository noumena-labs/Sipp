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
