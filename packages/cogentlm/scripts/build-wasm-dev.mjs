process.env.CE_WASM_BUILD_LABEL ??= '[build-wasm:dev]';
process.env.CE_WASM_BUILD_DIR_NAME ??= 'build-wasm-dev';
process.env.CE_WASM_LTO_MODE ??= 'OFF';
process.env.CE_WASM_BUILD_PARALLEL_LEVEL ??= '8';

await import('./build-wasm.mjs');
