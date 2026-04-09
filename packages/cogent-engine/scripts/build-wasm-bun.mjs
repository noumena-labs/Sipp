process.env.CE_WASM_BUILD_LABEL ??= '[build-wasm:bun]';
process.env.CE_WASM_BUILD_DIR_NAME ??= 'build-bun-mem32';
process.env.CE_WASM_OUTPUT_SUBDIR ??= 'wasm-bun';
process.env.CE_WASM_MEM64 ??= '0';
process.env.CE_WASM_MAXIMUM_MEMORY ??= '4096MB';

await import('./build-wasm.mjs');
