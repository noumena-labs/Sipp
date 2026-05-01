import test from 'node:test';
import assert from 'node:assert/strict';
import { WasmBridge } from './wasm-bridge.js';
import type { EngineModule } from './engine-module.js';

function createModule(
  calls: Record<string, number>,
  strings: Record<number, string>
): EngineModule {
  return {
    FS: {
      analyzePath: () => ({ exists: false }),
      mkdir: () => {},
      writeFile: () => {},
      unlink: () => {},
      mount: () => {},
      unmount: () => {},
    },
    WORKERFS: {},
    HEAP32: new Int32Array(8),
    HEAPF64: new Float64Array(8),
    HEAPU8: new Uint8Array(8),
    _free: () => {},
    _malloc: () => 0,
    ccall: (ident: string) => calls[ident] ?? 0,
    UTF8ToString: (ptr: number | bigint) => strings[Number(ptr)] ?? '',
  };
}

test('WasmBridge normalizes empty media markers to null', () => {
  const bridge = new WasmBridge(
    createModule(
      {
        CE_GetMediaMarker: 1,
      },
      {
        1: '',
      }
    )
  );

  assert.equal(bridge.readMediaMarker(), null);
});

test('WasmBridge normalizes empty chat templates to null', () => {
  const bridge = new WasmBridge(
    createModule(
      {
        CE_GetChatTemplate: 2,
      },
      {
        2: '',
      }
    )
  );

  assert.equal(bridge.readNativeChatTemplate(), null);
});
