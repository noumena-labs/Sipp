import assert from 'node:assert/strict';
import test from 'node:test';

import { FileSystemStorage } from './file-system-storage.js';

test('FileSystemStorage removes partial files when streamToDisk fails', async () => {
  const originalNavigator = globalThis.navigator;
  const hadNavigator = 'navigator' in globalThis;
  const removedEntries: string[] = [];

  const writable = new WritableStream<Uint8Array>({
    write() {
      throw new Error('disk write failed');
    },
    abort() {},
  });
  const root = {
    getFileHandle: async () => ({
      createWritable: async () => writable,
      getFile: async () => new File([Uint8Array.from([1])], 'model.gguf'),
    }),
    removeEntry: async (fileName: string) => {
      removedEntries.push(fileName);
    },
  };

  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      storage: {
        getDirectory: async () => ({
          getDirectoryHandle: async () => root,
        }),
      },
    },
  });

  const storage = new FileSystemStorage();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(Uint8Array.from([1, 2, 3]));
      controller.close();
    },
  });

  try {
    await assert.rejects(storage.streamToDisk('model.gguf', stream), /disk write failed/);
    assert.deepEqual(removedEntries, ['model.gguf']);
  } finally {
    if (hadNavigator) {
      Object.defineProperty(globalThis, 'navigator', {
        configurable: true,
        value: originalNavigator,
      });
    } else {
      delete (globalThis as { navigator?: Navigator }).navigator;
    }
  }
});
