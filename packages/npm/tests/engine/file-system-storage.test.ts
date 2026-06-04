import assert from 'node:assert/strict';
import test from 'node:test';

import { FileSystemStorage } from '../../src/engine/file-system-storage.js';

async function withNavigatorStorage<T>(
  storage: Navigator['storage'],
  run: () => Promise<T>
): Promise<T> {
  const originalNavigator = globalThis.navigator;
  const hadNavigator = 'navigator' in globalThis;
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: { storage },
  });
  try {
    return await run();
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
}

test('FileSystemStorage removes partial files when streamToDisk fails', async () => {
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

  await withNavigatorStorage({
    getDirectory: async () => ({
      getDirectoryHandle: async () => root,
    }),
  } as unknown as Navigator['storage'], async () => {
    const storage = new FileSystemStorage();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(Uint8Array.from([1, 2, 3]));
        controller.close();
      },
    });
    await assert.rejects(storage.streamToDisk('model.gguf', stream), /disk write failed/);
    assert.deepEqual(removedEntries, ['model.gguf']);
  });
});

test('FileSystemStorage batches small stream chunks into one OPFS write', async () => {
  const writes: Uint8Array[] = [];

  const writable = new WritableStream<Uint8Array>({
    write(chunk) {
      writes.push(chunk);
    },
    close() {},
    abort() {},
  });
  const root = {
    getFileHandle: async () => ({
      createWritable: async () => writable,
      getFile: async () => new File([Uint8Array.from([1, 2, 3])], 'model.gguf'),
    }),
    removeEntry: async () => {},
  };

  await withNavigatorStorage({
    getDirectory: async () => ({
      getDirectoryHandle: async () => root,
    }),
  } as unknown as Navigator['storage'], async () => {
    const storage = new FileSystemStorage();
    const progress: number[] = [];
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(Uint8Array.from([1]));
        controller.enqueue(Uint8Array.from([2]));
        controller.enqueue(Uint8Array.from([3]));
        controller.close();
      },
    });
    await storage.streamToDisk('model.gguf', stream, (bytes) => progress.push(bytes));
    assert.equal(writes.length, 1);
    assert.deepEqual([...writes[0]], [1, 2, 3]);
    assert.deepEqual(progress, [3]);
  });
});

test('FileSystemStorage prefers OPFS sync access handles when available', async () => {
  const writes: Uint8Array[] = [];
  let flushed = false;
  let closed = false;

  const root = {
    getFileHandle: async () => ({
      createSyncAccessHandle: async () => ({
        read: () => 0,
        write: (chunk: Uint8Array) => {
          writes.push(chunk.slice());
          return chunk.byteLength;
        },
        truncate: () => {},
        flush: () => {
          flushed = true;
        },
        close: () => {
          closed = true;
        },
      }),
      createWritable: async () => {
        throw new Error('async writable path should not be used');
      },
      getFile: async () => new File([Uint8Array.from([1, 2])], 'model.gguf'),
    }),
    removeEntry: async () => {},
  };

  await withNavigatorStorage({
    getDirectory: async () => ({
      getDirectoryHandle: async () => root,
    }),
  } as unknown as Navigator['storage'], async () => {
    const storage = new FileSystemStorage();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(Uint8Array.from([1]));
        controller.enqueue(Uint8Array.from([2]));
        controller.close();
      },
    });
    await storage.streamToDisk('model.gguf', stream);
    assert.deepEqual(writes.map((chunk) => [...chunk]), [[1, 2]]);
    assert.equal(flushed, true);
    assert.equal(closed, true);
  });
});
