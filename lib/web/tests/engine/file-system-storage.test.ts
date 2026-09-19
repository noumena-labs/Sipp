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

test('FileSystemStorage scopes files under a custom OPFS storage root', async () => {
  const directories: string[] = [];
  const fileNames: string[] = [];
  type TestDirectory = {
    getDirectoryHandle(name: string): Promise<TestDirectory>;
    getFileHandle(name: string): Promise<{ getFile(): Promise<File> }>;
  };
  const directory: TestDirectory = {
    getDirectoryHandle: async (name: string) => {
      directories.push(name);
      return directory;
    },
    getFileHandle: async (name: string) => {
      fileNames.push(name);
      return {
        getFile: async () => new File(['model'], name),
      };
    },
  };

  await withNavigatorStorage({
    getDirectory: async () => directory,
  } as unknown as Navigator['storage'], async () => {
    const storage = new FileSystemStorage('tenant-a/models');

    const stored = await storage.getFile('model.gguf');

    assert.equal(stored?.name, 'model.gguf');
    assert.deepEqual(directories, ['tenant-a', 'models']);
    assert.deepEqual(fileNames, ['model.gguf']);
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
    await storage.streamToDisk('model.gguf', stream, {
      onProgress: (bytes) => progress.push(bytes),
    });
    assert.equal(writes.length, 1);
    assert.deepEqual([...writes[0]], [1, 2, 3]);
    assert.deepEqual(progress, [3]);
  });
});

test('FileSystemStorage prefers OPFS sync access handles when available', async () => {
  const writes: Uint8Array[] = [];
  let flushCount = 0;
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
          flushCount += 1;
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
    assert.equal(flushCount, 2);
    assert.equal(closed, true);
  });
});

test('FileSystemStorage keeps partial file when keepFileOnFailure is true', async () => {
  const removedEntries: string[] = [];
  let closed = false;

  const writable = new WritableStream<Uint8Array>({
    write() {
      throw new Error('disk write failed');
    },
    close() {
      closed = true;
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
    await assert.rejects(
      storage.streamToDisk('model.gguf', stream, { keepFileOnFailure: true }),
      /disk write failed/
    );
    assert.deepEqual(removedEntries, []);
  });
});

test('FileSystemStorage resumes appending at startOffset', async () => {
  const writtenOffsets: number[] = [];
  const writes: Uint8Array[] = [];
  let truncatedSize: number | null = null;
  const progress: number[] = [];

  const root = {
    getFileHandle: async () => ({
      createSyncAccessHandle: async () => ({
        write: (chunk: Uint8Array, options?: { at?: number }) => {
          writtenOffsets.push(options?.at ?? 0);
          writes.push(chunk);
          return chunk.byteLength;
        },
        truncate: (size: number) => {
          truncatedSize = size;
        },
        flush: () => {},
        close: () => {},
      }),
      createWritable: async () => {
        throw new Error('async writable path should not be used');
      },
      getFile: async () => new File([Uint8Array.from([1, 2, 3, 4, 5])], 'model.gguf'),
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
        controller.enqueue(Uint8Array.from([4, 5]));
        controller.close();
      },
    });
    const file = await storage.streamToDisk('model.gguf', stream, {
      startOffset: 3,
      onProgress: (bytes) => progress.push(bytes),
    });
    assert.equal(truncatedSize, null);
    assert.deepEqual(writtenOffsets, [3]);
    assert.deepEqual(writes.map((chunk) => [...chunk]), [[4, 5]]);
    assert.deepEqual(progress, [3, 5]);
    assert.equal(file.size, 5);
  });
});

test('FileSystemStorage resumes async OPFS writes without truncating existing data', async () => {
  const writes: Uint8Array[] = [];
  let keepExistingData = false;
  let seekOffset: number | null = null;
  const writable = {
    async seek(offset: number) {
      seekOffset = offset;
    },
    async write(chunk: Uint8Array) {
      writes.push(chunk.slice());
    },
    async close() {},
    async abort() {},
  };
  const root = {
    getFileHandle: async () => ({
      createWritable: async (options?: { keepExistingData?: boolean }) => {
        keepExistingData = options?.keepExistingData === true;
        return writable;
      },
      getFile: async () => new File([Uint8Array.from([1, 2, 3, 4, 5])], 'model.gguf'),
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
        controller.enqueue(Uint8Array.from([4, 5]));
        controller.close();
      },
    });

    await storage.streamToDisk('model.gguf', stream, { startOffset: 3 });

    assert.equal(keepExistingData, true);
    assert.equal(seekOffset, 3);
    assert.deepEqual(writes.map((chunk) => [...chunk]), [[4, 5]]);
  });
});

test('FileSystemStorage aborts with stall timeout error when stream stalls', async () => {
  const root = {
    getFileHandle: async () => ({
      createWritable: async () =>
        new WritableStream<Uint8Array>({
          write() {},
          close() {},
          abort() {},
        }),
      getFile: async () => new File([], 'model.gguf'),
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
        controller.enqueue(Uint8Array.from([1, 2]));
        // Stall without closing or enqueuing more chunks
      },
    });
    await assert.rejects(
      storage.streamToDisk('model.gguf', stream, { stallTimeoutMs: 50 }),
      (error: unknown) => {
        const err = error as { code?: string; name?: string; message?: string };
        return (
          err.code === 'STALL_TIMEOUT' &&
          err.name === 'TimeoutError' &&
          /Download stream stalled/.test(err.message ?? '')
        );
      }
    );
  });
});
