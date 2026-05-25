import type { OpfsSyncAccessHandle } from '../storage/file-system-storage.js';
import type { EngineModule } from './engine-module.js';

export interface SyncAccessFile {
  name: string;
  handle: OpfsSyncAccessHandle;
  size: number;
}

export interface SyncAccessMountOptions {
  files: SyncAccessFile[];
}

interface FsNode {
  name: string;
  mode: number;
  parent: FsNode | null;
  mount: { mountpoint: string; opts: unknown };
  node_ops: unknown;
  stream_ops: unknown;
  timestamp: number;
  size?: number;
  contents?: Record<string, FsNode>;
  handle?: OpfsSyncAccessHandle;
}

interface FsStream {
  node: FsNode;
  position: number;
}

interface EmscriptenFsInternal {
  isDir(mode: number): boolean;
  isFile(mode: number): boolean;
  createNode(parent: FsNode | null, name: string, mode: number, dev: number): FsNode;
  ErrnoError: new (errno: number) => Error;
}

const S_IFREG = 0o100000;
const S_IFDIR = 0o040000;
const READ_MODE = 0o555;
const SEEK_SET = 0;
const SEEK_CUR = 1;
const SEEK_END = 2;
const ENOENT = 44;
const EISDIR = 31;
const EINVAL = 28;

/**
 * Emscripten FS provider backed by OPFS sync access handles.
 *
 * Each mounted file maps to one open OpfsSyncAccessHandle. `read` calls
 * `handle.read(view, { at })` directly into a Uint8Array view of the wasm
 * heap — one copy from OPFS storage into wasm linear memory, no intermediate
 * ArrayBuffer in JS heap. Replaces the prior WORKERFS path which paid two
 * copies (FileReaderSync → ArrayBuffer → HEAPU8.set) per read.
 */
export function createSyncAccessHandleFS(module: EngineModule): unknown {
  const fs = module.FS as unknown as EmscriptenFsInternal;

  const node_ops = {
    getattr(node: FsNode) {
      return {
        dev: 1,
        ino: 1,
        mode: node.mode,
        nlink: 1,
        uid: 0,
        gid: 0,
        rdev: 0,
        size: node.size ?? 0,
        atime: new Date(node.timestamp),
        mtime: new Date(node.timestamp),
        ctime: new Date(node.timestamp),
        blksize: 4096,
        blocks: Math.ceil((node.size ?? 0) / 4096),
      };
    },
    setattr() {
      // Read-only mount: ignore.
    },
    lookup(parent: FsNode, name: string): FsNode {
      const child = parent.contents?.[name];
      if (child == null) {
        throw new fs.ErrnoError(ENOENT);
      }
      return child;
    },
    readdir(node: FsNode): string[] {
      return ['.', '..', ...Object.keys(node.contents ?? {})];
    },
  };

  const stream_ops = {
    open(_stream: FsStream) {
      // No-op: the handle is already open from mount time.
    },
    close(_stream: FsStream) {
      // No-op: handles are closed at unmount.
    },
    read(
      stream: FsStream,
      buffer: Uint8Array | Int8Array,
      offset: number,
      length: number,
      position: number
    ): number {
      const handle = stream.node.handle;
      const size = stream.node.size ?? 0;
      if (handle == null) {
        throw new fs.ErrnoError(EISDIR);
      }
      if (position >= size) {
        return 0;
      }
      const toRead = Math.min(length, size - position);
      if (toRead <= 0) {
        return 0;
      }
      const view = new Uint8Array(buffer.buffer, buffer.byteOffset + offset, toRead);
      return handle.read(view, { at: position });
    },
    llseek(stream: FsStream, offset: number, whence: number): number {
      let next: number;
      switch (whence) {
        case SEEK_SET:
          next = offset;
          break;
        case SEEK_CUR:
          next = stream.position + offset;
          break;
        case SEEK_END:
          next = (stream.node.size ?? 0) + offset;
          break;
        default:
          throw new fs.ErrnoError(EINVAL);
      }
      if (next < 0) {
        throw new fs.ErrnoError(EINVAL);
      }
      stream.position = next;
      return next;
    },
  };

  return {
    mount(mount: { opts: SyncAccessMountOptions; mountpoint: string }): FsNode {
      const root = fs.createNode(null, '/', S_IFDIR | READ_MODE, 0);
      root.node_ops = node_ops;
      root.stream_ops = stream_ops;
      root.contents = {};
      for (const file of mount.opts.files) {
        if (file.name.includes('/') || file.name.includes('\\') || file.name === '..' || file.name === '.') {
          throw new Error(`Invalid file name for sync-access mount: "${file.name}"`);
        }
        const node = fs.createNode(root, file.name, S_IFREG | READ_MODE, 0);
        node.node_ops = node_ops;
        node.stream_ops = stream_ops;
        node.size = file.size;
        node.handle = file.handle;
        root.contents[file.name] = node;
      }
      return root;
    },
  };
}
