import type { SippClientOptions } from '../../engine/browser-client.js';
import type {
  RuntimeBundleDescriptor,
  RuntimeBundleFile,
} from '../../models/types.js';
import type { EngineModule } from '../../wasm/engine-module.js';
import { createSyncAccessHandleFS } from '../../wasm/sync-access-handle-fs.js';
import { attachCleanupFailures, releaseAll } from '../../utils/cleanup.js';

const DEFAULT_MAX_MODEL_BYTES = 8 * 1024 * 1024 * 1024;
const MODEL_MOUNT_DIR = '/sah_model';
const PROJECTOR_MOUNT_DIR = '/sah_projector';

function normalizeModelFileName(fileName: string): string {
  const trimmed = fileName.trim();
  if (!trimmed) {
    throw new Error('Model file name must not be empty.');
  }
  if (trimmed.includes('/') || trimmed.includes('\\') || trimmed.includes('..')) {
    throw new Error(
      `Invalid model file name "${fileName}". Provide a simple file name, not a path.`
    );
  }
  return trimmed;
}

/** Mounts model assets into one worker-owned Wasm runtime. */
export class WasmModelLoader {
  private mountedModelFiles: readonly RuntimeBundleFile[] = [];
  private mountedProjector: RuntimeBundleFile | null = null;
  private modelMountActive = false;
  private projectorMountActive = false;

  constructor(private readonly config: SippClientOptions) {}

  public cleanup(module: EngineModule): void {
    this.unmountAll(module);
  }

  public validateBundle(descriptor: RuntimeBundleDescriptor): void {
    if (descriptor.modelFiles.length === 0) {
      throw new Error('Model bundle must contain at least one model file.');
    }
    const files = descriptor.projector == null
      ? descriptor.modelFiles
      : [...descriptor.modelFiles, descriptor.projector];
    let totalBytes = 0;
    for (const file of files) {
      normalizeModelFileName(file.name);
      if (!Number.isSafeInteger(file.size) || file.size < 0) {
        throw new Error(`Invalid runtime file size for "${file.name}".`);
      }
      totalBytes += file.size;
    }
    const maxBytes = this.resolveMaxModelBytes();
    if (totalBytes > maxBytes) {
      throw new Error(
        `Total model size (${totalBytes} bytes) exceeds configured maxModelBytes (${maxBytes} bytes).`
      );
    }
  }

  public mountBundle(
    module: EngineModule,
    descriptor: RuntimeBundleDescriptor
  ): { modelPath: string; projectorPath: string | null } {
    this.mountedModelFiles = descriptor.modelFiles;
    this.mountedProjector = descriptor.projector ?? null;
    try {
      const modelPath = this.mountFiles(module, MODEL_MOUNT_DIR, descriptor.modelFiles);
      this.modelMountActive = true;
      const projectorPath = descriptor.projector == null
        ? null
        : this.mountFiles(module, PROJECTOR_MOUNT_DIR, [descriptor.projector]);
      this.projectorMountActive = projectorPath != null;
      return { modelPath, projectorPath };
    } catch (error) {
      try {
        this.unmountAll(module);
      } catch (cleanupError) {
        throw attachCleanupFailures(error, cleanupError);
      }
      throw error;
    }
  }

  public closeBundle(descriptor: RuntimeBundleDescriptor): void {
    const projector = descriptor.projector;
    releaseAll('Failed to close the runtime model bundle.', [
      ...descriptor.modelFiles.map((file) => ({
        label: `close model handle "${file.name}"`,
        release: () => file.handle.close(),
      })),
      ...(projector == null
        ? []
        : [{
          label: `close projector handle "${projector.name}"`,
          release: () => projector.handle.close(),
        }]),
    ]);
  }

  private mountFiles(
    module: EngineModule,
    mountDirectory: string,
    files: readonly RuntimeBundleFile[]
  ): string {
    const normalizedFiles = files.map((file) => ({
      name: normalizeModelFileName(file.name),
      handle: file.handle,
      size: file.size,
    }));
    this.ensureDir(module, mountDirectory);
    const provider = createSyncAccessHandleFS(module);
    module.FS.mount(provider, { files: normalizedFiles }, mountDirectory);
    return `${mountDirectory}/${normalizedFiles[0].name}`;
  }

  private unmountAll(module: EngineModule): void {
    const modelFiles = this.mountedModelFiles;
    const projector = this.mountedProjector;
    const modelMountActive = this.modelMountActive;
    const projectorMountActive = this.projectorMountActive;
    this.mountedModelFiles = [];
    this.mountedProjector = null;
    this.modelMountActive = false;
    this.projectorMountActive = false;

    releaseAll('Failed to release the mounted runtime model bundle.', [
      ...(projectorMountActive
        ? [{
          label: `unmount ${PROJECTOR_MOUNT_DIR}`,
          release: () => module.FS.unmount(PROJECTOR_MOUNT_DIR),
        }]
        : []),
      ...(modelMountActive
        ? [{
          label: `unmount ${MODEL_MOUNT_DIR}`,
          release: () => module.FS.unmount(MODEL_MOUNT_DIR),
        }]
        : []),
      ...modelFiles.map((file) => ({
        label: `close model handle "${file.name}"`,
        release: () => file.handle.close(),
      })),
      ...(projector == null
        ? []
        : [{
          label: `close projector handle "${projector.name}"`,
          release: () => projector.handle.close(),
        }]),
    ]);
  }

  private ensureDir(module: EngineModule, path: string): void {
    if (!module.FS.analyzePath(path).exists) {
      module.FS.mkdir(path);
    }
  }

  private resolveMaxModelBytes(): number {
    const maxModelBytes = this.config.maxModelBytes ?? DEFAULT_MAX_MODEL_BYTES;
    if (!Number.isInteger(maxModelBytes) || maxModelBytes <= 0) {
      throw new Error('"maxModelBytes" must be a positive integer.');
    }
    return maxModelBytes;
  }
}
