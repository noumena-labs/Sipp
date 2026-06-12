import type { SippClientOptions } from '../../engine/browser-client.js';
import type {
  InternalBundleDescriptor,
  ModelBundleShard,
  StagedModelBundle,
  StageModelBundleOptions,
} from '../../models/types.js';
import type { EngineModule } from '../../wasm/engine-module.js';
import { createSyncAccessHandleFS } from '../../wasm/sync-access-handle-fs.js';
import { createAbortError } from '../../utils/abort.js';

const DEFAULT_MAX_MODEL_BYTES = 8 * 1024 * 1024 * 1024;
const MODEL_MOUNT_DIR = '/sah_model';
const PROJECTOR_MOUNT_DIR = '/memfs_projector';

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

export class MainThreadModelLoader {
  private mountedShards: ModelBundleShard[] = [];
  private mountedProjectorPath: string | null = null;

  constructor(private readonly config: SippClientOptions) {}

  public cleanup(module: EngineModule): void {
    this.unmountAll(module);
  }

  public async stageModelBundle(
    module: EngineModule,
    descriptor: InternalBundleDescriptor,
    options: StageModelBundleOptions = {}
  ): Promise<StagedModelBundle> {
    this.unmountAll(module);
    if (options.signal?.aborted) {
      throw createAbortError('Model load aborted.');
    }
    if (descriptor.shards.length === 0) {
      throw new Error('Model bundle must contain at least one shard.');
    }

    const totalBytes = descriptor.shards.reduce((sum, shard) => sum + shard.size, 0);
    const maxBytes = this.resolveMaxModelBytes();
    if (totalBytes > maxBytes) {
      throw new Error(
        `Total model size (${totalBytes} bytes) exceeds configured maxModelBytes (${maxBytes} bytes).`
      );
    }

    const modelPath = this.mountShards(module, descriptor.shards);

    let projectorPath: string | null = null;
    if (descriptor.projector != null) {
      try {
        projectorPath = await this.stageProjectorFile(
          module,
          descriptor.projector.file,
          descriptor.projector.destFileName,
          options.signal
        );
      } catch (error) {
        this.unmountAll(module);
        throw error;
      }
    }

    return {
      sourceKind: 'installed',
      modelPath,
      projectorPath,
      isVisionModel: descriptor.detection.inspection.visionCapable,
      projectorStatus: descriptor.detection.inspection.visionCapable
        ? projectorPath != null
          ? 'paired'
          : 'missing'
        : 'not-required',
      modelName: descriptor.detection.modelName,
      detectionMethod: descriptor.detection.detectionMethod,
      modelType: descriptor.detection.modelType,
      modelArchitecture: descriptor.detection.modelArchitecture,
    };
  }

  private mountShards(module: EngineModule, shards: ModelBundleShard[]): string {
    const normalizedShards = shards.map((shard) => ({
      name: normalizeModelFileName(shard.name),
      handle: shard.handle,
      size: shard.size,
    }));
    this.ensureDir(module, MODEL_MOUNT_DIR);
    const provider = createSyncAccessHandleFS(module);
    module.FS.mount(provider, { files: normalizedShards }, MODEL_MOUNT_DIR);
    this.mountedShards = normalizedShards;
    return `${MODEL_MOUNT_DIR}/${normalizedShards[0].name}`;
  }

  private async stageProjectorFile(
    module: EngineModule,
    file: File,
    destFileName: string | undefined,
    signal: AbortSignal | undefined
  ): Promise<string> {
    if (signal?.aborted) {
      throw createAbortError('Model load aborted.');
    }
    this.ensureDir(module, PROJECTOR_MOUNT_DIR);
    const fileName = normalizeModelFileName(destFileName ?? file.name ?? 'mmproj.gguf');
    const path = `${PROJECTOR_MOUNT_DIR}/${fileName}`;
    this.removeFileIfExists(module, path);
    const data = new Uint8Array(await file.arrayBuffer());
    if (signal?.aborted) {
      throw createAbortError('Model load aborted.');
    }
    module.FS.writeFile(path, data);
    this.mountedProjectorPath = path;
    return path;
  }

  private unmountAll(module: EngineModule): void {
    if (this.mountedShards.length > 0) {
      try {
        module.FS.unmount(MODEL_MOUNT_DIR);
      } catch {}
      for (const shard of this.mountedShards) {
        try {
          shard.handle.close();
        } catch {}
      }
      this.mountedShards = [];
    }
    if (this.mountedProjectorPath != null) {
      this.removeFileIfExists(module, this.mountedProjectorPath);
      this.mountedProjectorPath = null;
    }
  }

  private ensureDir(module: EngineModule, path: string): void {
    if (!module.FS.analyzePath(path).exists) {
      module.FS.mkdir(path);
    }
  }

  private removeFileIfExists(module: EngineModule, path: string): void {
    if (module.FS.analyzePath(path).exists) {
      module.FS.unlink(path);
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
