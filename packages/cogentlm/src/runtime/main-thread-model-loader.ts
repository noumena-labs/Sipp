import { CogentConfig } from '../cogent-config.js';
import {
  detectModelFromGgufFile,
  resolveLocalModelAndProjectorFiles,
} from '../model-bundle/model-bundle-detection.js';
import {
  InternalBundleDescriptor,
  ModelBundleFileProjectorDescriptor,
  ModelDetectionResult,
  ModelBundleProjectorStatus,
  StagedModelBundle,
  StageModelBundleOptions,
} from '../model-bundle/model-bundle-types.js';
import {
  DEFAULT_MAX_MODEL_BYTES,
  MountableModelFile,
  createMountableModelFile,
  normalizeModelFileName,
} from './main-thread-runtime-constants.js';
import { EngineModule } from '../wasm/engine-module.js';
import { createAbortError } from '../utils/abort.js';

interface FileAssetRequest {
  file: File;
  mountFileName: string;
}

interface LoadedAssetSet {
  files: MountableModelFile[];
}

interface ResolvedAssetRequest {
  mountFileName: string;
  file: File;
}

interface ResolvedBundlePlan {
  detection: ModelDetectionResult;
  modelAssets: ResolvedAssetRequest[];
  projectorAsset: ResolvedAssetRequest | null;
  projectorStatus: ModelBundleProjectorStatus;
}

export class MainThreadModelLoader {
  private loadedAssetPaths: string[] = [];
  private activeMountPath: string | null = null;

  constructor(private readonly config: CogentConfig) {}

  public cleanupAfterEngineInit(module: EngineModule): void {
    this.removeAllLoadedAssets(module);
  }

  public cleanupAfterClose(module: EngineModule): void {
    this.removeAllLoadedAssets(module);
  }

  public async stageModelBundle(
    module: EngineModule,
    descriptor: InternalBundleDescriptor,
    options: StageModelBundleOptions = {}
  ): Promise<StagedModelBundle> {
    const plan = await this.resolveBundlePlan(descriptor, options.signal);
    const modelAssetSet = await this.loadResolvedAssetSet(plan.modelAssets, options.signal);
    const projectorAssetSet =
      plan.projectorAsset == null
        ? null
        : await this.loadResolvedAssetSet([plan.projectorAsset], options.signal);

    const mountedPaths = await this.mountAssetFiles(module, modelAssetSet.files);
    const modelPath = mountedPaths[0];
    let projectorPath: string | null = null;
    try {
      projectorPath =
        projectorAssetSet == null
          ? null
          : await this.stageProjectorFile(module, projectorAssetSet.files[0], options.signal);
    } catch (error) {
      this.removeAllLoadedAssets(module);
      throw error;
    }

    return {
      sourceKind: descriptor.kind,
      modelPath,
      multimodalProjectorPath: projectorPath,
      isVisionModel: plan.detection.inspection.visionCapable,
      projectorStatus: plan.projectorStatus,
      modelName: plan.detection.modelName,
      detectionMethod: plan.detection.detectionMethod,
      modelType: plan.detection.modelType,
      modelArchitecture: plan.detection.modelArchitecture,
    };
  }

  private async resolveBundlePlan(
    descriptor: InternalBundleDescriptor,
    signal?: AbortSignal
  ): Promise<ResolvedBundlePlan> {
    switch (descriptor.kind) {
      case 'file': {
        const detection = await detectModelFromGgufFile(descriptor.file, signal);
        this.ensureNotProjectorSource(
          detection.modelName,
          detection.inspection.role === 'projector'
        );
        return {
          detection,
          modelAssets: [
            {
              file: descriptor.file,
              mountFileName: descriptor.destFileName ?? descriptor.file.name,
            },
          ],
          projectorAsset: this.resolveExplicitProjectorAsset(descriptor.projector),
          projectorStatus: this.resolveProjectorStatus(
            detection.inspection.visionCapable,
            descriptor.projector == null ? null : 'explicit'
          ),
        };
      }
      case 'files': {
        if (!descriptor.files || descriptor.files.length === 0) {
          throw new Error('Model bundle file list must not be empty.');
        }
        const explicitProjectorAsset = this.resolveExplicitProjectorAsset(descriptor.projector);
        const localResolution =
          explicitProjectorAsset == null
            ? await resolveLocalModelAndProjectorFiles(descriptor.files, signal)
            : {
                modelFiles: [...descriptor.files],
                projectorFile: null,
                candidateFileNames: [],
                errorMessage: null,
              };

        if (localResolution.errorMessage != null) {
          throw new Error(localResolution.errorMessage);
        }
        if (localResolution.modelFiles.length === 0) {
          throw new Error('Model bundle file list does not contain any model GGUF files.');
        }

        const sortedModelFiles = [...localResolution.modelFiles].sort((left, right) =>
          normalizeModelFileName(left.name).localeCompare(normalizeModelFileName(right.name))
        );
        const detectionFile = sortedModelFiles[0];
        const detection = await detectModelFromGgufFile(detectionFile, signal);
        this.ensureNotProjectorSource(
          detection.modelName,
          detection.inspection.role === 'projector'
        );

        return {
          detection,
          modelAssets: sortedModelFiles.map((file) => ({
            file,
            mountFileName: file.name,
          })),
          projectorAsset:
            explicitProjectorAsset ??
            (localResolution.projectorFile == null
              ? null
              : {
                  file: localResolution.projectorFile,
                  mountFileName: localResolution.projectorFile.name,
                }),
          projectorStatus:
            explicitProjectorAsset != null
              ? this.resolveProjectorStatus(detection.inspection.visionCapable, 'explicit')
              : localResolution.projectorFile != null
                ? 'paired'
                : this.resolveProjectorStatus(detection.inspection.visionCapable, null),
        };
      }
    }
  }

  private resolveExplicitProjectorAsset(
    projector: ModelBundleFileProjectorDescriptor | undefined
  ): ResolvedAssetRequest | null {
    if (projector == null) {
      return null;
    }
    return {
      file: projector.file,
      mountFileName: projector.destFileName ?? projector.file.name,
    };
  }

  private resolveProjectorStatus(
    isVisionModel: boolean,
    resolvedStatus: Extract<ModelBundleProjectorStatus, 'explicit'> | null
  ): ModelBundleProjectorStatus {
    if (resolvedStatus != null) {
      return resolvedStatus;
    }
    return isVisionModel ? 'missing' : 'not-required';
  }

  private ensureNotProjectorSource(modelName: string, isProjector: boolean): void {
    if (isProjector) {
      throw new Error(
        `Model source "${modelName}" looks like a projector GGUF. Provide the main model GGUF instead.`
      );
    }
  }

  private async loadResolvedAssetSet(
    assets: ResolvedAssetRequest[],
    signal?: AbortSignal
  ): Promise<LoadedAssetSet> {
    if (assets.length === 0) {
      throw new Error('Asset set must not be empty.');
    }
    return this.loadFileAssetSet(
      assets.map((asset) => ({
        file: asset.file,
        mountFileName: asset.mountFileName,
      })),
      undefined,
      signal
    );
  }

  private async loadFileAssetSet(
    assets: FileAssetRequest[],
    onProgress?: (pct: number) => void,
    signal?: AbortSignal
  ): Promise<LoadedAssetSet> {
    if (!assets || assets.length === 0) {
      throw new Error('No file assets provided.');
    }
    if (signal?.aborted) {
      throw createAbortError('Model load aborted.');
    }

    const files = assets.map((asset) => createMountableModelFile(asset.file, asset.mountFileName));
    const byteLength = files.reduce((sum, file) => sum + file.size, 0);
    if (byteLength <= 0) {
      throw new Error('Model assets are empty.');
    }
    const maxModelBytes = this.resolveMaxModelBytes();
    if (byteLength > maxModelBytes) {
      throw new Error(
        `Total model size (${byteLength} bytes) exceeds configured maxModelBytes (${maxModelBytes} bytes).`
      );
    }

    onProgress?.(100);
    return {
      files,
    };
  }

  private resolveMaxModelBytes(): number {
    const maxModelBytes = this.config.maxModelBytes ?? DEFAULT_MAX_MODEL_BYTES;
    if (!Number.isInteger(maxModelBytes) || maxModelBytes <= 0) {
      throw new Error('"maxModelBytes" must be a positive integer.');
    }
    return maxModelBytes;
  }

  private removeFileIfExists(module: EngineModule, path: string): void {
    if (module.FS.analyzePath(path).exists) {
      module.FS.unlink(path);
    }
  }

  private async stageProjectorFile(
    module: EngineModule,
    file: MountableModelFile,
    signal?: AbortSignal,
    mountDir = '/memfs_projector'
  ): Promise<string> {
    if (signal?.aborted) {
      throw createAbortError('Model load aborted.');
    }

    const fs = module.FS;
    if (!fs.analyzePath(mountDir).exists) {
      fs.mkdir(mountDir);
    }

    const fileName = normalizeModelFileName(file.name || 'mmproj.gguf');
    const path = `${mountDir}/${fileName}`;
    this.removeFileIfExists(module, path);

    const data = new Uint8Array(await file.arrayBuffer());
    if (signal?.aborted) {
      throw createAbortError('Model load aborted.');
    }

    fs.writeFile(path, data);
    this.commitLoadedAssetPaths(module, [...this.loadedAssetPaths, path]);
    return path;
  }

  private removeAllLoadedAssets(module: EngineModule): void {
    for (const path of this.loadedAssetPaths) {
      if (this.activeMountPath && path.startsWith(this.activeMountPath)) {
        continue;
      }
      this.removeFileIfExists(module, path);
    }
    this.loadedAssetPaths = [];
    this.unmountActiveAssetSet(module);
  }

  private commitLoadedAssetPaths(module: EngineModule, paths: string[]): void {
    const newSet = new Set(paths);
    for (const path of this.loadedAssetPaths) {
      if (!newSet.has(path)) {
        if (this.activeMountPath && path.startsWith(this.activeMountPath)) {
          continue;
        }
        this.removeFileIfExists(module, path);
      }
    }
    this.loadedAssetPaths = [...paths];
  }

  private async mountAssetFiles(
    module: EngineModule,
    files: MountableModelFile[],
    mountDir = '/workerfs_model'
  ): Promise<string[]> {
    const fs = module.FS;
    const normalizedFiles = files.map((file) =>
      createMountableModelFile(file, file.name || 'model.gguf')
    );

    if (!fs.analyzePath(mountDir).exists) {
      fs.mkdir(mountDir);
    }
    this.unmountActiveAssetSet(module);

    if (!module.WORKERFS) {
      throw new Error(
        'WORKERFS is not available in the Emscripten module. Ensure the module was linked with -lworkerfs.js and WORKERFS is exported.'
      );
    }

    fs.mount(module.WORKERFS, { files: normalizedFiles }, mountDir);
    this.activeMountPath = mountDir;

    const mountedPaths = normalizedFiles.map(
      (file) => `${mountDir}/${file.name || 'model.gguf'}`
    );
    this.commitLoadedAssetPaths(module, mountedPaths);
    return mountedPaths;
  }

  private unmountActiveAssetSet(module: EngineModule): void {
    if (this.activeMountPath == null) {
      return;
    }
    const unmountedPath = this.activeMountPath;
    try {
      module.FS.unmount(this.activeMountPath);
    } catch {
      // Ignore stale unmount cleanup failures.
    }
    this.activeMountPath = null;
    this.loadedAssetPaths = this.loadedAssetPaths.filter(
      (path) => !path.startsWith(unmountedPath)
    );
  }
}
