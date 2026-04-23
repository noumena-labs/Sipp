export type ModelBundleSourceKind = 'file' | 'files';

export type ModelBundleProjectorStatus =
  | 'not-required'
  | 'explicit'
  | 'paired'
  | 'missing';

export type ModelDetectionMethod = 'filename' | 'gguf-metadata' | 'none';

export interface ModelBundleFileProjectorDescriptor {
  kind: 'file';
  file: File;
  destFileName?: string;
}

export interface FileBundleDescriptor {
  kind: 'file';
  file: File;
  destFileName?: string;
  projector?: ModelBundleFileProjectorDescriptor;
}

export interface FilesBundleDescriptor {
  kind: 'files';
  files: File[];
  projector?: ModelBundleFileProjectorDescriptor;
}

export type InternalBundleDescriptor =
  | FileBundleDescriptor
  | FilesBundleDescriptor;

export interface StageModelBundleOptions {
  signal?: AbortSignal;
}

export interface ModelDetectionResult {
  isVisionModel: boolean;
  isProjector: boolean;
  suggestedProjectorUrl: string | null;
  detectionMethod: ModelDetectionMethod;
  modelName: string;
  modelType: string | null;
  modelArchitecture: string | null;
}

export interface LocalProjectorResolutionResult<T> {
  modelFiles: T[];
  projectorFile: T | null;
  candidateFileNames: string[];
  errorMessage: string | null;
}

export interface StagedModelBundle {
  sourceKind: ModelBundleSourceKind;
  modelPath: string;
  multimodalProjectorPath: string | null;
  isVisionModel: boolean;
  projectorStatus: ModelBundleProjectorStatus;
  modelName: string;
  detectionMethod: ModelDetectionMethod;
  modelType: string | null;
  modelArchitecture: string | null;
}
