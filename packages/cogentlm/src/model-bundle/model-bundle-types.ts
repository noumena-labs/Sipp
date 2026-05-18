export type ModelBundleSourceKind = 'file' | 'files';

export type ModelBundleProjectorStatus =
  | 'not-required'
  | 'explicit'
  | 'paired'
  | 'missing';

export type ModelDetectionMethod = 'gguf-metadata' | 'none';
export type AssetRole = 'model' | 'projector' | 'unknown';

export interface AssetInspection {
  version: 1;
  role: AssetRole;
  architecture: string | null;
  visionCapable: boolean;
  compatibleVisionProjectorTypes: string[];
  providedVisionProjectorType: string | null;
}

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
  detection?: ModelDetectionResult;
}

export interface FilesBundleDescriptor {
  kind: 'files';
  files: File[];
  projector?: ModelBundleFileProjectorDescriptor;
  detection?: ModelDetectionResult;
}

export type InternalBundleDescriptor =
  | FileBundleDescriptor
  | FilesBundleDescriptor;

export interface StageModelBundleOptions {
  signal?: AbortSignal;
}

export interface ModelDetectionResult {
  inspection: AssetInspection;
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
  projectorPath: string | null;
  isVisionModel: boolean;
  projectorStatus: ModelBundleProjectorStatus;
  modelName: string;
  detectionMethod: ModelDetectionMethod;
  modelType: string | null;
  modelArchitecture: string | null;
}
