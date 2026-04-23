import { ModelLoadInfo } from '../core/inference-types.js';

export type ModelBundleSourceKind = 'url' | 'urls' | 'file' | 'files';

export type ModelBundleProjectorStatus =
  | 'not-required'
  | 'explicit'
  | 'paired'
  | 'missing';

export type ModelDetectionMethod = 'filename' | 'url' | 'gguf-metadata' | 'none';

export interface ModelBundleUrlProjectorDescriptor {
  kind: 'url';
  url: string;
  destFileName?: string;
}

export interface ModelBundleFileProjectorDescriptor {
  kind: 'file';
  file: File;
  destFileName?: string;
}

export type ModelBundleProjectorDescriptor =
  | ModelBundleUrlProjectorDescriptor
  | ModelBundleFileProjectorDescriptor;

export interface UrlBundleDescriptor {
  kind: 'url';
  url: string;
  destFileName?: string;
  projector?: ModelBundleProjectorDescriptor;
}

export interface UrlsBundleDescriptor {
  kind: 'urls';
  urls: string[];
  projector?: ModelBundleProjectorDescriptor;
}

export interface FileBundleDescriptor {
  kind: 'file';
  file: File;
  destFileName?: string;
  projector?: ModelBundleProjectorDescriptor;
}

export interface FilesBundleDescriptor {
  kind: 'files';
  files: File[];
  projector?: ModelBundleProjectorDescriptor;
}

export type InternalBundleDescriptor =
  | UrlBundleDescriptor
  | UrlsBundleDescriptor
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
  modelLoadInfo: ModelLoadInfo | null;
  projectorLoadInfo: ModelLoadInfo | null;
}
