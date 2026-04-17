import { ModelLoadInfo } from '../core/inference-types.js';

export type ModelBundleSourceKind = 'url' | 'urls' | 'file' | 'files';

export type ModelBundleProjectorStatus =
  | 'not-required'
  | 'explicit'
  | 'paired'
  | 'discovered'
  | 'missing';

export type ModelDetectionMethod = 'filename' | 'url' | 'hf-api' | 'gguf-metadata' | 'none';

export type ProjectorDiscoverySource = 'hf-api' | 'head-probe' | 'none';

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

export interface UrlModelBundleDescriptor {
  kind: 'url';
  url: string;
  destFileName?: string;
  projector?: ModelBundleProjectorDescriptor;
}

export interface UrlsModelBundleDescriptor {
  kind: 'urls';
  urls: string[];
  projector?: ModelBundleProjectorDescriptor;
}

export interface FileModelBundleDescriptor {
  kind: 'file';
  file: File;
  destFileName?: string;
  projector?: ModelBundleProjectorDescriptor;
}

export interface FilesModelBundleDescriptor {
  kind: 'files';
  files: File[];
  projector?: ModelBundleProjectorDescriptor;
}

export type ModelBundleDescriptor =
  | UrlModelBundleDescriptor
  | UrlsModelBundleDescriptor
  | FileModelBundleDescriptor
  | FilesModelBundleDescriptor;

export interface PrepareModelBundleOptions {
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

export interface ProjectorDiscoveryResult {
  projectorUrl: string | null;
  candidates: string[];
  source: ProjectorDiscoverySource;
  message: string;
}

export interface ParsedHuggingFaceModelUrl {
  org: string;
  repo: string;
  ref: string;
  filename: string;
  baseUrl: string;
}

export interface LocalProjectorResolutionResult<T> {
  modelFiles: T[];
  projectorFile: T | null;
  candidateFileNames: string[];
  errorMessage: string | null;
}

export interface PreparedModelBundle {
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
