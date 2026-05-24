import type { OpfsSyncAccessHandle } from '../storage/file-system-storage.js';

export type ModelBundleSourceKind = 'installed';

export type ModelBundleProjectorStatus =
  | 'not-required'
  | 'explicit'
  | 'paired'
  | 'missing';

export type ModelDetectionMethod = 'gguf-metadata' | 'none';
export type AssetRole = 'model' | 'projector' | 'unknown';

export interface GgufMetadataInspection {
  generalType: string | null;
  generalArchitecture: string | null;
  poolingType: number | null;
  clipProjectorType: string | null;
  clipVisionProjectorType: string | null;
  clipHasVisionEncoder: boolean | null;
  scannedKeyCount: number;
  stoppedEarlyAtKey: string | null;
}

export interface AssetInspection {
  version: 1;
  role: AssetRole;
  architecture: string | null;
  visionCapable: boolean;
  compatibleVisionProjectorTypes: string[];
  providedVisionProjectorType: string | null;
}

export interface ModelBundleFileProjectorDescriptor {
  file: File;
  destFileName?: string;
}

export interface ModelBundleShard {
  name: string;
  handle: OpfsSyncAccessHandle;
  size: number;
}

export interface InternalBundleDescriptor {
  shards: ModelBundleShard[];
  projector?: ModelBundleFileProjectorDescriptor;
  detection: ModelDetectionResult;
}

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
