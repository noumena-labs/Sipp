import type { AssetInspection } from '../bundle/model-bundle-types.js';
import type { ModelModality, ModelStatus } from './types.js';

export interface ClassifiedAsset {
  assetId: string;
  inspection: AssetInspection;
  name: string;
}

export interface ClassifiedAssetFile extends ClassifiedAsset {
  file: File;
}

export interface PairingPlan {
  modelAssetIds: string[];
  projectorAssetId?: string | null;
  name: string;
  modality: ModelModality;
  status: ModelStatus;
  compatibleVisionProjectorTypes: string[];
}
