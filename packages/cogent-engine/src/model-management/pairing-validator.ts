import { detectModelFromGgufFile } from '../model-bundle/model-bundle-detection.js';
import { QueryError, type ModelModality, type ModelStatus } from './model-types.js';

export interface ClassifiedAssetFile {
  assetId: string;
  file: File;
  isProjector: boolean;
  isVisionModel: boolean;
  name: string;
}

export interface PairingPlan {
  modelAssetIds: string[];
  projectorAssetId?: string;
  name: string;
  modality: ModelModality;
  status: ModelStatus;
}

export class PairingValidator {
  public async classify(
    assetId: string,
    file: File,
    signal?: AbortSignal
  ): Promise<ClassifiedAssetFile> {
    const detection = await detectModelFromGgufFile(file, signal);
    return {
      assetId,
      file,
      isProjector: detection.isProjector,
      isVisionModel: detection.isVisionModel,
      name: detection.modelName,
    };
  }

  public resolve(files: ClassifiedAssetFile[], explicitProjectorId?: string): PairingPlan {
    if (files.length === 0) {
      throw new QueryError('INVALID_MODEL_SOURCE', 'No model assets were provided.');
    }

    const projectors = files.filter((file) => file.isProjector);
    if (projectors.length > 1) {
      throw new QueryError(
        'INVALID_MODEL_PAIRING',
        `Multiple projector assets were provided: ${projectors.map((file) => file.name).join(', ')}.`
      );
    }

    const projector =
      explicitProjectorId == null
        ? projectors[0] ?? null
        : files.find((file) => file.assetId === explicitProjectorId) ?? null;
    if (explicitProjectorId != null && projector == null) {
      throw new QueryError('INVALID_MODEL_PAIRING', 'Explicit projector asset was not installed.');
    }
    if (projector != null && !projector.isProjector) {
      throw new QueryError('INVALID_MODEL_PAIRING', `"${projector.name}" is not a projector asset.`);
    }

    const modelFiles = files.filter((file) => file.assetId !== projector?.assetId);
    if (modelFiles.length === 0) {
      throw new QueryError('INVALID_MODEL_PAIRING', 'Projector assets are not runnable models.');
    }

    const hasVisionBase = modelFiles.some((file) => file.isVisionModel);
    if (projector != null && !hasVisionBase) {
      throw new QueryError(
        'INVALID_MODEL_PAIRING',
        'A projector can only be attached to a vision-capable base model.'
      );
    }

    return {
      modelAssetIds: modelFiles.map((file) => file.assetId),
      projectorAssetId: projector?.assetId,
      name: modelFiles[0].name,
      modality: projector != null ? 'vision' : 'text',
      status: hasVisionBase && projector == null ? 'needs_projector' : 'ready',
    };
  }
}
