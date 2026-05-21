import { detectModelFromGgufFile } from '../bundle/model-bundle-detection.js';
import type { AssetInspection } from '../bundle/model-bundle-types.js';
import { QueryError, type ModelModality, type ModelStatus } from './types.js';

export interface ClassifiedAssetFile {
  assetId: string;
  file: File;
  inspection: AssetInspection;
  name: string;
}

export interface PairingPlan {
  modelAssetIds: string[];
  projectorAssetId?: string;
  name: string;
  modality: ModelModality;
  status: ModelStatus;
  compatibleVisionProjectorTypes: string[];
}

interface BaseModelResolution {
  compatibleVisionProjectorTypes: string[];
  name: string;
  visionCapable: boolean;
}

interface AssetSelection {
  modelFiles: ClassifiedAssetFile[];
  projector: ClassifiedAssetFile | null;
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
      inspection: detection.inspection,
      name: detection.modelName,
    };
  }

  public resolve(files: ClassifiedAssetFile[], explicitProjectorId?: string): PairingPlan {
    if (explicitProjectorId != null) {
      return this.resolveExplicit(files, explicitProjectorId);
    }
    const selection = this.selectAssets(files);
    const base = this.resolveBaseModel(selection.modelFiles);
    return {
      modelAssetIds: selection.modelFiles.map((file) => file.assetId),
      name: base.name,
      modality: base.visionCapable ? 'vision' : 'text',
      status: base.visionCapable ? 'needs_projector' : 'ready',
      compatibleVisionProjectorTypes: base.compatibleVisionProjectorTypes,
    };
  }

  public resolveExplicit(
    files: ClassifiedAssetFile[],
    explicitProjectorId: string
  ): PairingPlan {
    const selection = this.selectAssets(files, explicitProjectorId);
    const projector = selection.projector;
    if (projector == null) {
      throw new QueryError('INVALID_MODEL_PAIRING', 'Explicit projector asset was not installed.');
    }
    const base = this.resolveBaseModel(selection.modelFiles);
    this.validateExplicitProjector(base, projector);
    return {
      modelAssetIds: selection.modelFiles.map((file) => file.assetId),
      projectorAssetId: projector.assetId,
      name: base.name,
      modality: 'vision',
      status: 'ready',
      compatibleVisionProjectorTypes: base.compatibleVisionProjectorTypes,
    };
  }

  private selectAssets(
    files: ClassifiedAssetFile[],
    explicitProjectorId?: string
  ): AssetSelection {
    if (files.length === 0) {
      throw new QueryError('INVALID_MODEL_SOURCE', 'No model assets were provided.');
    }

    const projectors = files.filter((file) => file.inspection.role === 'projector');
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
    if (projector != null && projector.inspection.role !== 'projector') {
      throw new QueryError('INVALID_MODEL_PAIRING', `"${projector.name}" is not a projector asset.`);
    }

    const modelFiles = [...files.filter((file) => file.assetId !== projector?.assetId)].sort((left, right) =>
      left.name.localeCompare(right.name)
    );
    if (modelFiles.length === 0) {
      throw new QueryError('INVALID_MODEL_PAIRING', 'Projector assets are not runnable models.');
    }
    return {
      modelFiles,
      projector,
    };
  }

  private resolveBaseModel(files: ClassifiedAssetFile[]): BaseModelResolution {
    const modelCandidates = files.filter((file) => file.inspection.role !== 'projector');
    if (modelCandidates.length === 0) {
      throw new QueryError('INVALID_MODEL_PAIRING', 'Projector assets are not runnable models.');
    }

    const visionCandidates = modelCandidates.filter((file) => file.inspection.visionCapable);
    const compatibilitySources = visionCandidates.filter(
      (file) => file.inspection.compatibleVisionProjectorTypes.length > 0
    );

    if (!compatibleVisionTypesAgree(compatibilitySources)) {
      throw new QueryError(
        'INVALID_MODEL_SOURCE',
        'Model assets disagree on compatible vision projector types.'
      );
    }

    const base = visionCandidates[0] ?? modelCandidates[0];
    return {
      compatibleVisionProjectorTypes:
        compatibilitySources[0]?.inspection.compatibleVisionProjectorTypes ?? [],
      name: base.name,
      visionCapable: visionCandidates.length > 0,
    };
  }

  private validateExplicitProjector(
    base: BaseModelResolution,
    projector: ClassifiedAssetFile
  ): void {
    const providedType = projector.inspection.providedVisionProjectorType;
    if (
      providedType != null &&
      base.compatibleVisionProjectorTypes.length > 0 &&
      !base.compatibleVisionProjectorTypes.includes(providedType)
    ) {
      throw new QueryError(
        'INVALID_MODEL_PAIRING',
        `Projector type "${providedType}" is not compatible with this model. Expected one of: ${base.compatibleVisionProjectorTypes.join(', ')}.`
      );
    }
  }
}

function compatibleVisionTypesAgree(files: ClassifiedAssetFile[]): boolean {
  if (files.length < 2) {
    return true;
  }
  const expected = stableTypeList(files[0].inspection.compatibleVisionProjectorTypes);
  return files
    .slice(1)
    .every((file) => expected === stableTypeList(file.inspection.compatibleVisionProjectorTypes));
}

function stableTypeList(projectorTypes: readonly string[]): string {
  return [...projectorTypes].sort((left, right) => left.localeCompare(right)).join('\u0000');
}
