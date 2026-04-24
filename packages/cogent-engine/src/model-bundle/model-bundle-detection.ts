import {
  LocalProjectorResolutionResult,
  type AssetInspection,
  type ModelDetectionResult,
} from './model-bundle-types.js';
import { inspectGgufMetadata } from './gguf-metadata.js';

interface VisionArchitectureRule {
  projectorTypes: readonly string[];
  requiresVisionEncoderFlag?: boolean;
}

const VISION_ARCHITECTURE_RULES = new Map<string, VisionArchitectureRule>([
  ['cogvlm', { projectorTypes: ['cogvlm'] }],
  ['gemma3', { projectorTypes: ['gemma3'], requiresVisionEncoderFlag: true }],
  ['gemma3n', { projectorTypes: ['gemma3nv'], requiresVisionEncoderFlag: true }],
  ['gemma4', { projectorTypes: ['gemma4v'], requiresVisionEncoderFlag: true }],
  ['hunyuan_vl', { projectorTypes: ['hunyuanvl'] }],
  ['lfm2', { projectorTypes: ['lfm2'], requiresVisionEncoderFlag: true }],
  ['llama4', { projectorTypes: ['llama4'], requiresVisionEncoderFlag: true }],
  ['paddleocr', { projectorTypes: ['paddleocr'] }],
  ['qwen2vl', { projectorTypes: ['qwen2vl_merger', 'qwen2.5vl_merger'] }],
  ['qwen3vl', { projectorTypes: ['qwen3vl_merger'] }],
  ['qwen3vlmoe', { projectorTypes: ['qwen3vl_merger'] }],
]);

const ASSET_INSPECTION_VERSION = 1;

export async function resolveLocalModelAndProjectorFiles<T extends Blob & { name: string }>(
  files: T[],
  signal?: AbortSignal
): Promise<LocalProjectorResolutionResult<T>> {
  const detections = await Promise.all(
    files.map(async (file) => ({
      file,
      detection: await detectModelFromGgufFile(file, signal),
    }))
  );

  const candidates = detections
    .filter(({ detection }) => detection.inspection.role === 'projector')
    .map(({ file }) => file);

  if (candidates.length > 1) {
    return {
      modelFiles: [...files],
      projectorFile: null,
      candidateFileNames: candidates.map((candidate) => candidate.name),
      errorMessage: `Multiple projector candidates found: ${candidates
        .map((candidate) => candidate.name)
        .join(', ')}.`,
    };
  }

  if (candidates.length === 0) {
    return {
      modelFiles: [...files],
      projectorFile: null,
      candidateFileNames: [],
      errorMessage: null,
    };
  }

  const projectorFile = candidates[0];
  const modelFiles = files.filter((file) => file !== projectorFile);
  return {
    modelFiles,
    projectorFile,
    candidateFileNames: [projectorFile.name],
    errorMessage:
      modelFiles.length > 0
        ? null
        : `No model files remain after removing projector candidate "${projectorFile.name}".`,
  };
}

export async function detectModelFromGgufFile(
  file: Blob & { name?: string },
  signal?: AbortSignal
): Promise<ModelDetectionResult> {
  const fileName = normalizeFileName(file.name);
  const metadata = await inspectGgufMetadata(file, { signal });

  if (metadata == null) {
    return {
      inspection: emptyInspection(),
      detectionMethod: 'none',
      modelName: fileName,
      modelType: null,
      modelArchitecture: null,
    };
  }

  const modelType = normalizeOptionalString(metadata.generalType);
  const modelArchitecture = normalizeOptionalString(metadata.generalArchitecture);
  const clipProjectorType = normalizeOptionalString(metadata.clipProjectorType);
  const clipVisionProjectorType = normalizeOptionalString(metadata.clipVisionProjectorType);
  const clipHasVisionEncoder = metadata.clipHasVisionEncoder === true;
  const providedVisionProjectorType = clipVisionProjectorType ?? clipProjectorType;

  const inspection = buildInspection(
    modelType,
    modelArchitecture,
    clipHasVisionEncoder,
    providedVisionProjectorType
  );

  return {
    inspection,
    detectionMethod: inspection.role === 'unknown' ? 'none' : 'gguf-metadata',
    modelName: fileName,
    modelType,
    modelArchitecture,
  };
}

export function inspectionFromDetection(detection: ModelDetectionResult): AssetInspection {
  return detection.inspection;
}

function buildInspection(
  modelType: string | null,
  architecture: string | null,
  clipHasVisionEncoder: boolean,
  providedVisionProjectorType: string | null
): AssetInspection {
  const isProjector =
    modelType === 'mmproj' || architecture === 'clip' || providedVisionProjectorType != null;

  const compatibleVisionProjectorTypes = isProjector
    ? []
    : resolveCompatibleVisionProjectorTypes(architecture, clipHasVisionEncoder);
  const visionCapable = !isProjector && (clipHasVisionEncoder || compatibleVisionProjectorTypes.length > 0);

  return {
    version: ASSET_INSPECTION_VERSION,
    role:
      isProjector
        ? 'projector'
        : modelType != null || architecture != null || clipHasVisionEncoder
          ? 'model'
          : 'unknown',
    architecture,
    visionCapable,
    compatibleVisionProjectorTypes,
    providedVisionProjectorType,
  };
}

function resolveCompatibleVisionProjectorTypes(
  architecture: string | null,
  clipHasVisionEncoder: boolean
): string[] {
  if (architecture == null) {
    return [];
  }
  const rule = VISION_ARCHITECTURE_RULES.get(architecture);
  if (rule == null) {
    return [];
  }
  if (rule.requiresVisionEncoderFlag && !clipHasVisionEncoder) {
    return [];
  }
  return [...rule.projectorTypes];
}

function emptyInspection(): AssetInspection {
  return {
    version: ASSET_INSPECTION_VERSION,
    role: 'unknown',
    architecture: null,
    visionCapable: false,
    compatibleVisionProjectorTypes: [],
    providedVisionProjectorType: null,
  };
}

function normalizeFileName(fileName: string | undefined): string {
  return typeof fileName === 'string' && fileName.trim().length > 0
    ? fileName
    : 'model.gguf';
}

function normalizeOptionalString(value: string | null | undefined): string | null {
  if (typeof value !== 'string') {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  return normalized.length > 0 ? normalized : null;
}
