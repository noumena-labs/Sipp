import {
  LocalProjectorResolutionResult,
  ModelDetectionResult,
} from './model-bundle-types.js';
import { inspectGgufMetadata } from './gguf-metadata.js';

const VLM_FILENAME_PATTERNS: RegExp[] = [
  /llava/i,
  /qwen[_-]?2?[_-]?vl/i,
  /lfm[\s._-]*(?:\d+(?:\.\d+)?)?[\s._-]*vl/i,
  /internvl/i,
  /cogvlm/i,
  /phi[_-]?\d*[_-]?vision/i,
  /minicpm[_-]?v/i,
  /moondream/i,
  /obsidian/i,
  /bunny/i,
  /nanollava/i,
  /bakllava/i,
  /mllama/i,
  /llama[_-]?\d*[_-]?vision/i,
  /pixtral/i,
  /smolvlm/i,
  /gemma.*pali/i,
];

const PROJECTOR_FILENAME_PATTERNS: RegExp[] = [
  /mmproj/i,
  /projector/i,
  /clip[_-]?model/i,
  /vision[_-]?encoder/i,
];

const VLM_METADATA_ARCHITECTURES = new Set([
  'cogvlm',
  'llama4',
  'mllama',
  'paddleocr',
  'qwen2vl',
  'qwen3vl',
  'qwen3vlmoe',
]);

function isGgufFileName(fileName: string): boolean {
  return fileName.trim().toLowerCase().endsWith('.gguf');
}

export function findProjectorFileCandidates<T extends { name: string }>(files: T[]): T[] {
  return files.filter(
    (file) =>
      isGgufFileName(file.name) &&
      PROJECTOR_FILENAME_PATTERNS.some((pattern) => pattern.test(file.name))
  );
}

export function splitModelAndProjectorFiles<T extends { name: string }>(
  files: T[]
): LocalProjectorResolutionResult<T> {
  const candidates = findProjectorFileCandidates(files);
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
    .filter(({ detection }) => detection.isProjector)
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

export function detectModelFromFilename(filename: string): ModelDetectionResult {
  const isProjector = PROJECTOR_FILENAME_PATTERNS.some((pattern) => pattern.test(filename));
  const isVision = VLM_FILENAME_PATTERNS.some((pattern) => pattern.test(filename));

  return {
    isVisionModel: isVision && !isProjector,
    isProjector,
    suggestedProjectorUrl: null,
    detectionMethod: isVision || isProjector ? 'filename' : 'none',
    modelName: filename,
    modelType: null,
    modelArchitecture: null,
  };
}

export async function detectModelFromGgufFile(
  file: Blob & { name?: string },
  signal?: AbortSignal
): Promise<ModelDetectionResult> {
  const fileName = normalizeFileName(file.name);
  const defaultValue = detectModelFromFilename(fileName);
  const metadata = await inspectGgufMetadata(file, { signal });

  if (metadata == null) {
    return defaultValue;
  }

  const modelType = normalizeOptionalString(metadata.generalType);
  const modelArchitecture = normalizeOptionalString(metadata.generalArchitecture);
  const clipProjectorType = normalizeOptionalString(metadata.clipProjectorType);
  const clipVisionProjectorType = normalizeOptionalString(metadata.clipVisionProjectorType);
  const clipHasVisionEncoder = metadata.clipHasVisionEncoder === true;

  const isProjector =
    modelType === 'mmproj' ||
    modelArchitecture === 'clip' ||
    clipProjectorType != null ||
    clipVisionProjectorType != null;
  const isVisionByMetadata =
    !isProjector &&
    (clipHasVisionEncoder ||
      (modelArchitecture != null && VLM_METADATA_ARCHITECTURES.has(modelArchitecture)));

  const detectionMethod =
    isProjector || isVisionByMetadata
      ? 'gguf-metadata'
      : modelType != null || modelArchitecture != null
        ? defaultValue.isVisionModel
          ? defaultValue.detectionMethod
          : 'gguf-metadata'
        : defaultValue.detectionMethod;

  return {
    isVisionModel: !isProjector && (isVisionByMetadata || defaultValue.isVisionModel),
    isProjector,
    suggestedProjectorUrl: null,
    detectionMethod,
    modelName: fileName,
    modelType,
    modelArchitecture,
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
