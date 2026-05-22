import type {
  LocalProjectorResolutionResult,
  ModelDetectionResult,
} from '../bundle/model-bundle-types.js';

export interface ModelDetectionProvider {
  detectModelFromGgufFile(
    file: Blob & { name?: string },
    signal?: AbortSignal
  ): Promise<ModelDetectionResult>;
}

export async function resolveLocalModelAndProjectorFiles<T extends Blob & { name: string }>(
  detector: ModelDetectionProvider,
  files: T[],
  signal?: AbortSignal
): Promise<LocalProjectorResolutionResult<T>> {
  const detections = await Promise.all(
    files.map(async (file) => ({
      file,
      detection: await detector.detectModelFromGgufFile(file, signal),
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
