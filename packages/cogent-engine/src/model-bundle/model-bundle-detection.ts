import {
  LocalProjectorResolutionResult,
  ModelDetectionResult,
  ParsedHuggingFaceModelUrl,
  ProjectorDiscoveryResult,
} from './model-bundle-types.js';
import { inspectGgufMetadata } from './gguf-metadata.js';

const VLM_FILENAME_PATTERNS: RegExp[] = [
  /llava/i,
  /qwen[_-]?2?[_-]?vl/i,
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
  const fallback = detectModelFromFilename(fileName);
  const metadata = await inspectGgufMetadata(file, { signal });

  if (metadata == null) {
    return fallback;
  }

  const modelType = normalizeOptionalString(metadata.generalType);
  const modelArchitecture = normalizeOptionalString(metadata.generalArchitecture);
  const clipProjectorType = normalizeOptionalString(metadata.clipProjectorType);
  const clipVisionProjectorType = normalizeOptionalString(metadata.clipVisionProjectorType);

  const isProjector =
    modelType === 'mmproj' ||
    modelArchitecture === 'clip' ||
    clipProjectorType != null ||
    clipVisionProjectorType != null;
  const isVisionByMetadata =
    !isProjector &&
    modelArchitecture != null &&
    VLM_METADATA_ARCHITECTURES.has(modelArchitecture);

  const detectionMethod =
    isProjector || isVisionByMetadata
      ? 'gguf-metadata'
      : modelType != null || modelArchitecture != null
        ? fallback.isVisionModel
          ? fallback.detectionMethod
          : 'gguf-metadata'
        : fallback.detectionMethod;

  return {
    isVisionModel: !isProjector && (isVisionByMetadata || fallback.isVisionModel),
    isProjector,
    suggestedProjectorUrl: null,
    detectionMethod,
    modelName: fileName,
    modelType,
    modelArchitecture,
  };
}

export function detectModelFromUrl(url: string): ModelDetectionResult {
  const filename = extractFilenameFromUrl(url);
  const result = detectModelFromFilename(filename);
  result.detectionMethod = result.isVisionModel ? 'url' : result.detectionMethod;

  if (result.isVisionModel) {
    result.suggestedProjectorUrl = deriveProjectorUrlFallback(url);
  }

  return result;
}

export function detectModel(
  source: 'url' | 'file',
  urlOrFilename: string
): ModelDetectionResult {
  return source === 'url'
    ? detectModelFromUrl(urlOrFilename)
    : detectModelFromFilename(urlOrFilename);
}

export function parseHuggingFaceUrl(url: string): ParsedHuggingFaceModelUrl | null {
  const match = url.match(
    /^(https:\/\/huggingface\.co\/([^/]+)\/([^/]+))\/(resolve|blob)\/([^/]+)\/(.+)$/
  );
  if (!match) {
    return null;
  }

  return {
    org: match[2],
    repo: match[3],
    ref: match[5],
    filename: match[6],
    baseUrl: match[1],
  };
}

export async function discoverProjectorFromHuggingFace(
  modelUrl: string
): Promise<ProjectorDiscoveryResult> {
  const hf = parseHuggingFaceUrl(modelUrl);
  if (!hf) {
    return {
      projectorUrl: null,
      candidates: [],
      source: 'none',
      message: 'Not a HuggingFace URL — cannot auto-discover projector.',
    };
  }

  try {
    const apiUrl = `https://huggingface.co/api/models/${hf.org}/${hf.repo}`;
    const response = await fetch(apiUrl);
    if (!response.ok) {
      return {
        projectorUrl: null,
        candidates: [],
        source: 'none',
        message: `HuggingFace API returned ${response.status} for ${hf.org}/${hf.repo}.`,
      };
    }

    const data = await response.json();
    const siblings = Array.isArray(data?.siblings)
      ? (data.siblings as Array<{ rfilename?: unknown }>)
      : [];
    const projectorFiles = siblings
      .map((sibling) => {
        if (typeof sibling?.rfilename === 'string') {
          return sibling.rfilename;
        }
        return '';
      })
      .filter(
        (name: string): name is string =>
          name.length > 0 &&
          isGgufFileName(name) &&
          PROJECTOR_FILENAME_PATTERNS.some((pattern) => pattern.test(name))
      );

    if (projectorFiles.length === 0) {
      return {
        projectorUrl: null,
        candidates: [],
        source: 'hf-api',
        message: `No mmproj/projector GGUF files found in ${hf.org}/${hf.repo}. This model may not have a published projector.`,
      };
    }

    const preferred =
      projectorFiles.find((name: string) => /f16/i.test(name)) ?? projectorFiles[0];
    return {
      projectorUrl: `${hf.baseUrl}/resolve/${hf.ref}/${preferred}`,
      candidates: projectorFiles,
      source: 'hf-api',
      message:
        projectorFiles.length === 1
          ? `Found projector: ${preferred}`
          : `Found ${projectorFiles.length} projector(s): ${projectorFiles.join(', ')}. Using: ${preferred}`,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      projectorUrl: null,
      candidates: [],
      source: 'none',
      message: `HuggingFace API error: ${message}`,
    };
  }
}

export async function discoverProjector(modelUrl: string): Promise<ProjectorDiscoveryResult> {
  const hfResult = await discoverProjectorFromHuggingFace(modelUrl);
  if (hfResult.projectorUrl) {
    return hfResult;
  }

  const hf = parseHuggingFaceUrl(modelUrl);
  if (hf) {
    const candidates = [
      'mmproj-model-f16.gguf',
      `${hf.repo}-mmproj-f16.gguf`,
      'mmproj.gguf',
    ];

    for (const candidate of candidates) {
      const candidateUrl = `${hf.baseUrl}/resolve/${hf.ref}/${candidate}`;
      try {
        const response = await fetch(candidateUrl, { method: 'HEAD' });
        if (response.ok) {
          return {
            projectorUrl: candidateUrl,
            candidates: [candidate],
            source: 'head-probe',
            message: `Found projector via HEAD probe: ${candidate}`,
          };
        }
      } catch {
        // Continue probing.
      }
    }
  }

  return {
    projectorUrl: null,
    candidates: [],
    source: 'none',
    message: hfResult.message || 'Could not discover a projector file. Please provide one manually.',
  };
}

export async function validateProjectorUrl(url: string): Promise<string | null> {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    return response.ok ? url : null;
  } catch {
    return null;
  }
}

function deriveProjectorUrlFallback(modelUrl: string): string | null {
  const hf = parseHuggingFaceUrl(modelUrl);
  if (!hf) {
    return null;
  }
  return `${hf.baseUrl}/resolve/${hf.ref}/mmproj-model-f16.gguf`;
}

function extractFilenameFromUrl(url: string): string {
  try {
    const pathname = new URL(url).pathname;
    return pathname.split('/').pop() ?? url;
  } catch {
    return url.split('/').pop()?.split('?')[0] ?? url;
  }
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
