import type { ModelSource } from '@noumena-labs/cogentlm';

export type ModelCapability = 'text' | 'vision';

export interface CuratedModel {
  readonly id: string;
  readonly name: string;
  readonly publisher: string;
  readonly detail: string;
  readonly sizeLabel: string;
  readonly capability: ModelCapability;
  readonly source: ModelSource;
  readonly recommended?: boolean;
}

export type ModelSelection =
  | {
      readonly kind: 'curated';
      readonly modelId: string;
    }
  | {
      readonly kind: 'custom-url';
      readonly url: string;
    }
  | {
      readonly kind: 'custom-file';
      readonly file: File;
    };

export interface ResolvedModelSelection {
  readonly id: string;
  readonly name: string;
  readonly capability: ModelCapability;
  readonly source: ModelSource;
  readonly custom: boolean;
}

export const CURATED_MODELS: readonly CuratedModel[] = [
  {
    id: 'qwen2.5-0.5b-instruct',
    name: 'Qwen 2.5 0.5B Instruct',
    publisher: 'Qwen',
    detail: 'Q4_0',
    sizeLabel: '429 MB',
    capability: 'text',
    recommended: true,
    source:
      'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf',
  },
  {
    id: 'smollm2-360m-instruct',
    name: 'SmolLM2 360M Instruct',
    publisher: 'Hugging Face',
    detail: 'Q8_0',
    sizeLabel: '386 MB',
    capability: 'text',
    source:
      'https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf',
  },
  {
    id: 'lfm2.5-vl-450m',
    name: 'LFM2.5 VL 450M',
    publisher: 'Liquid AI',
    detail: 'F16 model and projector',
    sizeLabel: '901 MB',
    capability: 'vision',
    source: {
      model:
        'https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/LFM2.5-VL-450M-F16.gguf',
      projector:
        'https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/mmproj-LFM2.5-VL-450m-F16.gguf',
    },
  },
];

export function getCuratedModel(modelId: string): CuratedModel {
  const model = CURATED_MODELS.find((candidate) => candidate.id === modelId);
  if (model == null) {
    throw new Error(`Unknown curated model "${modelId}".`);
  }
  return model;
}

export function resolveModelSelection(
  selection: ModelSelection
): ResolvedModelSelection {
  if (selection.kind === 'curated') {
    const model = getCuratedModel(selection.modelId);
    return {
      id: model.id,
      name: model.name,
      capability: model.capability,
      source: model.source,
      custom: false,
    };
  }

  if (selection.kind === 'custom-file') {
    if (selection.file.size <= 0) {
      throw new Error('Choose a non-empty GGUF model file.');
    }
    return {
      id: `custom-file:${selection.file.name}`,
      name: selection.file.name || 'Local GGUF model',
      capability: 'text',
      source: selection.file,
      custom: true,
    };
  }

  const url = normalizeModelUrl(selection.url);
  return {
    id: `custom-url:${url}`,
    name: fileNameFromUrl(url),
    capability: 'text',
    source: url,
    custom: true,
  };
}

export function projectorRequirementMessage(
  selection: ResolvedModelSelection
): string {
  return selection.custom
    ? 'This model needs a vision projector. Custom vision imports are not supported yet; choose a curated vision model.'
    : 'The curated vision bundle is missing its required projector.';
}

function normalizeModelUrl(rawUrl: string): string {
  const trimmed = rawUrl.trim();
  if (trimmed.length === 0) {
    throw new Error('Enter a GGUF model URL.');
  }

  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    throw new Error('Enter a valid HTTP or HTTPS model URL.');
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error('Model URLs must use HTTP or HTTPS.');
  }
  return parsed.toString();
}

function fileNameFromUrl(url: string): string {
  const parsed = new URL(url);
  const fileName = decodeURIComponent(parsed.pathname.split('/').pop() ?? '').trim();
  return fileName.length > 0 ? fileName : 'Custom GGUF model';
}
