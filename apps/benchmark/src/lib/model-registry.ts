/**
 * Curated model registry for the CogentLM benchmark.
 *
 * Each entry is still app-facing catalog data, but its engine-facing
 * portion is expressed as a `cogent-engine` model bundle descriptor.
 */

import type { ModelBundleDescriptor } from 'cogent-engine';

export type ModelCapability = 'text' | 'vision';

export interface ModelVariant {
  /** Quantization label (e.g. "Q4_0", "Q4_K_M") */
  quant: string;
  /** Approximate file size in bytes (for display) */
  sizeBytes: number;
  /** Approximate projector file size in bytes */
  projectorSizeBytes?: number;
  /** Bundle descriptor consumed by the engine. */
  bundle: ModelBundleDescriptor;
}

export interface ModelRegistryEntry {
  /** Unique identifier */
  id: string;
  /** Human-readable name */
  name: string;
  /** Model family / publisher */
  publisher: string;
  /** Parameter count label (e.g. "0.5B", "2B", "7B") */
  parameterCount: string;
  /** What this model can do */
  capability: ModelCapability;
  /** Available quantization variants */
  variants: ModelVariant[];
  /** Default variant index */
  defaultVariant?: number;
}

// ── Registry ─────────────────────────────────────────────────────────────

export const MODEL_REGISTRY: ModelRegistryEntry[] = [
  // ── Text models ──
  {
    id: 'qwen2.5-0.5b-instruct',
    name: 'Qwen 2.5 0.5B Instruct',
    publisher: 'Qwen',
    parameterCount: '0.5B',
    capability: 'text',
    variants: [
      {
        quant: 'Q4_0',
        sizeBytes: 397_000_000,
        bundle: {
          kind: 'url',
          url: 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf',
        },
      },
    ],
  },
  {
    id: 'qwen2.5-1.5b-instruct',
    name: 'Qwen 2.5 1.5B Instruct',
    publisher: 'Qwen',
    parameterCount: '1.5B',
    capability: 'text',
    variants: [
      {
        quant: 'Q4_K_M',
        sizeBytes: 1_050_000_000,
        bundle: {
          kind: 'url',
          url: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf',
        },
      },
    ],
  },
  {
    id: 'smollm2-360m-instruct',
    name: 'SmolLM2 360M Instruct',
    publisher: 'HuggingFace',
    parameterCount: '360M',
    capability: 'text',
    variants: [
      {
        quant: 'Q8_0',
        sizeBytes: 386_000_000,
        bundle: {
          kind: 'url',
          url: 'https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf',
        },
      },
    ],
  },

  // ── Vision models ──
  {
    id: 'qwen2-vl-2b-instruct',
    name: 'Qwen2-VL 2B Instruct',
    publisher: 'Qwen',
    parameterCount: '2B',
    capability: 'vision',
    variants: [
      {
        quant: 'Q4_K_M',
        sizeBytes: 1_400_000_000,
        projectorSizeBytes: 1_500_000_000,
        bundle: {
          kind: 'url',
          url: 'https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf',
          projector: {
            kind: 'url',
            url: 'https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-f16.gguf',
          },
        },
      },
    ],
  },
  {
    id: 'llava-v1.5-7b',
    name: 'LLaVA v1.5 7B',
    publisher: 'LLaVA',
    parameterCount: '7B',
    capability: 'vision',
    variants: [
      {
        quant: 'Q4_K_M',
        sizeBytes: 4_080_000_000,
        projectorSizeBytes: 624_000_000,
        bundle: {
          kind: 'url',
          url: 'https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/ggml-model-q4_k.gguf',
          projector: {
            kind: 'url',
            url: 'https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/mmproj-model-f16.gguf',
          },
        },
      },
    ],
  },
  {
    id: 'smolvlm-256m-instruct',
    name: 'SmolVLM 256M Instruct',
    publisher: 'HuggingFace',
    parameterCount: '256M',
    capability: 'vision',
    variants: [
      {
        quant: 'Q8_0',
        sizeBytes: 286_000_000,
        projectorSizeBytes: 360_000_000,
        bundle: {
          kind: 'url',
          url: 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf',
          projector: {
            kind: 'url',
            url: 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-f16.gguf',
          },
        },
      },
    ],
  },
  {
    id: 'smolvlm-500m-instruct',
    name: 'SmolVLM 500M Instruct',
    publisher: 'HuggingFace',
    parameterCount: '500M',
    capability: 'vision',
    variants: [
      {
        quant: 'Q8_0',
        sizeBytes: 534_000_000,
        projectorSizeBytes: 360_000_000,
        bundle: {
          kind: 'url',
          url: 'https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf',
          projector: {
            kind: 'url',
            url: 'https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-f16.gguf',
          },
        },
      },
    ],
  },
];

// ── Helpers ──────────────────────────────────────────────────────────────

export function getModelById(id: string): ModelRegistryEntry | undefined {
  return MODEL_REGISTRY.find((m) => m.id === id);
}

export function getDefaultVariant(model: ModelRegistryEntry): ModelVariant {
  return model.variants[model.defaultVariant ?? 0];
}

export function getVariantPrimaryUrl(variant: ModelVariant): string {
  switch (variant.bundle.kind) {
    case 'url':
      return variant.bundle.url;
    case 'urls':
      return variant.bundle.urls[0] ?? 'model.gguf';
    case 'file':
      return variant.bundle.file.name || 'model.gguf';
    case 'files':
      return variant.bundle.files[0]?.name || 'model.gguf';
  }
}

export function isVisionModel(model: ModelRegistryEntry): boolean {
  return model.capability === 'vision';
}

export function getVisionModels(): ModelRegistryEntry[] {
  return MODEL_REGISTRY.filter((m) => m.capability === 'vision');
}

export function getTextModels(): ModelRegistryEntry[] {
  return MODEL_REGISTRY.filter((m) => m.capability === 'text');
}

export function formatSize(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(0)} MB`;
  return `${(bytes / 1_000).toFixed(0)} KB`;
}
