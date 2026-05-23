import type { ModelDetectionResult } from '../bundle/model-bundle-types.js';

export interface ModelDetectionProvider {
  detectModelFromGgufFile(
    file: Blob & { name?: string },
    signal?: AbortSignal
  ): Promise<ModelDetectionResult>;
}
