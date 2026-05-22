import type { ModelDetectionProvider } from '../runtime/model-detection.js';
import type { ClassifiedAssetFile } from './pairing-types.js';

export class ModelAssetClassifier {
  public constructor(private readonly detector: ModelDetectionProvider) {}

  public async classify(
    assetId: string,
    file: File,
    signal?: AbortSignal
  ): Promise<ClassifiedAssetFile> {
    const detection = await this.detector.detectModelFromGgufFile(file, signal);
    return {
      assetId,
      file,
      inspection: detection.inspection,
      name: detection.modelName,
    };
  }
}
