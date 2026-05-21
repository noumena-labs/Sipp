import { detectModelFromGgufFile } from '../bundle/model-bundle-detection.js';
import type { ClassifiedAssetFile } from './pairing-types.js';

export class ModelAssetClassifier {
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
}
