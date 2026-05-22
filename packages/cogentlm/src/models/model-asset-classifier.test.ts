import assert from 'node:assert/strict';
import test from 'node:test';

import type { ModelDetectionResult } from '../types.js';
import { ModelAssetClassifier } from './model-asset-classifier.js';

const visionDetection: ModelDetectionResult = {
  inspection: {
    version: 1,
    role: 'model',
    architecture: 'lfm2',
    visionCapable: true,
    compatibleVisionProjectorTypes: ['lfm2'],
    providedVisionProjectorType: null,
  },
  detectionMethod: 'gguf-metadata',
  modelName: 'base.gguf',
  modelType: null,
  modelArchitecture: 'lfm2',
};

test('ModelAssetClassifier delegates GGUF detection to provider', async () => {
  const file = new File(['fake'], 'base.gguf');
  const seen: Array<{ file: File; signal?: AbortSignal }> = [];
  const signal = new AbortController().signal;
  const classifier = new ModelAssetClassifier({
    async detectModelFromGgufFile(input, inputSignal) {
      seen.push({ file: input as File, signal: inputSignal });
      return visionDetection;
    },
  });

  const classified = await classifier.classify('asset-model', file, signal);

  assert.deepEqual(seen, [{ file, signal }]);
  assert.equal(classified.assetId, 'asset-model');
  assert.equal(classified.file, file);
  assert.equal(classified.name, 'base.gguf');
  assert.equal(classified.inspection.role, 'model');
  assert.equal(classified.inspection.visionCapable, true);
  assert.deepEqual(classified.inspection.compatibleVisionProjectorTypes, ['lfm2']);
});
