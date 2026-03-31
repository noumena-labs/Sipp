import { CogentEngine, getBundledRuntimeUrls } from '../dist/esm/index.js';

async function run() {
  const engine = new CogentEngine(getBundledRuntimeUrls());
  try {
    await engine.initModule();
    console.log('Runtime ready.');
    console.log('Load a GGUF model with loadModelFromBuffer(), loadModelFromFile(), or loadModelFromUrl(), then call initEngine().');
  } catch (err) {
    console.error('Example failed:', err);
  } finally {
    engine.close();
  }
}

run();
