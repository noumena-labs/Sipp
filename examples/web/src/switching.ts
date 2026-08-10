import {
  Endpoint,
  SippClient,
  type EndpointRef,
  type ManagedModel,
  type ModelInfo,
} from '@noumena-labs/sipp';
import { reportError, write } from './common.js';

const ENDPOINT_ID = 'switching-local';
const app = document.querySelector<HTMLDivElement>('#app');
if (app == null) {
  throw new Error('missing #app element');
}

app.innerHTML = `
  <section class="shell">
    <header class="page-header">
      <nav class="top-nav"><a href="/">All examples</a></nav>
      <h1>Model Switching</h1>
      <p>
        Run six operations through one client and one endpoint ID. Each step
        replaces and destroys the previous native runtime before loading the next.
      </p>
    </header>
    <form id="switching-form" class="panel">
      <label>LLM GGUF<input id="llm-files" type="file" accept=".gguf" multiple /></label>
      <label>Embedding GGUF<input id="embedding-files" type="file" accept=".gguf" multiple /></label>
      <label>VLM model and projector GGUFs<input id="vlm-files" type="file" accept=".gguf" multiple /></label>
      <label>VLM image<input id="image-file" type="file" accept="image/*" /></label>
      <label>TTS model and projector GGUFs<input id="tts-files" type="file" accept=".gguf" multiple /></label>
      <label>STT model and projector GGUFs<input id="stt-files" type="file" accept=".gguf" multiple /></label>
      <label>Audio to transcribe<input id="audio-file" type="file" accept="audio/wav,audio/mpeg,audio/flac,.wav,.mp3,.flac" /></label>
      <label>Optional speaker reference<input id="speaker-file" type="file" accept="audio/wav,audio/mpeg,audio/flac,.wav,.mp3,.flac" /></label>
      <label>LLM prompt<textarea id="llm-prompt" rows="2">Hello from Sipp.</textarea></label>
      <label>Vision prompt<textarea id="vision-prompt" rows="2">Describe this image.</textarea></label>
      <label>TTS text<textarea id="tts-text" rows="2">Hello from Sipp.</textarea></label>
      <div class="field-row">
        <label>Language<input id="language" value="en" autocomplete="off" /></label>
        <label>Max output tokens<input id="max-tokens" type="number" min="1" step="1" value="128" /></label>
      </div>
      <button type="submit">Run switching sequence</button>
    </form>
    <audio id="player" controls hidden></audio>
    <pre id="output">Choose every required file, then run the sequence.</pre>
  </section>
`;

type RuntimeOperation = 'query' | 'chat' | 'embed' | 'listen' | 'speak';

const client = new SippClient();
const output = element<HTMLPreElement>('output');
const player = element<HTMLAudioElement>('player');
let audioUrl: string | null = null;

element<HTMLFormElement>('switching-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  output.dataset.status = 'running';
  try {
    const llmFiles = requiredFiles('llm-files', 'LLM GGUF');
    const embeddingFiles = requiredFiles('embedding-files', 'embedding GGUF');
    const vlmFiles = requiredFiles('vlm-files', 'VLM model and projector GGUFs', 2);
    const ttsFiles = requiredFiles('tts-files', 'TTS model and projector GGUFs', 2);
    const sttFiles = requiredFiles('stt-files', 'STT model and projector GGUFs', 2);
    const image = requiredFile('image-file', 'VLM image');
    const audio = requiredFile('audio-file', 'audio to transcribe');
    const speaker = optionalFile('speaker-file');
    const prompt = requiredText('llm-prompt', 'LLM prompt');
    const visionPrompt = requiredText('vision-prompt', 'vision prompt');
    const speechText = requiredText('tts-text', 'TTS text');
    const language = requiredText('language', 'language');
    const maxTokens = positiveInteger('max-tokens');
    const results: string[] = [];

    write(output, 'Registering LLM assets...');
    const llmModel = await client.models.add(llmFiles);
    write(output, 'Registering embedding assets...');
    const embeddingModel = await client.models.add(embeddingFiles);
    write(output, 'Registering VLM assets...');
    const vlmModel = await client.models.add(vlmFiles);
    write(output, 'Registering TTS assets...');
    const ttsModel = await client.models.add(ttsFiles);
    write(output, 'Registering STT assets...');
    const sttModel = await client.models.add(sttFiles);

    const firstLlm = await activate(llmModel, 'query');
    const firstText = await client.query(prompt, {
      endpoint: firstLlm.endpoint,
      maxTokens,
    }).response;
    results.push(stepResult('LLM query', firstLlm.model, firstText.text));

    const embedding = await activate(embeddingModel, 'embed');
    const vector = await client.embed(prompt, { endpoint: embedding.endpoint }).response;
    results.push(stepResult('Embedding', embedding.model, `dimensions=${vector.values.length}`));

    const vlm = await activate(vlmModel, 'chat');
    const vision = await client.chat(
      {
        messages: [{ role: 'user', content: visionPrompt }],
        media: [new Uint8Array(await image.arrayBuffer())],
      },
      { endpoint: vlm.endpoint, maxTokens }
    ).response;
    results.push(stepResult('VLM chat', vlm.model, vision.text));

    const tts = await activate(ttsModel, 'speak');
    const synthesized = await client.speak(speechText, {
      endpoint: tts.endpoint,
      language,
      speakerAudio:
        speaker == null ? undefined : new Uint8Array(await speaker.arrayBuffer()),
    }).response;
    showAudio(synthesized.audio);
    results.push(stepResult('TTS speak', tts.model, `audio_ms=${synthesized.durationMs}`));

    const stt = await activate(sttModel, 'listen');
    const transcript = await client.listen(new Uint8Array(await audio.arrayBuffer()), {
      endpoint: stt.endpoint,
      language,
      maxTokens,
    }).response;
    results.push(stepResult('STT listen', stt.model, transcript.text));

    const finalLlm = await activate(llmModel, 'query');
    const finalText = await client.query(prompt, {
      endpoint: finalLlm.endpoint,
      maxTokens,
    }).response;
    results.push(stepResult('LLM query again', finalLlm.model, finalText.text));

    write(output, `${results.join('\n\n')}\n\nsequence complete`);
    output.dataset.status = 'complete';
  } catch (error) {
    reportError(output, error);
    output.dataset.status = 'error';
  }
});

async function activate(
  managed: ManagedModel,
  operation: RuntimeOperation
): Promise<{ endpoint: EndpointRef; model: ModelInfo }> {
  write(output, `Loading ${operation} runtime...`);
  const endpoint = await client.add(
    ENDPOINT_ID,
    Endpoint.local(managed, {
      observability: 'runtime',
    })
  );
  const model = client.observability.current().model;
  if (model == null || model.id !== managed.id) {
    throw new Error(`Native runtime did not publish ${managed.name}.`);
  }
  if (model.capabilities?.operations[operation] !== true) {
    throw new Error(`Native runtime reports that ${managed.name} does not support ${operation}.`);
  }
  return { endpoint, model };
}

function stepResult(label: string, model: ModelInfo, detail: string): string {
  return `${label}: model=${model.name}\n${detail.trim()}`;
}

function showAudio(audio: Uint8Array): void {
  if (audioUrl != null) {
    URL.revokeObjectURL(audioUrl);
  }
  audioUrl = URL.createObjectURL(new Blob([Uint8Array.from(audio)], { type: 'audio/wav' }));
  player.src = audioUrl;
  player.hidden = false;
}

function requiredFiles(id: string, label: string, minimum = 1): File[] {
  const files = Array.from(element<HTMLInputElement>(id).files ?? []);
  if (files.length < minimum) {
    throw new Error(`${label} requires at least ${minimum} file${minimum === 1 ? '' : 's'}.`);
  }
  return files;
}

function requiredFile(id: string, label: string): File {
  const file = optionalFile(id);
  if (file == null) {
    throw new Error(`${label} is required.`);
  }
  return file;
}

function optionalFile(id: string): File | null {
  return element<HTMLInputElement>(id).files?.[0] ?? null;
}

function requiredText(id: string, label: string): string {
  const value = element<HTMLInputElement | HTMLTextAreaElement>(id).value.trim();
  if (value.length === 0) {
    throw new Error(`${label} is required.`);
  }
  return value;
}

function positiveInteger(id: string): number {
  const value = Number(element<HTMLInputElement>(id).value);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${id} must be a positive integer.`);
  }
  return value;
}

function element<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (node == null) {
    throw new Error(`missing #${id}`);
  }
  return node as T;
}
