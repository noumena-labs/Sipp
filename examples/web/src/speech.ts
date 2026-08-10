import { Endpoint, SippClient, type EndpointRef } from '@noumena-labs/sipp';
import { EXAMPLE_LOCAL_ENDPOINT_ID, reportError, write } from './common.js';
import './style.css';

const app = document.querySelector<HTMLDivElement>('#app');
if (app == null) {
  throw new Error('missing #app element');
}

app.innerHTML = `
  <section class="shell">
    <header class="page-header">
      <nav class="top-nav"><a href="/">All examples</a></nav>
      <h1>Local Speech</h1>
      <p>
        Load an ASR or TTS GGUF together with its projector, then run the
        matching operation.
      </p>
    </header>
    <form id="model-form" class="panel">
      <div class="field-row">
        <label>Model URL or path<input id="model" autocomplete="off" /></label>
        <label>Model file<input id="model-file" type="file" /></label>
      </div>
      <div class="field-row">
        <label>Projector URL or path<input id="projector" autocomplete="off" /></label>
        <label>Projector file<input id="projector-file" type="file" /></label>
      </div>
      <button type="submit">Load speech model</button>
    </form>
    <form id="run-form" class="panel">
      <div class="field-row">
        <label>
          Operation
          <select id="operation">
            <option value="listen">Listen</option>
            <option value="speak">Speak</option>
          </select>
        </label>
        <label>
          Language
          <input id="language" value="en" autocomplete="off" />
        </label>
        <label>
          Listen max tokens
          <input id="max-tokens" type="number" min="1" step="1" value="512" />
        </label>
        <label>
          Speak max duration (ms)
          <input
            id="max-duration-ms"
            type="number"
            min="1"
            step="1"
            placeholder="adapter default"
          />
        </label>
      </div>
      <label>
        Text to synthesize
        <textarea id="prompt" rows="3">Hello from Sipp.</textarea>
      </label>
      <div class="field-row">
        <label>
          Audio to transcribe
          <input
            id="audio"
            type="file"
            accept="audio/wav,audio/mpeg,audio/flac,.wav,.mp3,.flac"
          />
        </label>
        <label>
          Optional speaker reference
          <input
            id="speaker"
            type="file"
            accept="audio/wav,audio/mpeg,audio/flac,.wav,.mp3,.flac"
          />
        </label>
      </div>
      <button type="submit">Run speech operation</button>
    </form>
    <audio id="player" controls hidden></audio>
    <pre id="output">No model loaded.</pre>
  </section>
`;

const client = new SippClient();
const output = element<HTMLPreElement>('output');
const player = element<HTMLAudioElement>('player');
let endpoint: EndpointRef | null = null;
let outputUrl: string | null = null;

element<HTMLFormElement>('model-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    const model = readSource('model', 'model-file');
    const projector = readSource('projector', 'projector-file');
    if (model == null || projector == null) {
      write(output, 'Choose both a model GGUF and its projector GGUF.');
      return;
    }
    write(output, 'Loading speech model...');
    const managed = await client.models.add([model, projector]);
    endpoint = await client.add(
      EXAMPLE_LOCAL_ENDPOINT_ID,
      Endpoint.local(managed, { observability: 'runtime' })
    );
    write(output, `Loaded ${managed.name}.`);
  } catch (error) {
    reportError(output, error);
  }
});

element<HTMLFormElement>('run-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (endpoint == null) {
    write(output, 'Load a speech model before running the operation.');
    return;
  }
  const languageValue = element<HTMLInputElement>('language').value;
  const language = languageValue.length === 0 ? undefined : languageValue;
  try {
    if (element<HTMLSelectElement>('operation').value === 'listen') {
      const audio = element<HTMLInputElement>('audio').files?.[0];
      if (audio == null) {
        write(output, 'Choose a WAV, MP3, or FLAC file to transcribe.');
        return;
      }
      const maxTokens = Number(element<HTMLInputElement>('max-tokens').value);
      const encodedAudio = new Uint8Array(await audio.arrayBuffer());
      const startedAt = performance.now();
      const result = await client.listen(encodedAudio, {
        endpoint,
        language,
        maxTokens,
      }).response;
      const totalMs = performance.now() - startedAt;
      write(output, `${timingSummary(totalMs)}\ntranscript=${result.text.trim()}`);
      return;
    }

    const text = element<HTMLTextAreaElement>('prompt').value;
    if (text.length === 0) {
      write(output, 'Enter text to synthesize.');
      return;
    }
    const speakerFile = element<HTMLInputElement>('speaker').files?.[0];
    const speakerAudio =
      speakerFile == null ? undefined : new Uint8Array(await speakerFile.arrayBuffer());
    const durationValue = element<HTMLInputElement>('max-duration-ms').value;
    const maxDurationMs = durationValue.length === 0 ? undefined : Number(durationValue);
    const startedAt = performance.now();
    const result = await client.speak(text, {
      endpoint,
      language,
      speakerAudio,
      maxDurationMs,
    }).response;
    if (outputUrl != null) {
      URL.revokeObjectURL(outputUrl);
    }
    outputUrl = URL.createObjectURL(
      new Blob([Uint8Array.from(result.audio)], { type: 'audio/wav' })
    );
    player.src = outputUrl;
    player.hidden = false;
    const totalMs = performance.now() - startedAt;
    const summary = [
      timingSummary(totalMs),
      `audio_ms=${result.durationMs}`,
      `real_time_factor=${(totalMs / result.durationMs).toFixed(2)}`,
      `sample_rate_hz=${result.sampleRateHz}`,
    ].join('\n');
    write(output, summary);
  } catch (error) {
    reportError(output, error);
  }
});

function readSource(urlId: string, fileId: string): File | string | null {
  const file = element<HTMLInputElement>(fileId).files?.[0];
  if (file != null) {
    return file;
  }
  const url = element<HTMLInputElement>(urlId).value.trim();
  return url.length === 0 ? null : url;
}

function timingSummary(totalMs: number): string {
  const runtime = client.observability.current().runtime;
  if (runtime == null) {
    throw new Error('Speech request completed without runtime observability.');
  }
  const browserHostMs = totalMs - runtime.wasmRunLoopMs;
  return [
    `total_ms=${totalMs.toFixed(1)}`,
    `wasm_inference_loop_ms=${runtime.wasmRunLoopMs.toFixed(1)}`,
    `wasm_inference_loop_calls=${runtime.wasmRunLoopCalls}`,
    `browser_host_ms=${browserHostMs.toFixed(1)}`,
  ].join('\n');
}

function element<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (node == null) {
    throw new Error(`missing #${id}`);
  }
  return node as T;
}
