import { CogentClient } from '@noumena-labs/cogentlm';
import './style.css';

const app = document.querySelector<HTMLDivElement>('#app');

if (!app) {
  throw new Error('missing #app element');
}

app.innerHTML = `
  <section class="shell">
    <header>
      <h1>CogentLM Web</h1>
      <p>Run a local browser model through the public package API.</p>
    </header>
    <form id="model-form" class="panel">
      <label>
        GGUF model URL or path
        <input id="model" placeholder="/models/tiny.gguf" autocomplete="off" />
      </label>
      <button type="submit">Load model</button>
    </form>
    <form id="query-form" class="panel">
      <label>
        Prompt
        <textarea id="prompt" rows="5">Write one sentence about local browser inference.</textarea>
      </label>
      <button type="submit">Run query</button>
    </form>
    <pre id="output">No model loaded.</pre>
  </section>
`;

const modelForm = element<HTMLFormElement>('model-form');
const queryForm = element<HTMLFormElement>('query-form');
const modelInput = element<HTMLInputElement>('model');
const promptInput = element<HTMLTextAreaElement>('prompt');
const output = element<HTMLPreElement>('output');
const client = new CogentClient();
let modelLoaded = false;

modelForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const model = modelInput.value.trim();
  if (!model) {
    write('Enter a GGUF model URL or path.');
    return;
  }

  write(`Loading ${model}...`);
  const info = await client.addLocal(model, {
    runtime: {
      context: { n_ctx: 2048 },
      scheduler: { continuous_batching: true },
    },
  });
  modelLoaded = true;
  write(`Loaded ${info.name}.`);
});

queryForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!modelLoaded) {
    write('Load a model before running a query.');
    return;
  }

  const prompt = promptInput.value.trim();
  if (!prompt) {
    write('Enter a prompt.');
    return;
  }

  write('');
  const run = client.query(prompt, {
    emitTokens: true,
    maxTokens: 64,
    session: 'examples:web',
  });

  for await (const batch of run.tokens) {
    output.textContent += batch.text;
  }

  const result = await run.response;
  output.textContent = result.text.trim();
});

function element<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) {
    throw new Error(`missing #${id}`);
  }
  return node as T;
}

function write(message: string) {
  output.textContent = message;
}
