import {
  CogentClient,
  QueryError,
  type BrowserEmbeddingRun,
  type BrowserTextRun,
  type ChatMessage,
  type EmbeddingResult,
  type EndpointRef,
  type GenerationResult,
  type ModelInfo,
  type ModelSource,
  type NativeRuntimeConfig,
  type RemoteGatewayConfig,
} from '@noumena-labs/cogentlm';
import './style.css';

export const DEFAULT_MAX_TOKENS = 128;
export const DEFAULT_TEMPERATURE = 0.7;
export const DEFAULT_TOP_P = 0.8;
export const EXAMPLE_LOCAL_ENDPOINT: EndpointRef = { kind: 'local', id: 'default' };

export interface LocalPageElements {
  readonly loadForm: HTMLFormElement;
  readonly runForm: HTMLFormElement;
  readonly modelInput: HTMLInputElement;
  readonly modelFileInput: HTMLInputElement;
  readonly promptInput: HTMLTextAreaElement;
  readonly maxTokensInput?: HTMLInputElement;
  readonly output: HTMLPreElement;
}

export interface RemoteGatewayPageElements {
  readonly runForm: HTMLFormElement;
  readonly aliasInput: HTMLInputElement;
  readonly baseUrlInput: HTMLInputElement;
  readonly tokenInput: HTMLInputElement;
  readonly promptInput: HTMLTextAreaElement;
  readonly maxTokensInput?: HTMLInputElement;
  readonly output: HTMLPreElement;
}

export interface ExampleLink {
  readonly href: string;
  readonly label: string;
}

export function renderIndex(title: string, links: readonly ExampleLink[]): void {
  const app = appRoot();
  app.innerHTML = `
    <section class="shell">
      <header class="page-header">
        <h1>${title}</h1>
      </header>
      <nav class="example-grid">
        ${links.map((link) => `<a class="example-link" href="${link.href}">${link.label}</a>`).join('')}
      </nav>
    </section>
  `;
}

export function renderLocalPage(
  title: string,
  defaultPrompt: string,
  includeMaxTokens: boolean
): LocalPageElements {
  const app = appRoot();
  app.innerHTML = `
    <section class="shell">
      ${header(title)}
      <form id="model-form" class="panel">
        <div class="field-row">
          <label>
            GGUF model URL or path
            <input id="model" placeholder="/models/tiny.gguf" autocomplete="off" />
          </label>
          <label>
            GGUF model file
            <input id="model-file" type="file" />
          </label>
        </div>
        <button type="submit">Load model</button>
      </form>
      <form id="run-form" class="panel">
        <label>
          Input
          <textarea id="prompt" rows="5">${defaultPrompt}</textarea>
        </label>
        ${includeMaxTokens ? maxTokensField() : ''}
        <button type="submit">Run</button>
      </form>
      <pre id="output">No model loaded.</pre>
    </section>
  `;
  return {
    loadForm: element('model-form'),
    runForm: element('run-form'),
    modelInput: element('model'),
    modelFileInput: element('model-file'),
    promptInput: element('prompt'),
    maxTokensInput: includeMaxTokens ? element<HTMLInputElement>('max-tokens') : undefined,
    output: element('output'),
  };
}

export function renderRemoteGatewayPage(
  title: string,
  defaultPrompt: string,
  includeMaxTokens: boolean
): RemoteGatewayPageElements {
  const app = appRoot();
  app.innerHTML = `
    <section class="shell">
      ${header(title)}
      <form id="run-form" class="panel">
        <div class="field-row">
          <label>
            Gateway alias
            <input id="alias" value="default" autocomplete="off" />
          </label>
          <label>
            Gateway base URL
            <input id="base-url" placeholder="http://127.0.0.1:8080" autocomplete="off" />
          </label>
        </div>
        <label>
          Gateway token
          <input id="token" type="password" autocomplete="off" />
        </label>
        <label>
          Input
          <textarea id="prompt" rows="5">${defaultPrompt}</textarea>
        </label>
        ${includeMaxTokens ? maxTokensField() : ''}
        <button type="submit">Run</button>
      </form>
      <pre id="output">Ready.</pre>
    </section>
  `;
  return {
    runForm: element('run-form'),
    aliasInput: element('alias'),
    baseUrlInput: element('base-url'),
    tokenInput: element('token'),
    promptInput: element('prompt'),
    maxTokensInput: includeMaxTokens ? element<HTMLInputElement>('max-tokens') : undefined,
    output: element('output'),
  };
}

export function createClient(): CogentClient {
  return new CogentClient();
}

export async function loadLocalModel(
  client: CogentClient,
  source: ModelSource
): Promise<ModelInfo> {
  return client.addLocal(source, {
    runtime: runtimeConfig(),
  });
}

export function readModelSource(
  modelInput: HTMLInputElement,
  fileInput: HTMLInputElement
): ModelSource | null {
  const file = fileInput.files?.[0];
  if (file != null) {
    return file;
  }
  const model = modelInput.value.trim();
  return model === '' ? null : model;
}

export function readPrompt(input: HTMLTextAreaElement): string | null {
  const prompt = input.value.trim();
  return prompt === '' ? null : prompt;
}

export function readMaxTokens(input: HTMLInputElement | undefined): number {
  if (input == null) {
    return DEFAULT_MAX_TOKENS;
  }
  const parsed = Number.parseInt(input.value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_MAX_TOKENS;
}

export function readRemoteGatewayConfig(
  elements: RemoteGatewayPageElements
): RemoteGatewayConfig | null {
  const alias = elements.aliasInput.value.trim();
  const baseUrl = elements.baseUrlInput.value.trim();
  const token = elements.tokenInput.value;
  if (alias === '' || baseUrl === '' || token === '') {
    write(elements.output, 'Enter a gateway alias, base URL, and token.');
    return null;
  }
  return { alias, baseUrl, token };
}

export function localTextRunOptions(
  session: string,
  maxTokens: number,
  emitTokens = true
): {
  readonly emitTokens: boolean;
  readonly maxTokens: number;
  readonly session: string;
  readonly temperature: number;
  readonly topP: number;
} {
  return {
    ...textRunOptions(maxTokens, emitTokens),
    session,
  };
}

export function textRunOptions(
  maxTokens: number,
  emitTokens = true
): {
  readonly emitTokens: boolean;
  readonly maxTokens: number;
  readonly temperature: number;
  readonly topP: number;
} {
  return {
    emitTokens,
    maxTokens,
    temperature: DEFAULT_TEMPERATURE,
    topP: DEFAULT_TOP_P,
  };
}

export async function streamTextRun(
  output: HTMLPreElement,
  endpoint: EndpointRef,
  run: BrowserTextRun
): Promise<GenerationResult> {
  write(output, '');
  let streamed = '';
  for await (const batch of run.tokens) {
    output.textContent += batch.text;
    streamed += batch.text;
  }
  const result = await run.response;
  if (streamed !== '' && streamed !== result.text) {
    throw new Error('streamed token batches did not match final response text');
  }
  write(output, formatTextResult(endpoint, result));
  return result;
}

export async function printEmbeddingRun(
  output: HTMLPreElement,
  endpoint: EndpointRef,
  run: BrowserEmbeddingRun
): Promise<EmbeddingResult> {
  const result = await run.response;
  write(output, formatEmbeddingResult(endpoint, result));
  return result;
}

export function chatMessages(prompt: string): readonly ChatMessage[] {
  return [
    { role: 'system', content: 'Answer concisely.' },
    { role: 'user', content: prompt },
  ];
}

export function write(output: HTMLPreElement, message: string): void {
  output.textContent = message;
}

export function reportError(output: HTMLPreElement, error: unknown): void {
  if (error instanceof QueryError) {
    write(output, `${error.name}: ${error.code}: ${error.message}`);
    return;
  }
  if (error instanceof Error) {
    write(output, `${error.name}: ${error.message}`);
    return;
  }
  write(output, String(error));
}

export function formatTextResult(endpoint: EndpointRef, result: GenerationResult): string {
  const lines = [
    `endpoint=${JSON.stringify(endpoint)}`,
    `finish_reason=${result.finishReason}`,
    `text=${result.text.trim()}`,
  ];
  lines.push(
    `metrics=ttft_ms:${formatMetric(result.stats.ttftMs)} ` +
    `decode_ms:${formatMetric(result.stats.decodeMs)} ` +
    `output_tokens:${result.stats.outputTokens} ` +
    `e2e_tps:${formatMetric(result.stats.e2eTokensPerSecond)} ` +
    `decode_tps:${formatMetric(result.stats.decodeTokensPerSecond)}`
  );
  return lines.join('\n');
}

export function formatEmbeddingResult(endpoint: EndpointRef, result: EmbeddingResult): string {
  const preview = result.values.slice(0, 8).map((value) => value.toFixed(6)).join(', ');
  return [
    `endpoint=${JSON.stringify(endpoint)}`,
    `dimensions=${result.values.length}`,
    `pooling=${result.pooling}`,
    `normalized=${result.normalized}`,
    `preview=[${preview}]`,
  ].join('\n');
}

function runtimeConfig(): NativeRuntimeConfig {
  return {
    context: {
      n_ctx: 2048,
    },
    scheduler: {
      continuous_batching: true,
      prefill_chunk_size: 0,
    },
    cache: {
      mode: 'live_slot_prefix',
    },
    observability: {
      runtime_metrics: true,
    },
  };
}

function formatMetric(value: number | null | undefined): string {
  return typeof value === 'number' ? value.toFixed(3) : 'n/a';
}

function maxTokensField(): string {
  return `
    <label class="short-field">
      Max tokens
      <input id="max-tokens" type="number" min="1" step="1" value="${DEFAULT_MAX_TOKENS}" />
    </label>
  `;
}

function header(title: string): string {
  return `
    <header class="page-header">
      <nav class="top-nav">
        <a href="/query.html">Query</a>
        <a href="/chat.html">Chat</a>
        <a href="/embed.html">Embed</a>
        <a href="/remote_gateway_query.html">Gateway query</a>
        <a href="/remote_gateway_chat.html">Gateway chat</a>
        <a href="/remote_gateway_embed.html">Gateway embed</a>
      </nav>
      <h1>${title}</h1>
    </header>
  `;
}

function appRoot(): HTMLDivElement {
  const app = document.querySelector<HTMLDivElement>('#app');
  if (app == null) {
    throw new Error('missing #app element');
  }
  return app;
}

function element<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (node == null) {
    throw new Error(`missing #${id}`);
  }
  return node as T;
}
