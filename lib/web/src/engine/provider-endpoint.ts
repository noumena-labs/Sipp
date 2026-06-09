import type { ChatMessage, TokenBatch, TokenEmissionStats } from './inference-types.js';
import {
  QueryError,
  type ChatInput,
  type EmbedOptions,
  type EmbeddingResult,
  type EndpointRef,
  type FinishReason,
  type GenerationResult,
  type ProviderEndpointDescriptor,
  type QueryInput,
  type QueryOptions,
  type RequestStats,
} from '../models/types.js';
import { createTimedAbortController } from '../utils/abort.js';

type ProviderKind = 'openai' | 'anthropic' | 'openai_compatible';

export interface ProviderEndpoint {
  readonly id: string;
  readonly provider: ProviderKind;
  readonly model: string;
  readonly baseUrl: string;
  readonly apiKey?: string;
  readonly keyProvider?: () => string | Promise<string>;
  readonly timeoutMs?: number;
  readonly version?: string;
  readonly authHeaderName?: string;
  readonly authHeaderValue?: string;
  readonly authHeaderValueProvider?: () => string | Promise<string>;
  readonly staticHeaders: readonly { readonly name: string; readonly value: string }[];
}

interface ProviderUsage {
  readonly input_tokens?: number;
  readonly output_tokens?: number;
  readonly total_tokens?: number;
}

interface TextStreamState {
  readonly requestId: string;
  readonly secret: string;
  readonly provider: ProviderKind;
  readonly tokenBatchSink: (batch: TokenBatch) => void;
  text: string;
  finishReason: FinishReason;
  finished: boolean;
  usage?: ProviderUsage;
  sequence: number;
  stats: TokenEmissionStats;
}

const OPENAI_CHAT_TYPED_FIELDS = new Set([
  'model',
  'messages',
  'max_tokens',
  'temperature',
  'top_p',
  'stop',
  'stream',
]);
const OPENAI_COMPLETION_TYPED_FIELDS = new Set([
  'model',
  'prompt',
  'max_tokens',
  'temperature',
  'top_p',
  'stop',
  'stream',
]);
const OPENAI_EMBED_TYPED_FIELDS = new Set(['model', 'input', 'encoding_format']);
const ANTHROPIC_CHAT_TYPED_FIELDS = new Set([
  'model',
  'max_tokens',
  'messages',
  'system',
  'temperature',
  'top_p',
  'stop_sequences',
  'stream',
]);
const LOCAL_TEXT_FIELDS = new Set(['contextKey', 'grammar']);
const LOCAL_EMBED_FIELDS = new Set(['contextKey', 'normalize']);
const MAX_PROVIDER_ERROR_BYTES = 1 << 20;
const MAX_PROVIDER_SSE_EVENT_BYTES = 1 << 20;
const U32_MAX = 0xffffffff;
const F32_MAX = 3.4028234663852886e38;
const UTF8_ENCODER = new TextEncoder();
const UTF8_DECODER = new TextDecoder();
const DEFAULT_OPENAI_BASE_URL = 'https://api.openai.com/v1';
const DEFAULT_ANTHROPIC_BASE_URL = 'https://api.anthropic.com/v1';
const DEFAULT_ANTHROPIC_VERSION = '2023-06-01';
const DEFAULT_ANTHROPIC_MAX_TOKENS = 1024;

/** Registry for browser direct provider endpoints. */
export class ProviderEndpointRegistry {
  readonly #providers = new Map<string, ProviderEndpoint>();

  public prepare(id: string, descriptor: ProviderEndpointDescriptor): ProviderEndpoint {
    const normalizedId = normalizeId(id, 'provider id');
    return normalizeProviderDescriptor(normalizedId, descriptor);
  }

  public commit(provider: ProviderEndpoint): EndpointRef {
    this.#providers.set(provider.id, provider);
    return { kind: 'provider', id: provider.id };
  }

  public remove(id: string): void {
    this.#providers.delete(id);
  }

  public get(endpoint: EndpointRef | undefined): ProviderEndpoint | null {
    if (endpoint == null || endpoint.kind !== 'provider') {
      return null;
    }
    const provider = this.#providers.get(endpoint.id);
    if (provider == null) {
      throw new QueryError('MODEL_NOT_FOUND', `provider endpoint not found: ${endpoint.id}`);
    }
    return provider;
  }
}

/** Run a browser query request through a direct provider endpoint. */
export async function runProviderQuery(
  endpoint: ProviderEndpoint,
  input: QueryInput,
  options: QueryOptions,
  tokenBatchSink: ((batch: TokenBatch) => void) | undefined,
  signal: AbortSignal
): Promise<GenerationResult> {
  let prompt: string;
  let hasMedia = false;
  if (typeof input === 'string') {
    prompt = input;
  } else {
    const body = objectInput(input, 'query input');
    prompt = requiredStandaloneString(body.prompt, 'prompt');
    hasMedia = body.media != null && mediaHasEntries(body.media);
  }
  if (prompt.trim().length === 0) {
    throw new QueryError('QUERY_FAILED', 'prompt must not be empty');
  }
  rejectProviderTextInvalidOptions(options, hasMedia);
  if (endpoint.provider === 'anthropic') {
    return runAnthropicChat(endpoint, [{ role: 'user', content: prompt }], options, tokenBatchSink, signal);
  }
  const body = openAiTextBody(
    endpoint.model,
    options,
    options.providerOptions,
    { prompt },
    tokenBatchSink != null,
    OPENAI_COMPLETION_TYPED_FIELDS
  );
  return tokenBatchSink == null
    ? providerRequest(endpoint, '/completions', body, signal, async (response, secret) =>
        parseOpenAiCompletion(await providerJsonBody(response, secret, endpoint.provider))
      )
    : providerRequest(endpoint, '/completions', body, signal, (response, secret, resetTimeout) =>
        readOpenAiTextStream(response, endpoint, secret, tokenBatchSink, resetTimeout)
      );
}

/** Run a browser chat request through a direct provider endpoint. */
export async function runProviderChat(
  endpoint: ProviderEndpoint,
  input: ChatInput,
  options: QueryOptions,
  tokenBatchSink: ((batch: TokenBatch) => void) | undefined,
  signal: AbortSignal
): Promise<GenerationResult> {
  const structuredInput = Array.isArray(input) ? null : objectInput(input, 'chat input');
  const messages = chatMessagesFromInput(input, structuredInput);
  if (messages.length === 0) {
    throw new QueryError('QUERY_FAILED', 'messages must not be empty');
  }
  rejectProviderTextInvalidOptions(
    options,
    structuredInput?.media != null && mediaHasEntries(structuredInput.media)
  );
  if (endpoint.provider === 'anthropic') {
    return runAnthropicChat(endpoint, messages, options, tokenBatchSink, signal);
  }
  const body = openAiTextBody(
    endpoint.model,
    options,
    options.providerOptions,
    { messages: messages.map(providerMessage) },
    tokenBatchSink != null,
    OPENAI_CHAT_TYPED_FIELDS
  );
  return tokenBatchSink == null
    ? providerRequest(endpoint, '/chat/completions', body, signal, async (response, secret) =>
        parseOpenAiChat(await providerJsonBody(response, secret, endpoint.provider))
      )
    : providerRequest(endpoint, '/chat/completions', body, signal, (response, secret, resetTimeout) =>
        readOpenAiTextStream(response, endpoint, secret, tokenBatchSink, resetTimeout)
      );
}

/** Run a browser embedding request through a direct provider endpoint. */
export async function runProviderEmbedding(
  endpoint: ProviderEndpoint,
  input: string,
  options: EmbedOptions,
  signal: AbortSignal
): Promise<EmbeddingResult> {
  const embedInput = requiredStandaloneString(input, 'input');
  if (embedInput.trim().length === 0) {
    throw new QueryError('QUERY_FAILED', 'input must not be empty');
  }
  rejectProviderEmbedInvalidOptions(options);
  if (endpoint.provider === 'anthropic') {
    throw new QueryError(
      'UNSUPPORTED_OPERATION',
      'Anthropic native provider does not expose embeddings',
      { provider: 'anthropic' }
    );
  }
  const body = mergeProviderOptions(
    {
      model: endpoint.model,
      input: embedInput,
      encoding_format: 'float',
    },
    options.providerOptions,
    OPENAI_EMBED_TYPED_FIELDS
  );
  return providerRequest(endpoint, '/embeddings', body, signal, async (response, secret) =>
    parseOpenAiEmbedding(await providerJsonBody(response, secret, endpoint.provider))
  );
}

function normalizeProviderDescriptor(
  id: string,
  descriptor: ProviderEndpointDescriptor
): ProviderEndpoint {
  if (typeof descriptor !== 'object' || descriptor == null || Array.isArray(descriptor)) {
    throw new QueryError('QUERY_FAILED', 'provider descriptor must be an object');
  }
  const provider = normalizeProviderKind(descriptor.provider);
  const model = normalizeId(descriptor.model, 'provider model');
  const timeoutMs = optionalPositiveNumber(descriptor.timeoutMs, 'provider timeoutMs');
  const baseUrl = providerBaseUrl(provider, descriptor.baseUrl);
  validateProviderBaseUrl(baseUrl, 'provider baseUrl');
  const staticHeaders = normalizeStaticHeaders(descriptor.staticHeaders);
  if (provider === 'openai' || provider === 'anthropic') {
    if (
      descriptor.authHeaderName != null ||
      descriptor.authHeaderValue != null ||
      descriptor.authHeaderValueProvider != null ||
      staticHeaders.length > 0
    ) {
      throw new QueryError(
        'QUERY_FAILED',
        'custom auth headers and staticHeaders are only valid for OpenAI-compatible providers'
      );
    }
  }
  const auth = normalizeProviderAuth(provider, descriptor);
  return {
    id,
    provider,
    model,
    baseUrl,
    apiKey: auth.apiKey,
    keyProvider: auth.keyProvider,
    timeoutMs,
    version:
      provider === 'anthropic'
        ? descriptor.version ?? DEFAULT_ANTHROPIC_VERSION
        : undefined,
    authHeaderName: auth.authHeaderName,
    authHeaderValue: auth.authHeaderValue,
    authHeaderValueProvider: auth.authHeaderValueProvider,
    staticHeaders,
  };
}

function normalizeProviderKind(value: unknown): ProviderKind {
  if (value === 'openai' || value === 'anthropic') {
    return value;
  }
  if (value === 'openai_compatible' || value === 'openai-compatible') {
    return 'openai_compatible';
  }
  throw new QueryError(
    'QUERY_FAILED',
    'provider must be one of: openai, anthropic, openai_compatible'
  );
}

function providerBaseUrl(provider: ProviderKind, baseUrl: unknown): string {
  if (baseUrl == null) {
    if (provider === 'openai') {
      return DEFAULT_OPENAI_BASE_URL;
    }
    if (provider === 'anthropic') {
      return DEFAULT_ANTHROPIC_BASE_URL;
    }
    throw new QueryError(
      'QUERY_FAILED',
      'provider baseUrl is required for OpenAI-compatible providers'
    );
  }
  if (typeof baseUrl !== 'string') {
    throw new QueryError('QUERY_FAILED', 'provider baseUrl must be a string');
  }
  const trimmed = baseUrl.trim();
  if (trimmed.length === 0) {
    throw new QueryError('QUERY_FAILED', 'provider baseUrl must not be empty');
  }
  if (trimmed !== baseUrl) {
    throw new QueryError(
      'QUERY_FAILED',
      'provider baseUrl must not contain surrounding whitespace'
    );
  }
  return baseUrl.replace(/\/+$/u, '');
}

function normalizeProviderAuth(
  provider: ProviderKind,
  descriptor: ProviderEndpointDescriptor
): {
  readonly apiKey?: string;
  readonly keyProvider?: () => string | Promise<string>;
  readonly authHeaderName?: string;
  readonly authHeaderValue?: string;
  readonly authHeaderValueProvider?: () => string | Promise<string>;
} {
  const hasApiKey = descriptor.apiKey != null;
  const hasKeyProvider = descriptor.keyProvider != null;
  const hasHeader = descriptor.authHeaderName != null || descriptor.authHeaderValue != null;
  const hasHeaderProvider = descriptor.authHeaderValueProvider != null;
  if (provider === 'openai' || provider === 'anthropic') {
    if (hasApiKey === hasKeyProvider) {
      throw new QueryError('QUERY_FAILED', 'provider requires apiKey or keyProvider, not both');
    }
    if (descriptor.apiKey != null) {
      validateProviderSecret(descriptor.apiKey, 'provider apiKey');
    }
    if (descriptor.keyProvider != null && typeof descriptor.keyProvider !== 'function') {
      throw new QueryError('QUERY_FAILED', 'provider keyProvider must be a function');
    }
    return { apiKey: descriptor.apiKey, keyProvider: descriptor.keyProvider };
  }
  if (hasApiKey || hasKeyProvider) {
    if (hasApiKey === hasKeyProvider || hasHeader || hasHeaderProvider) {
      throw new QueryError(
        'QUERY_FAILED',
        'OpenAI-compatible provider requires either apiKey/keyProvider or ' +
          'authHeaderName with authHeaderValue/authHeaderValueProvider'
      );
    }
    if (descriptor.apiKey != null) {
      validateProviderSecret(descriptor.apiKey, 'provider apiKey');
    }
    return { apiKey: descriptor.apiKey, keyProvider: descriptor.keyProvider };
  }
  if (descriptor.authHeaderName == null) {
    throw new QueryError('QUERY_FAILED', 'provider authHeaderName is required');
  }
  if (typeof descriptor.authHeaderName !== 'string' || descriptor.authHeaderName.trim().length === 0) {
    throw new QueryError('QUERY_FAILED', 'provider authHeaderName must not be empty');
  }
  if (descriptor.authHeaderValue != null && descriptor.authHeaderValueProvider != null) {
    throw new QueryError(
      'QUERY_FAILED',
      'provider must set authHeaderValue or authHeaderValueProvider, not both'
    );
  }
  if (descriptor.authHeaderValue == null && descriptor.authHeaderValueProvider == null) {
    throw new QueryError(
      'QUERY_FAILED',
      'provider authHeaderValue or authHeaderValueProvider is required'
    );
  }
  if (descriptor.authHeaderValue != null) {
    validateProviderSecret(descriptor.authHeaderValue, 'provider authHeaderValue');
  }
  if (
    descriptor.authHeaderValueProvider != null &&
    typeof descriptor.authHeaderValueProvider !== 'function'
  ) {
    throw new QueryError('QUERY_FAILED', 'provider authHeaderValueProvider must be a function');
  }
  return {
    authHeaderName: descriptor.authHeaderName,
    authHeaderValue: descriptor.authHeaderValue,
    authHeaderValueProvider: descriptor.authHeaderValueProvider,
  };
}

function normalizeStaticHeaders(
  headers: ProviderEndpointDescriptor['staticHeaders']
): readonly { readonly name: string; readonly value: string }[] {
  if (headers == null) {
    return [];
  }
  if (!Array.isArray(headers)) {
    throw new QueryError('QUERY_FAILED', 'provider staticHeaders must be an array');
  }
  return headers.map((header) => {
    if (typeof header !== 'object' || header == null || Array.isArray(header)) {
      throw new QueryError('QUERY_FAILED', 'provider staticHeaders entries must be objects');
    }
    if (typeof header.name !== 'string' || header.name.trim().length === 0) {
      throw new QueryError('QUERY_FAILED', 'provider static header name must not be empty');
    }
    if (typeof header.value !== 'string') {
      throw new QueryError('QUERY_FAILED', 'provider static header value must be a string');
    }
    return { name: header.name, value: header.value };
  });
}

function openAiTextBody(
  model: string,
  options: QueryOptions,
  providerOptions: unknown,
  payload: { readonly prompt: string } | { readonly messages: readonly unknown[] },
  stream: boolean,
  typedFields: ReadonlySet<string>
): Record<string, unknown> {
  const maxTokens = optionalPositiveU32(options.maxTokens, 'max_tokens');
  const temperature = optionalTemperature(options.temperature);
  const topP = optionalTopP(options.topP);
  const stop = optionalStringArray(options.stop, 'stop');
  return mergeProviderOptions(
    {
      model,
      ...payload,
      ...(maxTokens == null ? {} : { max_tokens: maxTokens }),
      ...(temperature == null ? {} : { temperature }),
      ...(topP == null ? {} : { top_p: topP }),
      ...(stop == null ? {} : { stop }),
      ...(stream ? { stream: true } : {}),
    },
    providerOptions,
    typedFields
  );
}

function runAnthropicChat(
  endpoint: ProviderEndpoint,
  messages: readonly ChatMessage[],
  options: QueryOptions,
  tokenBatchSink: ((batch: TokenBatch) => void) | undefined,
  signal: AbortSignal
): Promise<GenerationResult> {
  const body = anthropicBody(endpoint.model, messages, options, tokenBatchSink != null);
  return tokenBatchSink == null
    ? providerRequest(endpoint, '/messages', body, signal, async (response, secret) =>
        parseAnthropicText(await providerJsonBody(response, secret, endpoint.provider))
      )
    : providerRequest(endpoint, '/messages', body, signal, (response, secret, resetTimeout) =>
        readAnthropicTextStream(response, endpoint, secret, tokenBatchSink, resetTimeout)
      );
}

function anthropicBody(
  model: string,
  messages: readonly ChatMessage[],
  options: QueryOptions,
  stream: boolean
): Record<string, unknown> {
  const { system, conversation } = anthropicMessages(messages);
  if (conversation.length === 0) {
    throw new QueryError(
      'QUERY_FAILED',
      'Anthropic messages must include at least one user or assistant message'
    );
  }
  const maxTokens = optionalPositiveU32(options.maxTokens, 'max_tokens') ?? DEFAULT_ANTHROPIC_MAX_TOKENS;
  const temperature = optionalTemperature(options.temperature);
  const topP = optionalTopP(options.topP);
  const stop = optionalStringArray(options.stop, 'stop_sequences');
  return mergeProviderOptions(
    {
      model,
      messages: conversation,
      ...(system == null ? {} : { system }),
      max_tokens: maxTokens,
      ...(temperature == null ? {} : { temperature }),
      ...(topP == null ? {} : { top_p: topP }),
      ...(stop == null ? {} : { stop_sequences: stop }),
      ...(stream ? { stream: true } : {}),
    },
    options.providerOptions,
    ANTHROPIC_CHAT_TYPED_FIELDS
  );
}

function anthropicMessages(messages: readonly ChatMessage[]): {
  readonly system?: string;
  readonly conversation: readonly Record<string, string>[];
} {
  const system: string[] = [];
  const conversation: Record<string, string>[] = [];
  for (const message of messages) {
    if (message.role === 'system') {
      if (message.content.trim().length > 0) {
        system.push(message.content);
      }
      continue;
    }
    conversation.push(providerMessage(message));
  }
  return {
    system: system.length === 0 ? undefined : system.join('\n\n'),
    conversation,
  };
}

async function providerRequest<T>(
  endpoint: ProviderEndpoint,
  path: string,
  body: Record<string, unknown>,
  signal: AbortSignal,
  read: (response: Response, secret: string, resetTimeout: () => void) => Promise<T>
): Promise<T> {
  const abort = createTimedAbortController(signal, endpoint.timeoutMs);
  let secret = '';
  try {
    throwIfProviderAborted(abort.signal, abort.timedOut());
    secret = await providerSecret(endpoint, abort.signal);
    throwIfProviderAborted(abort.signal, abort.timedOut());
    const response = await fetch(`${endpoint.baseUrl}${path}`, {
      method: 'POST',
      headers: requestHeaders(endpoint, secret),
      body: JSON.stringify(body),
      credentials: 'omit',
      mode: 'cors',
      redirect: 'error',
      signal: abort.signal,
    });
    if (!response.ok) {
      throw await providerError(response, endpoint, secret);
    }
    return await read(response, secret, abort.resetTimeout);
  } catch (error) {
    if (abort.timedOut()) {
      throw new QueryError('QUERY_FAILED', 'provider request timed out', {
        provider: endpoint.provider,
      });
    }
    if (abort.signal.aborted) {
      throw new QueryError('QUERY_FAILED', 'provider request aborted', {
        provider: endpoint.provider,
      });
    }
    if (error instanceof QueryError) {
      throw error;
    }
    throw new QueryError('QUERY_FAILED', 'provider request failed', {
      provider: endpoint.provider,
      cause: error,
    });
  } finally {
    abort.dispose();
  }
}

function requestHeaders(endpoint: ProviderEndpoint, secret: string): Headers {
  const headers = new Headers();
  headers.set('Content-Type', 'application/json');
  for (const header of endpoint.staticHeaders) {
    headers.set(header.name, header.value);
  }
  if (endpoint.provider === 'anthropic') {
    headers.set('x-api-key', secret);
    headers.set('anthropic-version', endpoint.version ?? DEFAULT_ANTHROPIC_VERSION);
  } else if (endpoint.authHeaderName != null) {
    headers.set(endpoint.authHeaderName, secret);
  } else {
    headers.set('Authorization', `Bearer ${secret}`);
  }
  return headers;
}

async function providerSecret(endpoint: ProviderEndpoint, signal: AbortSignal): Promise<string> {
  let value: unknown;
  try {
    value =
      endpoint.apiKey ??
      endpoint.authHeaderValue ??
      (await providerSecretFromCallback(
        endpoint.keyProvider ?? endpoint.authHeaderValueProvider,
        signal
      ));
  } catch {
    if (signal.aborted) {
      throw new Error('provider key provider aborted');
    }
    throw new QueryError('QUERY_FAILED', 'provider key provider failed', {
      provider: endpoint.provider,
    });
  }
  if (typeof value !== 'string') {
    throw new QueryError('QUERY_FAILED', 'provider key must be a string', {
      provider: endpoint.provider,
    });
  }
  validateProviderSecret(value, 'provider key');
  return value;
}

async function providerSecretFromCallback(
  keyProvider: (() => string | Promise<string>) | undefined,
  signal: AbortSignal
): Promise<unknown> {
  if (keyProvider == null) {
    return undefined;
  }
  if (signal.aborted) {
    throw new Error('provider key provider aborted');
  }
  let removeAbortListener = (): void => {};
  const abortPromise = new Promise<never>((_resolve, reject) => {
    const abortListener = (): void => reject(new Error('provider key provider aborted'));
    signal.addEventListener('abort', abortListener, { once: true });
    removeAbortListener = () => signal.removeEventListener('abort', abortListener);
  });
  try {
    const providerPromise = Promise.resolve().then(() => keyProvider());
    return await Promise.race([providerPromise, abortPromise]);
  } finally {
    removeAbortListener();
  }
}

async function providerJsonBody(
  response: Response,
  secret: string,
  provider: ProviderKind
): Promise<Record<string, unknown>> {
  const body = objectValue(await response.json(), 'provider response');
  rejectProviderBodyError(body, secret, provider, providerRequestId(response.headers));
  return body;
}

function rejectProviderBodyError(
  body: Record<string, unknown>,
  secret: string,
  provider: ProviderKind,
  requestId: string | undefined
): void {
  if (body.error == null) {
    return;
  }
  throw new QueryError('QUERY_FAILED', redactSecret(providerBodyErrorMessage(body), secret), {
    provider,
    providerCode: redactOptionalSecret(providerErrorCode(body), secret),
    requestId: redactOptionalSecret(requestId, secret),
  });
}

async function providerError(
  response: Response,
  endpoint: ProviderEndpoint,
  secret: string
): Promise<QueryError> {
  const body = await providerErrorBody(response);
  const message =
    typeof body === 'object' && body != null
      ? providerBodyErrorMessage(body as Record<string, unknown>)
      : response.statusText || 'provider error';
  return new QueryError('QUERY_FAILED', redactSecret(message, secret), {
    status: response.status,
    provider: endpoint.provider,
    providerCode: redactOptionalSecret(providerErrorCode(body), secret),
    requestId: redactOptionalSecret(providerRequestId(response.headers), secret),
    retryAfterMs: retryAfterMs(response.headers),
  });
}

async function providerErrorBody(response: Response): Promise<unknown> {
  const text = await responseTextWithinLimit(response);
  if (text == null) {
    return { error: { message: 'provider error response exceeded body limit' } };
  }
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

async function responseTextWithinLimit(response: Response): Promise<string | null> {
  if (response.body == null) {
    const text = await response.text();
    return UTF8_ENCODER.encode(text).byteLength > MAX_PROVIDER_ERROR_BYTES ? null : text;
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    totalBytes += value.byteLength;
    if (totalBytes > MAX_PROVIDER_ERROR_BYTES) {
      try {
        await reader.cancel();
      } catch {
        // Preserve bounded error result when cancellation itself fails.
      }
      return null;
    }
    chunks.push(value);
  }
  const body = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return UTF8_DECODER.decode(body);
}

function parseOpenAiChat(body: Record<string, unknown>): GenerationResult {
  const choice = firstChoice(body, 'chat response');
  const message = objectValue(choice.message, 'chat response message');
  const text = typeof message.content === 'string' ? message.content : '';
  return textResult(body, text, choice.finish_reason);
}

function parseOpenAiCompletion(body: Record<string, unknown>): GenerationResult {
  const choice = firstChoice(body, 'completion response');
  return textResult(body, requiredStringField(choice, 'text', 'completion response'), choice.finish_reason);
}

function parseOpenAiEmbedding(body: Record<string, unknown>): EmbeddingResult {
  const data = arrayField(body, 'data', 'embedding response');
  const first = objectValue(data[0], 'embedding response item');
  const values = numericArray(first.embedding, 'embedding');
  const usage = openAiUsage(body.usage);
  return {
    id: stringField(body, 'id', 'provider_embed'),
    values,
    pooling: 'none',
    normalized: false,
    stats: requestStats(usage),
  };
}

function parseAnthropicText(body: Record<string, unknown>): GenerationResult {
  const content = arrayField(body, 'content', 'Anthropic response');
  const text = content
    .map((item) => objectValue(item, 'Anthropic content block'))
    .filter((item) => item.type === 'text')
    .map((item) => requiredStringField(item, 'text', 'Anthropic text content block'))
    .join('');
  return {
    id: stringField(body, 'id', 'provider_text'),
    text,
    finishReason: finishReason(body.stop_reason),
    stats: requestStats(anthropicUsage(body.usage)),
  };
}

async function readOpenAiTextStream(
  response: Response,
  endpoint: ProviderEndpoint,
  secret: string,
  tokenBatchSink: (batch: TokenBatch) => void,
  resetTimeout: () => void
): Promise<GenerationResult> {
  const state = textStreamState(response, endpoint, secret, tokenBatchSink);
  await readSseStream(response, resetTimeout, (payload) => pushOpenAiStreamPayload(state, payload));
  if (!state.finished) {
    throw new QueryError('QUERY_FAILED', 'OpenAI-compatible stream ended before finish_reason', {
      provider: endpoint.provider,
    });
  }
  return streamResult(state);
}

async function readAnthropicTextStream(
  response: Response,
  endpoint: ProviderEndpoint,
  secret: string,
  tokenBatchSink: (batch: TokenBatch) => void,
  resetTimeout: () => void
): Promise<GenerationResult> {
  const state = textStreamState(response, endpoint, secret, tokenBatchSink);
  await readSseStream(response, resetTimeout, (payload) =>
    pushAnthropicStreamPayload(state, payload)
  );
  if (!state.finished) {
    throw new QueryError('QUERY_FAILED', 'Anthropic stream ended before message_stop', {
      provider: endpoint.provider,
    });
  }
  return streamResult(state);
}

async function readSseStream(
  response: Response,
  resetTimeout: () => void,
  pushPayload: (payload: string) => void
): Promise<void> {
  if (response.body == null) {
    throw new QueryError('STREAMING_UNAVAILABLE', 'provider response body is not streamable');
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  resetTimeout();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    resetTimeout();
    buffer += value;
    let boundary = eventBoundary(buffer);
    while (boundary != null) {
      const raw = buffer.slice(0, boundary.index);
      buffer = buffer.slice(boundary.index + boundary.length);
      assertStreamEventWithinLimit(raw);
      for (const payload of parseSsePayloads(raw)) {
        pushPayload(payload);
      }
      boundary = eventBoundary(buffer);
    }
    assertStreamEventWithinLimit(buffer);
  }
  if (buffer.trim().length > 0) {
    assertStreamEventWithinLimit(buffer);
    for (const payload of parseSsePayloads(buffer)) {
      pushPayload(payload);
    }
  }
}

function pushOpenAiStreamPayload(state: TextStreamState, payload: string): void {
  if (payload.trim() === '[DONE]') {
    if (!state.finished) {
      state.finished = true;
    }
    return;
  }
  const body = parseStreamJson(payload, state.provider);
  rejectProviderStreamError(body, state);
  if (body.usage != null) {
    state.usage = openAiUsage(body.usage);
  }
  const choices = body.choices;
  if (!Array.isArray(choices)) {
    return;
  }
  for (const rawChoice of choices) {
    const choice = objectValue(rawChoice, 'stream choice');
    const delta = typeof choice.text === 'string'
      ? choice.text
      : stringFromObjectField(choice.delta, 'content');
    if (delta != null) {
      pushText(state, delta);
    }
    if (typeof choice.finish_reason === 'string') {
      state.finishReason = finishReason(choice.finish_reason);
      state.finished = true;
    }
  }
}

function pushAnthropicStreamPayload(state: TextStreamState, payload: string): void {
  const body = parseStreamJson(payload, state.provider);
  rejectProviderStreamError(body, state);
  const eventType = body.type;
  if (eventType === 'content_block_delta') {
    const delta = objectValue(body.delta, 'Anthropic stream delta');
    if (delta.type === 'text_delta') {
      pushText(state, requiredStringField(delta, 'text', 'Anthropic text_delta'));
    }
  } else if (eventType === 'message_start') {
    const message = objectValue(body.message, 'Anthropic message_start');
    state.usage = anthropicUsage(message.usage) ?? state.usage;
  } else if (eventType === 'message_delta') {
    const delta = objectValue(body.delta, 'Anthropic message_delta');
    if (typeof delta.stop_reason === 'string') {
      state.finishReason = finishReason(delta.stop_reason);
    }
    state.usage = anthropicUsage(body.usage) ?? state.usage;
  } else if (eventType === 'message_stop') {
    state.finished = true;
  }
}

function rejectProviderStreamError(body: Record<string, unknown>, state: TextStreamState): void {
  if (body.type !== 'error' && body.error == null) {
    return;
  }
  throw new QueryError('QUERY_FAILED', redactSecret(providerBodyErrorMessage(body), state.secret), {
    provider: state.provider,
    providerCode: redactOptionalSecret(providerErrorCode(body), state.secret),
    requestId: redactOptionalSecret(state.requestId || undefined, state.secret),
  });
}

function textStreamState(
  response: Response,
  endpoint: ProviderEndpoint,
  secret: string,
  tokenBatchSink: (batch: TokenBatch) => void
): TextStreamState {
  return {
    requestId: providerRequestId(response.headers) ?? '',
    secret,
    provider: endpoint.provider,
    tokenBatchSink,
    text: '',
    finishReason: 'stop',
    finished: false,
    sequence: 0,
    stats: emptyTokenEmissionStats(),
  };
}

function streamResult(state: TextStreamState): GenerationResult {
  return {
    id: state.requestId || 'provider_stream',
    text: state.text,
    finishReason: state.finishReason,
    stats: requestStats(state.usage),
  };
}

function pushText(state: TextStreamState, text: string): void {
  state.text += text;
  const byteCount = UTF8_ENCODER.encode(text).byteLength;
  state.stats = {
    framesSent: state.stats.framesSent + 1,
    bytesSent: state.stats.bytesSent + byteCount,
    batchesSent: state.stats.batchesSent + 1,
    drainMs: state.stats.drainMs,
    drainCalls: state.stats.drainCalls,
  };
  const batch: TokenBatch = {
    requestId: state.requestId,
    streamId: 0,
    sequenceStart: state.sequence,
    text,
    frameCount: 1,
    byteCount,
    stats: state.stats,
  };
  state.sequence += 1;
  state.tokenBatchSink(batch);
}

function mergeProviderOptions(
  body: Record<string, unknown>,
  providerOptions: unknown,
  typedFields: ReadonlySet<string>
): Record<string, unknown> {
  if (providerOptions == null) {
    return body;
  }
  if (typeof providerOptions !== 'object' || Array.isArray(providerOptions)) {
    throw new QueryError('QUERY_FAILED', 'providerOptions must be a JSON object');
  }
  if (!isJsonObject(providerOptions)) {
    throw new QueryError('QUERY_FAILED', 'providerOptions must be a JSON object');
  }
  for (const [key, value] of Object.entries(providerOptions)) {
    if (typedFields.has(key)) {
      throw new QueryError('QUERY_FAILED', `providerOptions cannot override typed field: ${key}`);
    }
    body[key] = snapshotJsonCompatibleProviderOption(value);
  }
  return body;
}

function rejectProviderTextInvalidOptions(options: QueryOptions, hasMedia: boolean): void {
  if (options.endpointOptions != null) {
    throw new QueryError('UNSUPPORTED_OPERATION', 'endpointOptions are not valid for provider endpoints');
  }
  if (hasMedia) {
    throw new QueryError('UNSUPPORTED_OPERATION', 'local media options are not valid for provider endpoints');
  }
  for (const field of LOCAL_TEXT_FIELDS) {
    if ((options as Record<string, unknown>)[field] != null) {
      throw new QueryError('UNSUPPORTED_OPERATION', 'local text options are not valid for provider endpoints');
    }
  }
}

function rejectProviderEmbedInvalidOptions(options: EmbedOptions): void {
  if (options.endpointOptions != null) {
    throw new QueryError('UNSUPPORTED_OPERATION', 'endpointOptions are not valid for provider endpoints');
  }
  for (const field of LOCAL_EMBED_FIELDS) {
    if ((options as Record<string, unknown>)[field] != null) {
      throw new QueryError('UNSUPPORTED_OPERATION', 'local embed options are not valid for provider endpoints');
    }
  }
}

function firstChoice(body: Record<string, unknown>, context: string): Record<string, unknown> {
  const choices = arrayField(body, 'choices', context);
  return objectValue(choices[0], `${context} first choice`);
}

function textResult(
  body: Record<string, unknown>,
  text: string,
  rawFinishReason: unknown
): GenerationResult {
  return {
    id: stringField(body, 'id', 'provider_text'),
    text,
    finishReason: finishReason(rawFinishReason),
    stats: requestStats(openAiUsage(body.usage)),
  };
}

function requestStats(usage: ProviderUsage | undefined): RequestStats {
  return {
    inputTokens: usage?.input_tokens ?? 0,
    outputTokens: usage?.output_tokens ?? 0,
    cacheMode: null,
    cacheSource: null,
    cacheHits: 0,
    prefillTokens: usage?.input_tokens ?? null,
    ttftMs: null,
    interTokenMs: null,
    e2eMs: null,
    decodeTokensPerSecond: null,
    e2eTokensPerSecond: null,
    prefillTokensPerSecond: null,
    prefillMs: 0,
    decodeMs: 0,
  };
}

function openAiUsage(value: unknown): ProviderUsage | undefined {
  if (value == null) {
    return undefined;
  }
  const body = objectValue(value, 'usage');
  return {
    input_tokens: optionalUsageU32(body, 'prompt_tokens'),
    output_tokens: optionalUsageU32(body, 'completion_tokens'),
    total_tokens: optionalUsageU32(body, 'total_tokens'),
  };
}

function anthropicUsage(value: unknown): ProviderUsage | undefined {
  if (value == null) {
    return undefined;
  }
  const body = objectValue(value, 'usage');
  const input = checkedUsageSum([
    optionalUsageU32(body, 'input_tokens'),
    optionalUsageU32(body, 'cache_creation_input_tokens'),
    optionalUsageU32(body, 'cache_read_input_tokens'),
  ]);
  const output = optionalUsageU32(body, 'output_tokens');
  return {
    input_tokens: input,
    output_tokens: output,
    total_tokens: tokenUsageTotal(input, output),
  };
}

function snapshotJsonCompatibleProviderOption(value: unknown): unknown {
  return snapshotJsonCompatibleValue(value, new WeakSet<object>());
}

function snapshotJsonCompatibleValue(value: unknown, ancestors: WeakSet<object>): unknown {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new QueryError('QUERY_FAILED', 'providerOptions cannot contain non-finite numbers');
    }
    return value;
  }
  if (typeof value !== 'object') {
    throw new QueryError('QUERY_FAILED', 'providerOptions must contain JSON-compatible values');
  }
  if (ancestors.has(value)) {
    throw new QueryError('QUERY_FAILED', 'providerOptions must contain JSON-compatible values');
  }
  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      return value.map((item) => snapshotJsonCompatibleValue(item, ancestors));
    }
    if (!isJsonObject(value)) {
      throw new QueryError('QUERY_FAILED', 'providerOptions must contain JSON-compatible values');
    }
    const snapshot: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value)) {
      snapshot[key] = snapshotJsonCompatibleValue(item, ancestors);
    }
    return snapshot;
  } finally {
    ancestors.delete(value);
  }
}

function providerMessage(message: ChatMessage): Record<string, string> {
  if (message.role !== 'system' && message.role !== 'user' && message.role !== 'assistant') {
    throw new QueryError('QUERY_FAILED', 'message role must be system, user, or assistant');
  }
  return {
    role: message.role,
    content: requiredStandaloneString(message.content, 'message content'),
  };
}

function chatMessagesFromInput(
  input: ChatInput,
  structuredInput: Record<string, unknown> | null
): readonly ChatMessage[] {
  const rawMessages = Array.isArray(input) ? input : structuredInput?.messages;
  if (!Array.isArray(rawMessages)) {
    throw new QueryError('QUERY_FAILED', 'messages must be an array');
  }
  return rawMessages.map((value) => {
    const message = objectInput(value, 'message');
    return {
      role: requiredStandaloneString(message.role, 'message role') as ChatMessage['role'],
      content: requiredStandaloneString(message.content, 'message content'),
    };
  });
}

function objectInput(value: unknown, name: string): Record<string, unknown> {
  if (typeof value !== 'object' || value == null || Array.isArray(value)) {
    throw new QueryError('QUERY_FAILED', `${name} must be a JSON object`);
  }
  return value as Record<string, unknown>;
}

function objectValue(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== 'object' || value == null || Array.isArray(value)) {
    throw new QueryError('QUERY_FAILED', `${context} must be a JSON object`);
  }
  return value as Record<string, unknown>;
}

function arrayField(
  body: Record<string, unknown>,
  key: string,
  context: string
): readonly unknown[] {
  const value = body[key];
  if (!Array.isArray(value)) {
    throw new QueryError('QUERY_FAILED', `${context} missing ${key}`);
  }
  return value;
}

function requiredStandaloneString(value: unknown, key: string): string {
  if (typeof value !== 'string') {
    throw new QueryError('QUERY_FAILED', `${key} must be a string`);
  }
  return value;
}

function requiredStringField(
  body: Record<string, unknown>,
  key: string,
  context: string
): string {
  return requiredStandaloneString(body[key], `${context} ${key}`);
}

function stringField(body: Record<string, unknown>, key: string, fallback: string): string {
  const value = body[key];
  return typeof value === 'string' ? value : fallback;
}

function stringFromObjectField(value: unknown, key: string): string | undefined {
  if (value == null) {
    return undefined;
  }
  const body = objectValue(value, 'stream delta');
  return typeof body[key] === 'string' ? body[key] : undefined;
}

function numericArray(value: unknown, key: string): number[] {
  if (!Array.isArray(value)) {
    throw new QueryError('QUERY_FAILED', `provider response missing ${key}`);
  }
  return value.map((item) => {
    if (typeof item !== 'number' || !Number.isFinite(item)) {
      throw new QueryError('QUERY_FAILED', `provider ${key} contains non-finite value`);
    }
    if (item < -F32_MAX || item > F32_MAX) {
      throw new QueryError('QUERY_FAILED', `provider ${key} contains value outside f32 range`);
    }
    return item;
  });
}

function optionalPositiveU32(value: unknown, key: string): number | undefined {
  if (value == null) {
    return undefined;
  }
  if (!Number.isInteger(value) || typeof value !== 'number' || value < 0 || value > U32_MAX) {
    throw new QueryError('QUERY_FAILED', `${key} must be a u32 integer`);
  }
  if (value === 0) {
    throw new QueryError('QUERY_FAILED', `${key} must be greater than zero`);
  }
  return value;
}

function optionalPositiveNumber(value: unknown, key: string): number | undefined {
  if (value == null) {
    return undefined;
  }
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new QueryError('QUERY_FAILED', `${key} must be positive`);
  }
  return value;
}

function optionalFiniteNumber(value: unknown, key: string): number | undefined {
  if (value == null) {
    return undefined;
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new QueryError('QUERY_FAILED', `${key} must be finite`);
  }
  return value;
}

function optionalTemperature(value: unknown): number | undefined {
  const temperature = optionalFiniteNumber(value, 'temperature');
  if (temperature != null && temperature < 0) {
    throw new QueryError('QUERY_FAILED', 'temperature must be greater than or equal to zero');
  }
  return temperature;
}

function optionalTopP(value: unknown): number | undefined {
  const topP = optionalFiniteNumber(value, 'top_p');
  if (topP != null && (topP < 0 || topP > 1)) {
    throw new QueryError('QUERY_FAILED', 'top_p must be between 0 and 1');
  }
  return topP;
}

function optionalStringArray(value: unknown, key: string): readonly string[] | undefined {
  if (value == null) {
    return undefined;
  }
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'string')) {
    throw new QueryError('QUERY_FAILED', `${key} must be an array of strings`);
  }
  return [...value];
}

function optionalUsageU32(body: Record<string, unknown>, key: string): number | undefined {
  if (!Object.prototype.hasOwnProperty.call(body, key)) {
    return undefined;
  }
  const value = body[key];
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0 || value > U32_MAX) {
    throw new QueryError('QUERY_FAILED', `usage field is not a u32 integer: ${key}`);
  }
  return value;
}

function checkedUsageSum(values: readonly (number | undefined)[]): number | undefined {
  let total: number | undefined;
  for (const value of values) {
    if (value == null) {
      continue;
    }
    total = (total ?? 0) + value;
    if (total > U32_MAX) {
      throw new QueryError('QUERY_FAILED', 'usage field exceeds u32');
    }
  }
  return total;
}

function tokenUsageTotal(input: number | undefined, output: number | undefined): number | undefined {
  if (input == null || output == null) {
    return undefined;
  }
  const total = input + output;
  return total > U32_MAX ? undefined : total;
}

function finishReason(value: unknown): FinishReason {
  return value === 'length' || value === 'max_tokens' || value === 'max_output_tokens'
    ? 'length'
    : 'stop';
}

function mediaHasEntries(value: unknown): boolean {
  return Array.isArray(value) ? value.length > 0 : true;
}

function eventBoundary(buffer: string): { readonly index: number; readonly length: number } | null {
  const crlf = buffer.indexOf('\r\n\r\n');
  const lf = buffer.indexOf('\n\n');
  if (crlf === -1 && lf === -1) {
    return null;
  }
  if (crlf !== -1 && (lf === -1 || crlf < lf)) {
    return { index: crlf, length: 4 };
  }
  return { index: lf, length: 2 };
}

function parseSsePayloads(raw: string): readonly string[] {
  const data: string[] = [];
  for (const line of raw.split(/\r?\n/u)) {
    if (line.startsWith('data:')) {
      data.push(line.slice('data:'.length).trimStart());
    }
  }
  return data.length === 0 ? [] : [data.join('\n')];
}

function assertStreamEventWithinLimit(raw: string): void {
  if (UTF8_ENCODER.encode(raw).byteLength > MAX_PROVIDER_SSE_EVENT_BYTES) {
    throw new QueryError('QUERY_FAILED', 'provider stream event exceeded buffer limit');
  }
}

function parseStreamJson(payload: string, provider: ProviderKind): Record<string, unknown> {
  let value: unknown;
  try {
    value = JSON.parse(payload) as unknown;
  } catch {
    throw new QueryError('QUERY_FAILED', 'invalid provider stream JSON payload', { provider });
  }
  return objectValue(value, 'provider stream payload');
}

function providerBodyErrorMessage(body: Record<string, unknown>): string {
  if (typeof body.message === 'string' && body.message.length > 0) {
    return body.message;
  }
  if (typeof body.error === 'object' && body.error != null) {
    const error = body.error as { readonly message?: unknown };
    if (typeof error.message === 'string' && error.message.length > 0) {
      return error.message;
    }
  }
  return 'provider error';
}

function providerErrorCode(body: unknown): string | undefined {
  if (typeof body !== 'object' || body == null) {
    return undefined;
  }
  const record = body as Record<string, unknown>;
  const error = typeof record.error === 'object' && record.error != null
    ? (record.error as Record<string, unknown>)
    : record;
  if (typeof error.code === 'string') {
    return error.code;
  }
  return typeof error.type === 'string' ? error.type : undefined;
}

function providerRequestId(headers: Headers): string | undefined {
  return headers.get('x-request-id') ?? headers.get('request-id') ?? undefined;
}

function retryAfterMs(headers: Headers): number | undefined {
  const retryAfterMs = positiveIntegerHeader(headers.get('retry-after-ms'));
  if (retryAfterMs != null) {
    return retryAfterMs;
  }
  const retryAfterSeconds = positiveIntegerHeader(headers.get('retry-after'));
  if (retryAfterSeconds == null || retryAfterSeconds > Number.MAX_SAFE_INTEGER / 1000) {
    return undefined;
  }
  return retryAfterSeconds * 1000;
}

function positiveIntegerHeader(value: string | null): number | undefined {
  if (value == null) {
    return undefined;
  }
  const trimmed = value.trim();
  if (!/^\d+$/u.test(trimmed)) {
    return undefined;
  }
  const parsed = Number(trimmed);
  return Number.isSafeInteger(parsed) ? parsed : undefined;
}

function validateProviderSecret(value: string, name: string): void {
  if (value.trim().length === 0) {
    throw new QueryError('QUERY_FAILED', `${name} must not be empty`);
  }
  if (/\s/u.test(value)) {
    throw new QueryError('QUERY_FAILED', `${name} must not contain whitespace`);
  }
}

function validateProviderBaseUrl(baseUrl: string, name: string): void {
  let url: URL;
  try {
    url = new URL(baseUrl);
  } catch {
    throw new QueryError('QUERY_FAILED', `${name} is invalid`);
  }
  if ((url.protocol !== 'http:' && url.protocol !== 'https:') || url.hostname.length === 0) {
    throw new QueryError('QUERY_FAILED', `${name} must be an absolute http(s) URL`);
  }
  if (url.username.length > 0 || url.password.length > 0) {
    throw new QueryError('QUERY_FAILED', `${name} must not include userinfo`);
  }
  if (url.search.length > 0 || url.hash.length > 0) {
    throw new QueryError('QUERY_FAILED', `${name} must not include query or fragment`);
  }
  if (url.protocol === 'http:' && !isLoopbackHostname(url.hostname)) {
    throw new QueryError('QUERY_FAILED', `${name} must use HTTPS unless it targets loopback`);
  }
}

function normalizeId(value: unknown, name: string): string {
  if (typeof value !== 'string') {
    throw new QueryError('QUERY_FAILED', `${name} must be a string`);
  }
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    throw new QueryError('QUERY_FAILED', `${name} must not be empty`);
  }
  if (trimmed !== value) {
    throw new QueryError('QUERY_FAILED', `${name} must not contain surrounding whitespace`);
  }
  return value;
}

function isLoopbackHostname(hostname: string): boolean {
  const normalized = hostname.toLowerCase().replace(/^\[/u, '').replace(/\]$/u, '');
  return (
    normalized === 'localhost' ||
    isIpv4LoopbackHostname(normalized) ||
    normalized === '::1' ||
    normalized === '0:0:0:0:0:0:0:1'
  );
}

function isIpv4LoopbackHostname(hostname: string): boolean {
  const parts = hostname.split('.');
  if (parts.length !== 4) {
    return false;
  }
  const octets = parts.map((part) => {
    if (!/^\d+$/u.test(part)) {
      return null;
    }
    const value = Number(part);
    return value >= 0 && value <= 255 ? value : null;
  });
  return octets.every((octet) => octet != null) && octets[0] === 127;
}

function isJsonObject(value: object): value is Record<string, unknown> {
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function redactSecret(message: string, secret: string): string {
  return secret.length === 0 ? message : message.split(secret).join('[redacted]');
}

function redactOptionalSecret(value: string | undefined, secret: string): string | undefined {
  return value == null ? undefined : redactSecret(value, secret);
}

function throwIfProviderAborted(signal: AbortSignal, timedOut: boolean): void {
  if (!signal.aborted) {
    return;
  }
  throw new QueryError('QUERY_FAILED', timedOut ? 'provider request timed out' : 'provider request aborted');
}

function emptyTokenEmissionStats(): TokenEmissionStats {
  return {
    framesSent: 0,
    bytesSent: 0,
    batchesSent: 0,
    drainMs: 0,
    drainCalls: 0,
  };
}
