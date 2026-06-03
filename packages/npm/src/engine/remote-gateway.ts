import type { ChatMessage, TokenBatch, TokenEmissionStats } from './inference-types.js';
import {
  QueryError,
  type ChatInput,
  type EmbedOptions,
  type EmbeddingResult,
  type EndpointRef,
  type FinishReason,
  type GenerationResult,
  type QueryInput,
  type QueryOptions,
  type RequestStats,
  type RemoteGatewayConfig,
} from '../models/types.js';

/** Normalized remote gateway endpoint stored by the browser client. */
export interface RemoteEndpoint {
  readonly id: string;
  readonly alias: string;
  readonly baseUrl: string;
  readonly token?: string;
  readonly tokenProvider?: () => string | Promise<string>;
  readonly timeoutMs?: number;
}

interface GatewayUsage {
  readonly input_tokens?: number;
  readonly output_tokens?: number;
  readonly total_tokens?: number;
}

interface TextStreamState {
  readonly requestId: string;
  readonly token: string;
  readonly tokenBatchSink?: (batch: TokenBatch) => void;
  text: string;
  finishReason: FinishReason;
  usage?: GatewayUsage;
  sequence: number;
  stats: TokenEmissionStats;
}

interface GatewayFetchResult {
  readonly response: Response;
  readonly token: string;
}

const TEXT_TYPED_FIELDS = new Set([
  'model',
  'prompt',
  'messages',
  'max_tokens',
  'temperature',
  'top_p',
  'stop',
  'stream',
]);
const EMBED_TYPED_FIELDS = new Set(['model', 'input']);
const LOCAL_ONLY_GATEWAY_FIELDS = new Set([
  'context_key',
  'contextKey',
  'session',
  'grammar',
  'json_schema',
  'jsonSchema',
  'sampling',
  'media',
  'normalize',
  'local',
]);
const REMOTE_CONFIG_FIELDS = new Set(['alias', 'baseUrl', 'token', 'tokenProvider', 'timeoutMs']);
const MAX_GATEWAY_SSE_EVENT_BYTES = 1 << 20;
const GATEWAY_REQUEST_TIMEOUT_MESSAGE = 'remote gateway request timed out';
const GATEWAY_REQUEST_FAILED_MESSAGE = 'remote gateway request failed';
const GATEWAY_TOKEN_PROVIDER_FAILED_MESSAGE = 'remote gateway token provider failed';
const GATEWAY_SSE_EVENT_TOO_LARGE_MESSAGE =
  'gateway stream event exceeded buffer limit';

/** Registry for browser remote gateway endpoints. */
export class RemoteGatewayRegistry {
  readonly #remotes = new Map<string, RemoteEndpoint>();

  public add(id: string, config: RemoteGatewayConfig): EndpointRef {
    const normalizedId = normalizeId(id, 'remote id');
    if (this.#remotes.has(normalizedId)) {
      throw new QueryError('QUERY_FAILED', 'remote endpoint already registered');
    }
    this.#remotes.set(normalizedId, normalizeConfig(normalizedId, config));
    return { kind: 'remote', id: normalizedId };
  }

  public update(id: string, config: RemoteGatewayConfig): EndpointRef {
    const normalizedId = normalizeId(id, 'remote id');
    if (!this.#remotes.has(normalizedId)) {
      throw new QueryError('MODEL_NOT_FOUND', `remote endpoint not found: ${normalizedId}`);
    }
    this.#remotes.set(normalizedId, normalizeConfig(normalizedId, config));
    return { kind: 'remote', id: normalizedId };
  }

  public get(endpoint: EndpointRef | undefined): RemoteEndpoint | null {
    if (endpoint == null || endpoint.kind !== 'remote') {
      return null;
    }
    const remote = this.#remotes.get(endpoint.id);
    if (remote == null) {
      throw new QueryError('MODEL_NOT_FOUND', `remote endpoint not found: ${endpoint.id}`);
    }
    return remote;
  }
}

/** Run a browser query request through a CogentLM remote gateway. */
export async function runRemoteQuery(
  remote: RemoteEndpoint,
  input: QueryInput,
  options: QueryOptions,
  tokenBatchSink: ((batch: TokenBatch) => void) | undefined,
  signal: AbortSignal
): Promise<GenerationResult> {
  const prompt = typeof input === 'string' ? input : input.prompt;
  rejectRemoteTextLocalOptions(
    options,
    typeof input !== 'string' && input.media != null && input.media.length > 0
  );
  const body = textBody(
    remote.alias,
    options,
    options.gatewayOptions,
    { prompt },
    tokenBatchSink != null
  );
  const result = await gatewayFetch(remote, '/v1/query', body, signal);
  return tokenBatchSink == null
    ? parseTextResponse(await result.response.json())
    : readTextStream(result.response, result.token, tokenBatchSink);
}

/** Run a browser chat request through a CogentLM remote gateway. */
export async function runRemoteChat(
  remote: RemoteEndpoint,
  input: ChatInput,
  options: QueryOptions,
  tokenBatchSink: ((batch: TokenBatch) => void) | undefined,
  signal: AbortSignal
): Promise<GenerationResult> {
  const structuredInput = isChatMessageArray(input) ? null : input;
  const messages: readonly ChatMessage[] = isChatMessageArray(input) ? input : input.messages;
  rejectRemoteTextLocalOptions(
    options,
    structuredInput?.media != null && structuredInput.media.length > 0
  );
  const body = textBody(
    remote.alias,
    options,
    options.gatewayOptions,
    { messages: messages.map(gatewayMessage) },
    tokenBatchSink != null
  );
  const result = await gatewayFetch(remote, '/v1/chat', body, signal);
  return tokenBatchSink == null
    ? parseTextResponse(await result.response.json())
    : readTextStream(result.response, result.token, tokenBatchSink);
}

/** Run a browser embedding request through a CogentLM remote gateway. */
export async function runRemoteEmbedding(
  remote: RemoteEndpoint,
  input: string,
  options: EmbedOptions,
  signal: AbortSignal
): Promise<EmbeddingResult> {
  rejectRemoteEmbedLocalOptions(options);
  const body = mergeGatewayOptions(
    {
      model: remote.alias,
      input,
    },
    options.gatewayOptions,
    EMBED_TYPED_FIELDS
  );
  const result = await gatewayFetch(remote, '/v1/embed', body, signal);
  const value = await result.response.json();
  return {
    id: stringField(value, 'id', 'gw_embed'),
    values: numericArrayField(value, 'embedding'),
    pooling: 'none',
    normalized: false,
    stats: requestStats(usageFromValue(value.usage)),
  };
}

function normalizeConfig(id: string, config: RemoteGatewayConfig): RemoteEndpoint {
  if (typeof config !== 'object' || config == null || Array.isArray(config)) {
    throw new QueryError('QUERY_FAILED', 'remote gateway config must be an object');
  }
  rejectUnknownRemoteConfigFields(config);
  const alias = normalizeId(config.alias, 'remote alias');
  if (typeof config.baseUrl !== 'string') {
    throw new QueryError('QUERY_FAILED', 'remote gateway baseUrl must be a string');
  }
  const baseUrl = config.baseUrl.trim().replace(/\/+$/, '');
  if (baseUrl.length === 0) {
    throw new QueryError('QUERY_FAILED', 'remote gateway baseUrl must not be empty');
  }
  validateGatewayBaseUrl(baseUrl);
  if (config.token == null && config.tokenProvider == null) {
    throw new QueryError('QUERY_FAILED', 'remote gateway token or tokenProvider is required');
  }
  if (config.token != null && config.tokenProvider != null) {
    throw new QueryError('QUERY_FAILED', 'remote gateway must set token or tokenProvider, not both');
  }
  if (config.token != null && typeof config.token !== 'string') {
    throw new QueryError('QUERY_FAILED', 'remote gateway token must be a string');
  }
  if (config.token != null && config.token.length === 0) {
    throw new QueryError('QUERY_FAILED', 'remote gateway token must not be empty');
  }
  if (config.tokenProvider != null && typeof config.tokenProvider !== 'function') {
    throw new QueryError('QUERY_FAILED', 'remote gateway tokenProvider must be a function');
  }
  if (
    config.timeoutMs != null &&
    (typeof config.timeoutMs !== 'number' ||
      !Number.isFinite(config.timeoutMs) ||
      config.timeoutMs <= 0)
  ) {
    throw new QueryError('QUERY_FAILED', 'remote gateway timeoutMs must be positive');
  }
  return {
    id,
    alias,
    baseUrl,
    token: config.token,
    tokenProvider: config.tokenProvider,
    timeoutMs: config.timeoutMs,
  };
}

function rejectUnknownRemoteConfigFields(config: RemoteGatewayConfig): void {
  for (const field of Object.keys(config)) {
    if (!REMOTE_CONFIG_FIELDS.has(field)) {
      throw new QueryError('QUERY_FAILED', `unsupported remote gateway config field: ${field}`);
    }
  }
}

function validateGatewayBaseUrl(baseUrl: string): void {
  let url: URL;
  try {
    url = new URL(baseUrl);
  } catch (error) {
    throw new QueryError('QUERY_FAILED', 'remote gateway baseUrl is invalid', { cause: error });
  }
  if ((url.protocol !== 'http:' && url.protocol !== 'https:') || url.hostname.length === 0) {
    throw new QueryError('QUERY_FAILED', 'remote gateway baseUrl must be an absolute http(s) URL');
  }
  if (url.username.length > 0 || url.password.length > 0) {
    throw new QueryError('QUERY_FAILED', 'remote gateway baseUrl must not include userinfo');
  }
  if (url.protocol === 'http:' && !isLoopbackHostname(url.hostname)) {
    throw new QueryError(
      'QUERY_FAILED',
      'remote gateway baseUrl must use HTTPS unless it targets loopback'
    );
  }
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

function normalizeId(value: unknown, name: string): string {
  if (typeof value !== 'string') {
    throw new QueryError('QUERY_FAILED', `${name} must be a string`);
  }
  const normalized = value.trim();
  if (normalized.length === 0) {
    throw new QueryError('QUERY_FAILED', `${name} must not be empty`);
  }
  return normalized;
}

function rejectRemoteTextLocalOptions(options: QueryOptions, hasMedia: boolean): void {
  if (options.session != null || options.grammar != null || hasMedia) {
    throw new QueryError('UNSUPPORTED_OPERATION', 'local text options are not valid for remote endpoints');
  }
}

function rejectRemoteEmbedLocalOptions(options: EmbedOptions): void {
  if (options.contextKey != null || options.normalize != null) {
    throw new QueryError('UNSUPPORTED_OPERATION', 'local embed options are not valid for remote endpoints');
  }
}

function textBody(
  alias: string,
  options: QueryOptions,
  gatewayOptions: Record<string, unknown> | undefined,
  payload: { readonly prompt: string } | { readonly messages: readonly unknown[] },
  stream: boolean
): Record<string, unknown> {
  return mergeGatewayOptions(
    {
      model: alias,
      ...payload,
      ...(options.maxTokens == null ? {} : { max_tokens: options.maxTokens }),
      ...(options.temperature == null ? {} : { temperature: options.temperature }),
      ...(options.topP == null ? {} : { top_p: options.topP }),
      ...(options.stop == null ? {} : { stop: [...options.stop] }),
      stream,
    },
    gatewayOptions,
    TEXT_TYPED_FIELDS
  );
}

function mergeGatewayOptions(
  body: Record<string, unknown>,
  gatewayOptions: Record<string, unknown> | undefined,
  typedFields: ReadonlySet<string>
): Record<string, unknown> {
  if (gatewayOptions == null) {
    return body;
  }
  for (const [key, value] of Object.entries(gatewayOptions)) {
    if (typedFields.has(key)) {
      throw new QueryError('QUERY_FAILED', `gatewayOptions cannot override typed field: ${key}`);
    }
    if (LOCAL_ONLY_GATEWAY_FIELDS.has(key)) {
      throw new QueryError('QUERY_FAILED', `gatewayOptions cannot contain local-only field: ${key}`);
    }
    body[key] = value;
  }
  return body;
}

function gatewayMessage(message: ChatMessage): Record<string, string> {
  return {
    role: message.role,
    content: message.content,
  };
}

function isChatMessageArray(input: ChatInput): input is readonly ChatMessage[] {
  return Array.isArray(input);
}

async function gatewayFetch(
  remote: RemoteEndpoint,
  path: string,
  body: Record<string, unknown>,
  signal: AbortSignal
): Promise<GatewayFetchResult> {
  const controller = new AbortController();
  const abort = (): void => controller.abort(signal.reason);
  signal.addEventListener('abort', abort, { once: true });
  const timeout =
    remote.timeoutMs == null
      ? undefined
      : setTimeout(() => controller.abort(new Error(GATEWAY_REQUEST_TIMEOUT_MESSAGE)), remote.timeoutMs);

  try {
    const token = await remoteToken(remote);
    const response = await fetch(`${remote.baseUrl}${path}`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      credentials: 'omit',
      mode: 'cors',
      redirect: 'error',
      signal: controller.signal,
    });
    if (!response.ok) {
      throw await gatewayError(response, token);
    }
    return { response, token };
  } catch (error) {
    if (error instanceof QueryError) {
      throw error;
    }
    if (timeout != null && timeoutAbort(controller.signal)) {
      throw new QueryError('QUERY_FAILED', GATEWAY_REQUEST_TIMEOUT_MESSAGE);
    }
    throw new QueryError('QUERY_FAILED', GATEWAY_REQUEST_FAILED_MESSAGE);
  } finally {
    signal.removeEventListener('abort', abort);
    if (timeout != null) {
      clearTimeout(timeout);
    }
  }
}

async function remoteToken(remote: RemoteEndpoint): Promise<string> {
  let token: unknown;
  try {
    token = remote.token ?? (await remote.tokenProvider?.());
  } catch {
    throw new QueryError('QUERY_FAILED', GATEWAY_TOKEN_PROVIDER_FAILED_MESSAGE);
  }
  if (typeof token !== 'string') {
    throw new QueryError('QUERY_FAILED', 'remote gateway token must be a string');
  }
  if (token == null || token.length === 0) {
    throw new QueryError('QUERY_FAILED', 'remote gateway token must not be empty');
  }
  return token;
}

function timeoutAbort(signal: AbortSignal): boolean {
  return (
    signal.aborted &&
    signal.reason instanceof Error &&
    signal.reason.message === GATEWAY_REQUEST_TIMEOUT_MESSAGE
  );
}

async function gatewayError(response: Response, token: string): Promise<QueryError> {
  const text = await response.text();
  let body: unknown = text;
  try {
    body = JSON.parse(text) as unknown;
  } catch {
    body = text;
  }
  const message =
    typeof body === 'object' && body != null && 'error' in body
      ? errorMessage((body as { readonly error?: { readonly message?: unknown } }).error?.message)
      : response.statusText || 'remote gateway error';
  return new QueryError('QUERY_FAILED', redactSecret(message, token), {
    status: response.status,
    gatewayCode: redactOptionalSecret(gatewayErrorCode(body), token),
    requestId: redactOptionalSecret(gatewayRequestId(response.headers), token),
    retryAfterMs: retryAfterMs(response.headers),
  });
}

function redactSecret(message: string, secret: string): string {
  return secret.length === 0 ? message : message.split(secret).join('[redacted]');
}

function redactOptionalSecret(value: string | undefined, secret: string): string | undefined {
  return value == null ? undefined : redactSecret(value, secret);
}

function gatewayErrorCode(body: unknown): string | undefined {
  if (typeof body !== 'object' || body == null || !('error' in body)) {
    return undefined;
  }
  const error = (body as { readonly error?: { readonly code?: unknown } }).error;
  return typeof error?.code === 'string' ? error.code : undefined;
}

function gatewayRequestId(headers: Headers): string | undefined {
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

function parseTextResponse(value: unknown): GenerationResult {
  const body = objectValue(value);
  const usage = usageFromValue(body.usage);
  return {
    id: stringField(body, 'id', 'gw_text'),
    text: stringField(body, 'text', ''),
    finishReason: finishReason(body.finish_reason),
    stats: requestStats(usage),
  };
}

async function readTextStream(
  response: Response,
  token: string,
  tokenBatchSink: (batch: TokenBatch) => void
): Promise<GenerationResult> {
  if (response.body == null) {
    throw new QueryError('STREAMING_UNAVAILABLE', 'remote gateway response body is not streamable');
  }
  const state: TextStreamState = {
    requestId: response.headers.get('x-request-id') ?? '',
    token,
    tokenBatchSink,
    text: '',
    finishReason: 'stop',
    sequence: 0,
    stats: emptyTokenEmissionStats(),
  };
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';

  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += value;
    let boundary = eventBoundary(buffer);
    while (boundary != null) {
      const raw = buffer.slice(0, boundary.index);
      buffer = buffer.slice(boundary.index + boundary.length);
      assertStreamEventWithinLimit(raw);
      pushStreamEvent(state, raw);
      boundary = eventBoundary(buffer);
    }
    assertStreamEventWithinLimit(buffer);
  }
  if (buffer.trim().length > 0) {
    assertStreamEventWithinLimit(buffer);
    pushStreamEvent(state, buffer);
  }

  return {
    id: state.requestId || 'gw_stream',
    text: state.text,
    finishReason: state.finishReason,
    stats: requestStats(state.usage),
  };
}

function pushStreamEvent(state: TextStreamState, raw: string): void {
  const event = parseSseEvent(raw);
  if (event == null || event.data === '[DONE]') {
    return;
  }
  const data = parseStreamJson(event.data);
  if (event.event === 'token') {
    const text = stringField(data, 'text', '');
    const sequence = optionalStreamSequence(data);
    state.text += text;
    const batch = tokenBatch(state, text, sequence);
    state.tokenBatchSink?.(batch);
  } else if (event.event === 'usage') {
    state.usage = usageFromValue(data);
  } else if (event.event === 'done') {
    state.finishReason = finishReason(data.finish_reason);
  } else if (event.event === 'error') {
    throw new QueryError('QUERY_FAILED', redactSecret(streamErrorMessage(data), state.token), {
      gatewayCode: redactOptionalSecret(gatewayErrorCode(data), state.token),
      requestId: redactOptionalSecret(state.requestId || undefined, state.token),
    });
  } else {
    throw new QueryError('QUERY_FAILED', `unsupported gateway stream event: ${event.event}`);
  }
}

function parseStreamJson(payload: string): Record<string, unknown> {
  let value: unknown;
  try {
    value = JSON.parse(payload) as unknown;
  } catch (error) {
    throw new QueryError('QUERY_FAILED', 'invalid gateway stream JSON payload', { cause: error });
  }
  if (typeof value !== 'object' || value == null) {
    throw new QueryError('QUERY_FAILED', 'gateway stream payload must be a JSON object');
  }
  return value as Record<string, unknown>;
}

function assertStreamEventWithinLimit(raw: string): void {
  if (new TextEncoder().encode(raw).byteLength > MAX_GATEWAY_SSE_EVENT_BYTES) {
    throw new QueryError('QUERY_FAILED', GATEWAY_SSE_EVENT_TOO_LARGE_MESSAGE);
  }
}

function streamErrorMessage(body: Record<string, unknown>): string {
  if (typeof body.message === 'string' && body.message.length > 0) {
    return body.message;
  }
  const error = body.error;
  if (typeof error === 'object' && error != null) {
    const message = (error as { readonly message?: unknown }).message;
    if (typeof message === 'string' && message.length > 0) {
      return message;
    }
  }
  return errorMessage(error);
}

function optionalStreamSequence(body: Record<string, unknown>): number | undefined {
  const raw = body.sequence;
  if (raw == null) {
    return undefined;
  }
  if (
    typeof raw !== 'number' ||
    !Number.isInteger(raw) ||
    raw < 0 ||
    raw > 0xffffffff
  ) {
    throw new QueryError('QUERY_FAILED', 'gateway stream sequence must be a u32 integer');
  }
  return raw;
}

function tokenBatch(
  state: TextStreamState,
  text: string,
  sequenceStart = state.sequence
): TokenBatch {
  const byteCount = new TextEncoder().encode(text).byteLength;
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
    sequenceStart,
    text,
    frameCount: 1,
    byteCount,
    stats: state.stats,
  };
  state.sequence = sequenceStart + 1;
  return batch;
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

function parseSseEvent(raw: string): { readonly event: string; readonly data: string } | null {
  let event = 'message';
  const data: string[] = [];
  for (const line of raw.split(/\r?\n/u)) {
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trimStart();
    } else if (line.startsWith('data:')) {
      data.push(line.slice('data:'.length).trimStart());
    }
  }
  return data.length === 0 ? null : { event, data: data.join('\n') };
}

function usageFromValue(value: unknown): GatewayUsage | undefined {
  if (typeof value !== 'object' || value == null) {
    return undefined;
  }
  const body = value as Record<string, unknown>;
  return {
    input_tokens: optionalNumber(body.input_tokens),
    output_tokens: optionalNumber(body.output_tokens),
    total_tokens: optionalNumber(body.total_tokens),
  };
}

function requestStats(usage: GatewayUsage | undefined): RequestStats {
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

function objectValue(value: unknown): Record<string, unknown> {
  if (typeof value !== 'object' || value == null) {
    throw new QueryError('QUERY_FAILED', 'remote gateway response must be a JSON object');
  }
  return value as Record<string, unknown>;
}

function stringField(body: Record<string, unknown>, key: string, fallback: string): string {
  const value = body[key];
  return typeof value === 'string' ? value : fallback;
}

function numericArrayField(body: Record<string, unknown>, key: string): number[] {
  const value = body[key];
  if (!Array.isArray(value)) {
    throw new QueryError('QUERY_FAILED', `remote gateway response missing ${key}`);
  }
  return value.map((item) => {
    if (typeof item !== 'number' || !Number.isFinite(item)) {
      throw new QueryError('QUERY_FAILED', `remote gateway ${key} contains non-finite value`);
    }
    return item;
  });
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function finishReason(value: unknown): FinishReason {
  return value === 'length' || value === 'max_tokens' || value === 'max_output_tokens'
    ? 'length'
    : 'stop';
}

function errorMessage(value: unknown): string {
  return typeof value === 'string' && value.length > 0 ? value : 'remote gateway error';
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
