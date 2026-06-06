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
import { createTimedAbortController } from '../utils/abort.js';

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
  done: boolean;
  usage?: GatewayUsage;
  sequence: number;
  stats: TokenEmissionStats;
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
const REMOTE_CONFIG_FIELDS = new Set([
  'kind',
  'alias',
  'baseUrl',
  'token',
  'tokenProvider',
  'timeoutMs',
]);
const MAX_GATEWAY_ERROR_BYTES = 1 << 20;
const MAX_GATEWAY_SSE_EVENT_BYTES = 1 << 20;
const GATEWAY_REQUEST_TIMEOUT_MESSAGE = 'remote gateway request timed out';
const GATEWAY_REQUEST_ABORTED_MESSAGE = 'remote gateway request aborted';
const GATEWAY_REQUEST_FAILED_MESSAGE = 'remote gateway request failed';
const GATEWAY_TOKEN_PROVIDER_FAILED_MESSAGE = 'remote gateway token provider failed';
const GATEWAY_ERROR_BODY_TOO_LARGE_MESSAGE =
  'remote gateway error response exceeded body limit';
const GATEWAY_SSE_EVENT_TOO_LARGE_MESSAGE =
  'gateway stream event exceeded buffer limit';
const U32_MAX = 0xffffffff;
const F32_MAX = 3.4028234663852886e38;
const UTF8_ENCODER = new TextEncoder();
const UTF8_DECODER = new TextDecoder();

/** Registry for browser remote gateway endpoints. */
export class RemoteGatewayRegistry {
  readonly #remotes = new Map<string, RemoteEndpoint>();

  public prepare(id: string, config: RemoteGatewayConfig): RemoteEndpoint {
    const normalizedId = normalizeId(id, 'remote id');
    return normalizeConfig(normalizedId, config);
  }

  public commit(remote: RemoteEndpoint): EndpointRef {
    this.#remotes.set(remote.id, remote);
    return { kind: 'remote', id: remote.id };
  }

  public remove(id: string): void {
    this.#remotes.delete(id);
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
  let prompt: string;
  let hasMedia = false;
  if (typeof input === 'string') {
    prompt = input;
  } else {
    const queryInput = objectInput(input, 'query input');
    prompt = requiredInputString(queryInput, 'prompt');
    hasMedia = queryInput.media != null && mediaHasEntries(queryInput.media);
  }
  if (prompt.trim().length === 0) {
    throw new QueryError('QUERY_FAILED', 'prompt must not be empty');
  }
  rejectRemoteTextLocalOptions(options, hasMedia);
  const body = textBody(
    remote.alias,
    options,
    options.gatewayOptions,
    { prompt },
    tokenBatchSink != null
  );
  return tokenBatchSink == null
    ? gatewayRequest(remote, '/v1/query', body, signal, async (response, token) =>
        parseTextResponse(await gatewayJsonBody(response, token))
      )
    : gatewayRequest(remote, '/v1/query', body, signal, (response, token, resetTimeout) =>
        readTextStream(response, token, tokenBatchSink, resetTimeout)
      );
}

/** Run a browser chat request through a CogentLM remote gateway. */
export async function runRemoteChat(
  remote: RemoteEndpoint,
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
  rejectRemoteTextLocalOptions(
    options,
    structuredInput?.media != null && mediaHasEntries(structuredInput.media)
  );
  const body = textBody(
    remote.alias,
    options,
    options.gatewayOptions,
    { messages: messages.map(gatewayMessage) },
    tokenBatchSink != null
  );
  return tokenBatchSink == null
    ? gatewayRequest(remote, '/v1/chat', body, signal, async (response, token) =>
        parseTextResponse(await gatewayJsonBody(response, token))
      )
    : gatewayRequest(remote, '/v1/chat', body, signal, (response, token, resetTimeout) =>
        readTextStream(response, token, tokenBatchSink, resetTimeout)
      );
}

/** Run a browser embedding request through a CogentLM remote gateway. */
export async function runRemoteEmbedding(
  remote: RemoteEndpoint,
  input: string,
  options: EmbedOptions,
  signal: AbortSignal
): Promise<EmbeddingResult> {
  const embedInput = requiredStandaloneString(input, 'input');
  if (embedInput.trim().length === 0) {
    throw new QueryError('QUERY_FAILED', 'input must not be empty');
  }
  rejectRemoteEmbedLocalOptions(options);
  const body = mergeGatewayOptions(
    {
      model: remote.alias,
      input: embedInput,
    },
    options.gatewayOptions,
    EMBED_TYPED_FIELDS
  );
  return gatewayRequest(remote, '/v1/embed', body, signal, async (response, token) =>
    parseEmbeddingResponse(await gatewayJsonBody(response, token))
  );
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
  const trimmedBaseUrl = config.baseUrl.trim();
  if (trimmedBaseUrl.length === 0) {
    throw new QueryError('QUERY_FAILED', 'remote gateway baseUrl must not be empty');
  }
  if (trimmedBaseUrl !== config.baseUrl) {
    throw new QueryError(
      'QUERY_FAILED',
      'remote gateway baseUrl must not contain surrounding whitespace'
    );
  }
  const baseUrl = config.baseUrl.replace(/\/+$/, '');
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
  if (config.token != null) {
    validateGatewayToken(config.token);
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
  } catch {
    throw new QueryError('QUERY_FAILED', 'remote gateway baseUrl is invalid');
  }
  if ((url.protocol !== 'http:' && url.protocol !== 'https:') || url.hostname.length === 0) {
    throw new QueryError('QUERY_FAILED', 'remote gateway baseUrl must be an absolute http(s) URL');
  }
  if (url.username.length > 0 || url.password.length > 0) {
    throw new QueryError('QUERY_FAILED', 'remote gateway baseUrl must not include userinfo');
  }
  if (url.search.length > 0 || url.hash.length > 0) {
    throw new QueryError(
      'QUERY_FAILED',
      'remote gateway baseUrl must not include query or fragment'
    );
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
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    throw new QueryError('QUERY_FAILED', `${name} must not be empty`);
  }
  if (trimmed !== value) {
    throw new QueryError('QUERY_FAILED', `${name} must not contain surrounding whitespace`);
  }
  return value;
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
  gatewayOptions: unknown,
  payload: { readonly prompt: string } | { readonly messages: readonly unknown[] },
  stream: boolean
): Record<string, unknown> {
  const maxTokens = optionalPositiveU32(options.maxTokens, 'max_tokens');
  const temperature = optionalTemperature(options.temperature);
  const topP = optionalTopP(options.topP);
  const stop = optionalStringArray(options.stop, 'stop');
  return mergeGatewayOptions(
    {
      model: alias,
      ...payload,
      ...(maxTokens == null ? {} : { max_tokens: maxTokens }),
      ...(temperature == null ? {} : { temperature }),
      ...(topP == null ? {} : { top_p: topP }),
      ...(stop == null ? {} : { stop }),
      stream,
    },
    gatewayOptions,
    TEXT_TYPED_FIELDS
  );
}

function optionalPositiveU32(value: unknown, key: string): number | undefined {
  if (value == null) {
    return undefined;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0 || value > U32_MAX) {
    throw new QueryError('QUERY_FAILED', `${key} must be a u32 integer`);
  }
  if (value === 0) {
    throw new QueryError('QUERY_FAILED', `${key} must be greater than zero`);
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

function mergeGatewayOptions(
  body: Record<string, unknown>,
  gatewayOptions: unknown,
  typedFields: ReadonlySet<string>
): Record<string, unknown> {
  if (gatewayOptions == null) {
    return body;
  }
  if (typeof gatewayOptions !== 'object' || Array.isArray(gatewayOptions)) {
    throw new QueryError('QUERY_FAILED', 'gatewayOptions must be a JSON object');
  }
  if (!isJsonObject(gatewayOptions)) {
    throw new QueryError('QUERY_FAILED', 'gatewayOptions must be a JSON object');
  }
  for (const [key, value] of Object.entries(gatewayOptions)) {
    if (typedFields.has(key)) {
      throw new QueryError('QUERY_FAILED', `gatewayOptions cannot override typed field: ${key}`);
    }
    if (LOCAL_ONLY_GATEWAY_FIELDS.has(key)) {
      throw new QueryError('QUERY_FAILED', `gatewayOptions cannot contain local-only field: ${key}`);
    }
    body[key] = snapshotJsonCompatibleGatewayOption(value);
  }
  return body;
}

function snapshotJsonCompatibleGatewayOption(value: unknown): unknown {
  return snapshotJsonCompatibleValue(value, new WeakSet<object>());
}

function snapshotJsonCompatibleValue(value: unknown, ancestors: WeakSet<object>): unknown {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new QueryError('QUERY_FAILED', 'gatewayOptions cannot contain non-finite numbers');
    }
    return value;
  }
  if (typeof value !== 'object') {
    throw new QueryError('QUERY_FAILED', 'gatewayOptions must contain JSON-compatible values');
  }

  if (ancestors.has(value)) {
    throw new QueryError('QUERY_FAILED', 'gatewayOptions must contain JSON-compatible values');
  }
  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      return value.map((item) => snapshotJsonCompatibleValue(item, ancestors));
    }
    if (!isJsonObject(value)) {
      throw new QueryError('QUERY_FAILED', 'gatewayOptions must contain JSON-compatible values');
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

function isJsonObject(value: object): value is Record<string, unknown> {
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function gatewayMessage(message: ChatMessage): Record<string, string> {
  return {
    role: message.role,
    content: message.content,
  };
}

function objectInput(value: unknown, name: string): Record<string, unknown> {
  if (typeof value !== 'object' || value == null || Array.isArray(value)) {
    throw new QueryError('QUERY_FAILED', `${name} must be a JSON object`);
  }
  return value as Record<string, unknown>;
}

function requiredInputString(body: Record<string, unknown>, key: string): string {
  return requiredStandaloneString(body[key], key);
}

function requiredStandaloneString(value: unknown, key: string): string {
  if (typeof value !== 'string') {
    throw new QueryError('QUERY_FAILED', `${key} must be a string`);
  }
  return value;
}

function mediaHasEntries(value: unknown): boolean {
  return Array.isArray(value) ? value.length > 0 : true;
}

function chatMessagesFromInput(
  input: ChatInput,
  structuredInput: Record<string, unknown> | null
): readonly ChatMessage[] {
  const rawMessages = Array.isArray(input) ? input : structuredInput?.messages;
  if (!Array.isArray(rawMessages)) {
    throw new QueryError('QUERY_FAILED', 'messages must be an array');
  }
  return rawMessages.map(chatMessageFromValue);
}

function chatMessageFromValue(value: unknown): ChatMessage {
  const message = objectInput(value, 'message');
  const role = message.role;
  if (role !== 'system' && role !== 'user' && role !== 'assistant') {
    throw new QueryError('QUERY_FAILED', 'message role must be system, user, or assistant');
  }
  const content = message.content;
  if (typeof content !== 'string') {
    throw new QueryError('QUERY_FAILED', 'message content must be a string');
  }
  return { role, content };
}

async function gatewayRequest<T>(
  remote: RemoteEndpoint,
  path: string,
  body: Record<string, unknown>,
  signal: AbortSignal,
  read: (response: Response, token: string, resetTimeout: () => void) => Promise<T>
): Promise<T> {
  const abort = createTimedAbortController(signal, remote.timeoutMs);

  try {
    throwIfGatewayAborted(abort.signal, abort.timedOut());
    const token = await remoteToken(remote, abort.signal);
    throwIfGatewayAborted(abort.signal, abort.timedOut());
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
      signal: abort.signal,
    });
    if (!response.ok) {
      throw await gatewayError(response, token);
    }
    return await read(response, token, abort.resetTimeout);
  } catch (error) {
    if (abort.timedOut()) {
      throw new QueryError('QUERY_FAILED', GATEWAY_REQUEST_TIMEOUT_MESSAGE);
    }
    if (abort.signal.aborted) {
      throw new QueryError('QUERY_FAILED', GATEWAY_REQUEST_ABORTED_MESSAGE);
    }
    if (error instanceof QueryError) {
      throw error;
    }
    throw new QueryError('QUERY_FAILED', GATEWAY_REQUEST_FAILED_MESSAGE);
  } finally {
    abort.dispose();
  }
}

async function remoteToken(remote: RemoteEndpoint, signal: AbortSignal): Promise<string> {
  let token: unknown;
  try {
    token = remote.token ?? (await remoteProviderToken(remote.tokenProvider, signal));
  } catch {
    if (signal.aborted) {
      throw new Error('remote gateway token provider aborted');
    }
    throw new QueryError('QUERY_FAILED', GATEWAY_TOKEN_PROVIDER_FAILED_MESSAGE);
  }
  if (typeof token !== 'string') {
    throw new QueryError('QUERY_FAILED', 'remote gateway token must be a string');
  }
  validateGatewayToken(token);
  return token;
}

async function remoteProviderToken(
  tokenProvider: RemoteEndpoint['tokenProvider'],
  signal: AbortSignal
): Promise<unknown> {
  if (tokenProvider == null) {
    return undefined;
  }
  if (signal.aborted) {
    throw new Error('remote gateway token provider aborted');
  }

  let removeAbortListener = (): void => {};
  const abortPromise = new Promise<never>((_resolve, reject) => {
    const abortListener = (): void => {
      reject(new Error('remote gateway token provider aborted'));
    };
    signal.addEventListener('abort', abortListener, { once: true });
    removeAbortListener = () => {
      signal.removeEventListener('abort', abortListener);
    };
  });

  try {
    const providerPromise = Promise.resolve().then(() => tokenProvider());
    return await Promise.race([providerPromise, abortPromise]);
  } finally {
    removeAbortListener();
  }
}

function validateGatewayToken(token: string): void {
  if (token.trim().length === 0) {
    throw new QueryError('QUERY_FAILED', 'remote gateway token must not be empty');
  }
  if (/\s/u.test(token)) {
    throw new QueryError('QUERY_FAILED', 'remote gateway token must not contain whitespace');
  }
}

function throwIfGatewayAborted(signal: AbortSignal, timedOut: boolean): void {
  if (!signal.aborted) {
    return;
  }
  throw new QueryError(
    'QUERY_FAILED',
    timedOut ? GATEWAY_REQUEST_TIMEOUT_MESSAGE : GATEWAY_REQUEST_ABORTED_MESSAGE
  );
}

async function gatewayError(response: Response, token: string): Promise<QueryError> {
  const body = await gatewayErrorBody(response);
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

async function gatewayErrorBody(response: Response): Promise<unknown> {
  const text = await responseTextWithinLimit(response);
  if (text == null) {
    return { error: { message: GATEWAY_ERROR_BODY_TOO_LARGE_MESSAGE } };
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
    return UTF8_ENCODER.encode(text).byteLength > MAX_GATEWAY_ERROR_BYTES ? null : text;
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
    if (totalBytes > MAX_GATEWAY_ERROR_BYTES) {
      try {
        await reader.cancel();
      } catch {
        // Preserve the bounded-error result even if the stream source rejects cancellation.
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
  const error = (body as { readonly error?: { readonly code?: unknown; readonly type?: unknown } })
    .error;
  if (typeof error?.code === 'string') {
    return error.code;
  }
  return typeof error?.type === 'string' ? error.type : undefined;
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

async function gatewayJsonBody(
  response: Response,
  token: string
): Promise<Record<string, unknown>> {
  const body = objectValue(await response.json());
  rejectGatewayBodyError(body, token, gatewayRequestId(response.headers));
  return body;
}

function rejectGatewayBodyError(
  body: Record<string, unknown>,
  token: string,
  requestId: string | undefined
): void {
  if (body.error == null) {
    return;
  }
  throw new QueryError('QUERY_FAILED', redactSecret(gatewayBodyErrorMessage(body), token), {
    gatewayCode: redactOptionalSecret(gatewayErrorCode(body), token),
    requestId: redactOptionalSecret(requestId, token),
  });
}

function gatewayBodyErrorMessage(body: Record<string, unknown>): string {
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
  requiredStringField(body, 'model');
  const text = requiredStringField(body, 'text');
  const finishReasonRaw = requiredStringField(body, 'finish_reason');
  const usage = usageFromValue(body.usage);
  return {
    id: stringField(body, 'id', 'gw_text'),
    text,
    finishReason: finishReason(finishReasonRaw),
    stats: requestStats(usage),
  };
}

function parseEmbeddingResponse(value: unknown): EmbeddingResult {
  const body = objectValue(value);
  requiredStringField(body, 'model');
  const values = numericArrayField(body, 'embedding');
  const usage = usageFromValue(body.usage);
  return {
    id: stringField(body, 'id', 'gw_embed'),
    values,
    pooling: 'none',
    normalized: false,
    stats: requestStats(usage),
  };
}

async function readTextStream(
  response: Response,
  token: string,
  tokenBatchSink: (batch: TokenBatch) => void,
  resetTimeout: () => void
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
    done: false,
    sequence: 0,
    stats: emptyTokenEmissionStats(),
  };
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
      pushStreamEvent(state, raw);
      boundary = eventBoundary(buffer);
    }
    assertStreamEventWithinLimit(buffer);
  }
  if (buffer.trim().length > 0) {
    assertStreamEventWithinLimit(buffer);
    pushStreamEvent(state, buffer);
  }
  if (!state.done) {
    throw new QueryError('QUERY_FAILED', 'gateway stream ended before done event');
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
  if (state.done) {
    throw new QueryError('QUERY_FAILED', 'gateway stream event received after done event');
  }
  if (event.event === 'token') {
    const data = parseStreamJson(event.data);
    const text = requiredStringField(data, 'text');
    const sequence = optionalStreamSequence(data);
    state.text += text;
    const batch = tokenBatch(state, text, sequence);
    state.tokenBatchSink?.(batch);
  } else if (event.event === 'usage') {
    const data = parseStreamJson(event.data);
    state.usage = usageFromValue(data);
  } else if (event.event === 'done') {
    const data = parseStreamJson(event.data);
    state.finishReason = finishReason(requiredStringField(data, 'finish_reason', 'gateway stream done event'));
    state.done = true;
  } else if (event.event === 'error') {
    const data = parseStreamJson(event.data);
    throw new QueryError('QUERY_FAILED', redactSecret(streamErrorMessage(data), state.token), {
      gatewayCode: redactOptionalSecret(gatewayErrorCode(data), state.token),
      requestId: redactOptionalSecret(state.requestId || undefined, state.token),
    });
  } else {
    throw new QueryError(
      'QUERY_FAILED',
      redactSecret(`unsupported gateway stream event: ${event.event}`, state.token)
    );
  }
}

function parseStreamJson(payload: string): Record<string, unknown> {
  let value: unknown;
  try {
    value = JSON.parse(payload) as unknown;
  } catch {
    throw new QueryError('QUERY_FAILED', 'invalid gateway stream JSON payload');
  }
  if (typeof value !== 'object' || value == null || Array.isArray(value)) {
    throw new QueryError('QUERY_FAILED', 'gateway stream payload must be a JSON object');
  }
  return value as Record<string, unknown>;
}

function assertStreamEventWithinLimit(raw: string): void {
  if (UTF8_ENCODER.encode(raw).byteLength > MAX_GATEWAY_SSE_EVENT_BYTES) {
    throw new QueryError('QUERY_FAILED', GATEWAY_SSE_EVENT_TOO_LARGE_MESSAGE);
  }
}

function streamErrorMessage(body: Record<string, unknown>): string {
  return gatewayBodyErrorMessage(body);
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
  if (value == null) {
    return undefined;
  }
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new QueryError('QUERY_FAILED', 'usage must be a JSON object');
  }
  const body = value as Record<string, unknown>;
  return {
    input_tokens: optionalUsageU32(body, 'input_tokens'),
    output_tokens: optionalUsageU32(body, 'output_tokens'),
    total_tokens: optionalUsageU32(body, 'total_tokens'),
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
  if (typeof value !== 'object' || value == null || Array.isArray(value)) {
    throw new QueryError('QUERY_FAILED', 'remote gateway response must be a JSON object');
  }
  return value as Record<string, unknown>;
}

function stringField(body: Record<string, unknown>, key: string, fallback: string): string {
  const value = body[key];
  return typeof value === 'string' ? value : fallback;
}

function requiredStringField(
  body: Record<string, unknown>,
  key: string,
  context = 'remote gateway response'
): string {
  const value = body[key];
  if (typeof value !== 'string') {
    throw new QueryError('QUERY_FAILED', `${context} missing ${key}`);
  }
  return value;
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
    if (item < -F32_MAX || item > F32_MAX) {
      throw new QueryError(
        'QUERY_FAILED',
        `remote gateway ${key} contains value outside f32 range`
      );
    }
    return item;
  });
}

function optionalUsageU32(body: Record<string, unknown>, key: string): number | undefined {
  if (!Object.prototype.hasOwnProperty.call(body, key)) {
    return undefined;
  }
  const value = body[key];
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new QueryError('QUERY_FAILED', `usage field is not a number: ${key}`);
  }
  if (value > U32_MAX) {
    throw new QueryError('QUERY_FAILED', `usage field exceeds u32: ${key}`);
  }
  return value;
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
