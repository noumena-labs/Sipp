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
  type GatewayEndpointDescriptor,
} from '../models/types.js';
import { createTimedAbortController } from '../utils/abort.js';

/** Normalized gateway endpoint stored by the browser client. */
export interface GatewayEndpoint {
  readonly id: string;
  readonly target: string;
  readonly baseUrl: string;
  readonly routes: {
    readonly query: string;
    readonly chat: string;
    readonly embed: string;
  };
  readonly authentication: {
    readonly kind: 'none' | 'bearer' | 'header';
    readonly headerName?: string;
    readonly value?: string;
    readonly valueProvider?: () => string | Promise<string>;
  };
  readonly staticHeaders: Readonly<Record<string, string>>;
  readonly timeoutMs?: number;
  readonly protocolOptions: Readonly<Record<string, unknown>>;
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
  'grammar',
  'json_schema',
  'jsonSchema',
  'sampling',
  'media',
  'normalize',
  'local',
]);
const ENDPOINT_CONFIG_FIELDS = new Set([
  'kind',
  'target',
  'baseUrl',
  'routes',
  'authentication',
  'staticHeaders',
  'timeoutMs',
  'protocolOptions',
]);
const MAX_GATEWAY_ERROR_BYTES = 1 << 20;
const MAX_GATEWAY_SSE_EVENT_BYTES = 1 << 20;
const GATEWAY_REQUEST_TIMEOUT_MESSAGE = 'gateway endpoint request timed out';
const GATEWAY_REQUEST_ABORTED_MESSAGE = 'gateway endpoint request aborted';
const GATEWAY_REQUEST_FAILED_MESSAGE = 'gateway endpoint request failed';
const GATEWAY_TOKEN_PROVIDER_FAILED_MESSAGE = 'gateway endpoint secret provider failed';
const GATEWAY_ERROR_BODY_TOO_LARGE_MESSAGE =
  'gateway endpoint error response exceeded body limit';
const GATEWAY_SSE_EVENT_TOO_LARGE_MESSAGE =
  'gateway stream event exceeded buffer limit';
const U32_MAX = 0xffffffff;
const F32_MAX = 3.4028234663852886e38;
const UTF8_ENCODER = new TextEncoder();
const UTF8_DECODER = new TextDecoder();

/** Registry for browser gateway endpoints. */
export class GatewayEndpointRegistry {
  readonly #endpoints = new Map<string, GatewayEndpoint>();

  public prepare(id: string, config: GatewayEndpointDescriptor): GatewayEndpoint {
    const normalizedId = normalizeId(id, 'endpoint id');
    return normalizeConfig(normalizedId, config);
  }

  public commit(endpoint: GatewayEndpoint): EndpointRef {
    this.#endpoints.set(endpoint.id, endpoint);
    return { kind: 'gateway', id: endpoint.id };
  }

  public remove(id: string): void {
    this.#endpoints.delete(id);
  }

  public get(endpoint: EndpointRef | undefined): GatewayEndpoint | null {
    if (endpoint == null || endpoint.kind !== 'gateway') {
      return null;
    }
    const registered = this.#endpoints.get(endpoint.id);
    if (registered == null) {
      throw new QueryError('MODEL_NOT_FOUND', `gateway endpoint not found: ${endpoint.id}`);
    }
    return registered;
  }
}

/** Run a browser query through a gateway endpoint. */
export async function runGatewayQuery(
  endpoint: GatewayEndpoint,
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
  rejectGatewayTextLocalOptions(options, hasMedia);
  const body = textBody(
    endpoint.target,
    options,
    combineEndpointOptions(endpoint.protocolOptions, options.endpointOptions),
    { prompt },
    tokenBatchSink != null
  );
  return tokenBatchSink == null
    ? gatewayRequest(endpoint, endpoint.routes.query, body, signal, async (response, token) =>
        parseTextResponse(await gatewayJsonBody(response, token))
      )
    : gatewayRequest(endpoint, endpoint.routes.query, body, signal, (response, token, resetTimeout) =>
        readTextStream(response, token, tokenBatchSink, resetTimeout)
      );
}

/** Run a browser chat request through a gateway endpoint. */
export async function runGatewayChat(
  endpoint: GatewayEndpoint,
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
  rejectGatewayTextLocalOptions(
    options,
    structuredInput?.media != null && mediaHasEntries(structuredInput.media)
  );
  const body = textBody(
    endpoint.target,
    options,
    combineEndpointOptions(endpoint.protocolOptions, options.endpointOptions),
    { messages: messages.map(gatewayMessage) },
    tokenBatchSink != null
  );
  return tokenBatchSink == null
    ? gatewayRequest(endpoint, endpoint.routes.chat, body, signal, async (response, token) =>
        parseTextResponse(await gatewayJsonBody(response, token))
      )
    : gatewayRequest(endpoint, endpoint.routes.chat, body, signal, (response, token, resetTimeout) =>
        readTextStream(response, token, tokenBatchSink, resetTimeout)
      );
}

/** Run a browser embedding request through a gateway endpoint. */
export async function runGatewayEmbedding(
  endpoint: GatewayEndpoint,
  input: string,
  options: EmbedOptions,
  signal: AbortSignal
): Promise<EmbeddingResult> {
  const embedInput = requiredStandaloneString(input, 'input');
  if (embedInput.trim().length === 0) {
    throw new QueryError('QUERY_FAILED', 'input must not be empty');
  }
  rejectGatewayEmbedLocalOptions(options);
  const body = mergeEndpointOptions(
    {
      model: endpoint.target,
      input: embedInput,
    },
    combineEndpointOptions(endpoint.protocolOptions, options.endpointOptions),
    EMBED_TYPED_FIELDS
  );
  return gatewayRequest(endpoint, endpoint.routes.embed, body, signal, async (response, token) =>
    parseEmbeddingResponse(await gatewayJsonBody(response, token))
  );
}

function normalizeConfig(id: string, config: GatewayEndpointDescriptor): GatewayEndpoint {
  if (typeof config !== 'object' || config == null || Array.isArray(config)) {
    throw new QueryError('QUERY_FAILED', 'gateway endpoint config must be an object');
  }
  rejectUnknownEndpointConfigFields(config);
  const target = normalizeId(config.target, 'endpoint target');
  if (typeof config.baseUrl !== 'string') {
    throw new QueryError('QUERY_FAILED', 'gateway endpoint baseUrl must be a string');
  }
  const trimmedBaseUrl = config.baseUrl.trim();
  if (trimmedBaseUrl.length === 0) {
    throw new QueryError('QUERY_FAILED', 'gateway endpoint baseUrl must not be empty');
  }
  if (trimmedBaseUrl !== config.baseUrl) {
    throw new QueryError(
      'QUERY_FAILED',
      'gateway endpoint baseUrl must not contain surrounding whitespace'
    );
  }
  const baseUrl = config.baseUrl.replace(/\/+$/, '');
  if (baseUrl.length === 0) {
    throw new QueryError('QUERY_FAILED', 'gateway endpoint baseUrl must not be empty');
  }
  validateEndpointBaseUrl(baseUrl);
  const authentication = normalizeAuthentication(config.authentication);
  if (
    config.timeoutMs != null &&
    (typeof config.timeoutMs !== 'number' ||
      !Number.isFinite(config.timeoutMs) ||
      config.timeoutMs <= 0)
  ) {
    throw new QueryError('QUERY_FAILED', 'gateway endpoint timeoutMs must be positive');
  }
  return {
    id,
    target,
    baseUrl,
    routes: {
      query: config.routes?.query ?? '/v1/query',
      chat: config.routes?.chat ?? '/v1/chat',
      embed: config.routes?.embed ?? '/v1/embed',
    },
    authentication,
    staticHeaders: config.staticHeaders ?? {},
    timeoutMs: config.timeoutMs,
    protocolOptions: config.protocolOptions ?? {},
  };
}

function rejectUnknownEndpointConfigFields(config: GatewayEndpointDescriptor): void {
  for (const field of Object.keys(config)) {
    if (!ENDPOINT_CONFIG_FIELDS.has(field)) {
      throw new QueryError('QUERY_FAILED', `unsupported gateway endpoint field: ${field}`);
    }
  }
}

function normalizeAuthentication(
  authentication: GatewayEndpointDescriptor['authentication']
): GatewayEndpoint['authentication'] {
  if (authentication == null || authentication.kind === 'none') {
    return { kind: 'none' };
  }
  if (authentication.kind === 'bearer') {
    if (authentication.value == null && authentication.valueProvider == null) {
      throw new QueryError('QUERY_FAILED', 'bearer authentication requires a value or valueProvider');
    }
    return authentication;
  }
  if (authentication.kind === 'header') {
    if (authentication.headerName == null) {
      throw new QueryError('QUERY_FAILED', 'header authentication requires headerName');
    }
    if (authentication.value == null && authentication.valueProvider == null) {
      throw new QueryError('QUERY_FAILED', 'header authentication requires a value or valueProvider');
    }
    return authentication;
  }
  throw new QueryError('QUERY_FAILED', 'unsupported authentication strategy');
}

function combineEndpointOptions(
  profileOptions: Readonly<Record<string, unknown>>,
  requestOptions: unknown
): Record<string, unknown> {
  if (requestOptions == null) {
    return { ...profileOptions };
  }
  if (typeof requestOptions !== 'object' || Array.isArray(requestOptions)) {
    throw new QueryError('QUERY_FAILED', 'endpointOptions must be a JSON object');
  }
  return { ...profileOptions, ...(requestOptions as Record<string, unknown>) };
}

function validateEndpointBaseUrl(baseUrl: string): void {
  let url: URL;
  try {
    url = new URL(baseUrl);
  } catch {
    throw new QueryError('QUERY_FAILED', 'gateway endpoint baseUrl is invalid');
  }
  if ((url.protocol !== 'http:' && url.protocol !== 'https:') || url.hostname.length === 0) {
    throw new QueryError(
      'QUERY_FAILED',
      'gateway endpoint baseUrl must be an absolute http(s) URL'
    );
  }
  if (url.username.length > 0 || url.password.length > 0) {
    throw new QueryError('QUERY_FAILED', 'gateway endpoint baseUrl must not include userinfo');
  }
  if (url.search.length > 0 || url.hash.length > 0) {
    throw new QueryError(
      'QUERY_FAILED',
      'gateway endpoint baseUrl must not include query or fragment'
    );
  }
  if (url.protocol === 'http:' && !isLoopbackHostname(url.hostname)) {
    throw new QueryError(
      'QUERY_FAILED',
      'gateway endpoint baseUrl must use HTTPS unless it targets loopback'
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

function rejectGatewayTextLocalOptions(options: QueryOptions, hasMedia: boolean): void {
  if (options.contextKey != null || options.grammar != null || hasMedia) {
    throw new QueryError(
      'UNSUPPORTED_OPERATION',
      'local text options are not valid for gateway endpoints'
    );
  }
}

function rejectGatewayEmbedLocalOptions(options: EmbedOptions): void {
  if (options.contextKey != null || options.normalize != null) {
    throw new QueryError(
      'UNSUPPORTED_OPERATION',
      'local embed options are not valid for gateway endpoints'
    );
  }
}

function textBody(
  target: string,
  options: QueryOptions,
  endpointOptions: unknown,
  payload: { readonly prompt: string } | { readonly messages: readonly unknown[] },
  stream: boolean
): Record<string, unknown> {
  const maxTokens = optionalPositiveU32(options.maxTokens, 'max_tokens');
  const temperature = optionalTemperature(options.temperature);
  const topP = optionalTopP(options.topP);
  const stop = optionalStringArray(options.stop, 'stop');
  return mergeEndpointOptions(
    {
      model: target,
      ...payload,
      ...(maxTokens == null ? {} : { max_tokens: maxTokens }),
      ...(temperature == null ? {} : { temperature }),
      ...(topP == null ? {} : { top_p: topP }),
      ...(stop == null ? {} : { stop }),
      stream,
    },
    endpointOptions,
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

function mergeEndpointOptions(
  body: Record<string, unknown>,
  endpointOptions: unknown,
  typedFields: ReadonlySet<string>
): Record<string, unknown> {
  if (endpointOptions == null) {
    return body;
  }
  if (typeof endpointOptions !== 'object' || Array.isArray(endpointOptions)) {
    throw new QueryError('QUERY_FAILED', 'endpointOptions must be a JSON object');
  }
  if (!isJsonObject(endpointOptions)) {
    throw new QueryError('QUERY_FAILED', 'endpointOptions must be a JSON object');
  }
  for (const [key, value] of Object.entries(endpointOptions)) {
    if (typedFields.has(key)) {
      throw new QueryError('QUERY_FAILED', `endpointOptions cannot override typed field: ${key}`);
    }
    if (LOCAL_ONLY_GATEWAY_FIELDS.has(key)) {
      throw new QueryError('QUERY_FAILED', `endpointOptions cannot contain local-only field: ${key}`);
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
      throw new QueryError('QUERY_FAILED', 'endpointOptions cannot contain non-finite numbers');
    }
    return value;
  }
  if (typeof value !== 'object') {
    throw new QueryError('QUERY_FAILED', 'endpointOptions must contain JSON-compatible values');
  }

  if (ancestors.has(value)) {
    throw new QueryError('QUERY_FAILED', 'endpointOptions must contain JSON-compatible values');
  }
  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      return value.map((item) => snapshotJsonCompatibleValue(item, ancestors));
    }
    if (!isJsonObject(value)) {
      throw new QueryError('QUERY_FAILED', 'endpointOptions must contain JSON-compatible values');
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
  endpoint: GatewayEndpoint,
  path: string,
  body: Record<string, unknown>,
  signal: AbortSignal,
  read: (response: Response, token: string, resetTimeout: () => void) => Promise<T>
): Promise<T> {
  const abort = createTimedAbortController(signal, endpoint.timeoutMs);

  try {
    throwIfEndpointAborted(abort.signal, abort.timedOut());
    const token = await endpointSecret(endpoint, abort.signal);
    const headers = await endpointHeaders(endpoint, token);
    throwIfEndpointAborted(abort.signal, abort.timedOut());
    const response = await fetch(`${endpoint.baseUrl}${path}`, {
      method: 'POST',
      headers: {
        ...headers,
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

async function endpointSecret(
  endpoint: GatewayEndpoint,
  signal: AbortSignal
): Promise<string> {
  if (endpoint.authentication.kind === 'none') {
    return '';
  }
  let token: unknown;
  try {
    token =
      endpoint.authentication.value ??
      (await endpointValueProvider(endpoint.authentication.valueProvider, signal));
  } catch {
    if (signal.aborted) {
      throw new Error('gateway endpoint secret provider aborted');
    }
    throw new QueryError('QUERY_FAILED', GATEWAY_TOKEN_PROVIDER_FAILED_MESSAGE);
  }
  if (typeof token !== 'string') {
    throw new QueryError('QUERY_FAILED', 'gateway endpoint secret must be a string');
  }
  validateEndpointSecret(token);
  return token;
}

async function endpointValueProvider(
  tokenProvider: GatewayEndpoint['authentication']['valueProvider'],
  signal: AbortSignal
): Promise<unknown> {
  if (tokenProvider == null) {
    return undefined;
  }
  if (signal.aborted) {
    throw new Error('gateway endpoint secret provider aborted');
  }

  let removeAbortListener = (): void => {};
  const abortPromise = new Promise<never>((_resolve, reject) => {
    const abortListener = (): void => {
      reject(new Error('gateway endpoint secret provider aborted'));
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

async function endpointHeaders(
  endpoint: GatewayEndpoint,
  secret: string
): Promise<Record<string, string>> {
  const headers: Record<string, string> = { ...endpoint.staticHeaders };
  if (endpoint.authentication.kind === 'bearer') {
    headers.Authorization = `Bearer ${secret}`;
  } else if (endpoint.authentication.kind === 'header') {
    headers[endpoint.authentication.headerName ?? 'Authorization'] = secret;
  }
  return headers;
}

function validateEndpointSecret(token: string): void {
  if (token.trim().length === 0) {
    throw new QueryError('QUERY_FAILED', 'gateway endpoint secret must not be empty');
  }
  if (/\s/u.test(token)) {
    throw new QueryError('QUERY_FAILED', 'gateway endpoint secret must not contain whitespace');
  }
}

function throwIfEndpointAborted(signal: AbortSignal, timedOut: boolean): void {
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
      : response.statusText || 'gateway endpoint error';
  return new QueryError('QUERY_FAILED', redactSecret(message, token), {
    status: response.status,
    protocolCode: redactOptionalSecret(gatewayErrorCode(body), token),
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
    protocolCode: redactOptionalSecret(gatewayErrorCode(body), token),
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
    throw new QueryError(
      'STREAMING_UNAVAILABLE',
      'gateway endpoint response body is not streamable'
    );
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
      protocolCode: redactOptionalSecret(gatewayErrorCode(data), state.token),
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
    throw new QueryError('QUERY_FAILED', 'gateway endpoint response must be a JSON object');
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
  context = 'gateway endpoint response'
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
    throw new QueryError('QUERY_FAILED', `gateway endpoint response missing ${key}`);
  }
  return value.map((item) => {
    if (typeof item !== 'number' || !Number.isFinite(item)) {
      throw new QueryError('QUERY_FAILED', `gateway endpoint ${key} contains non-finite value`);
    }
    if (item < -F32_MAX || item > F32_MAX) {
      throw new QueryError(
        'QUERY_FAILED',
        `gateway endpoint ${key} contains value outside f32 range`
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
  return typeof value === 'string' && value.length > 0
    ? value
    : 'gateway endpoint error';
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
