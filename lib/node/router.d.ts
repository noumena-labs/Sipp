import type {
  SippChatRequest,
  SippEmbedRequest,
  SippEmbeddingResponse,
  SippQueryRequest,
  SippTextResponse,
  SippTextRun,
} from './index'
export * from './index'

/** Native backend selected by the Node package loader. */
export type ActiveNodeBackend = 'cpu' | 'cuda' | 'metal' | 'vulkan'

/** Return the backend selected for the currently loaded native binding. */
export declare function getActiveBackend(): ActiveNodeBackend

/** Error raised while decoding or formatting the first-party gateway profile. */
export declare class GatewayProfileError extends Error {
  constructor(
    code: string,
    message: string,
    options?: { readonly status?: number }
  )
  readonly code: string
  readonly status: number
}

/** Query request decoded from the first-party gateway JSON profile. */
export interface GatewayDecodedQuery {
  readonly target: string
  readonly stream: boolean
  readonly request: SippQueryRequest
}

/** Chat request decoded from the first-party gateway JSON profile. */
export interface GatewayDecodedChat {
  readonly target: string
  readonly stream: boolean
  readonly request: SippChatRequest
}

/** Embedding request decoded from the first-party gateway JSON profile. */
export interface GatewayDecodedEmbed {
  readonly target: string
  readonly stream: false
  readonly request: SippEmbedRequest
}

/** Token usage encoded with first-party gateway snake_case field names. */
export interface GatewayUsageBody {
  readonly input_tokens?: number
  readonly output_tokens?: number
  readonly total_tokens?: number
}

/** Text response body consumed by first-party gateway clients. */
export interface GatewayTextResponseBody {
  readonly id: string
  readonly model: string
  readonly text: string
  readonly finish_reason: string
  readonly usage?: GatewayUsageBody
}

/** Embedding response body consumed by first-party gateway clients. */
export interface GatewayEmbeddingResponseBody {
  readonly id: string
  readonly model: string
  readonly embedding: readonly number[]
  readonly usage?: GatewayUsageBody
}

/** JSON error body consumed by first-party gateway clients. */
export interface GatewayErrorBody {
  readonly error: {
    readonly code: string
    readonly message: string
  }
}

/** Response payload and status selected for a gateway profile error. */
export interface GatewayErrorResponse {
  readonly body: GatewayErrorBody
  readonly init: {
    readonly status: number
  }
}

/** Decode a first-party gateway query request body. */
export declare function decodeGatewayQueryBody(body: unknown): GatewayDecodedQuery

/** Decode a first-party gateway chat request body. */
export declare function decodeGatewayChatBody(body: unknown): GatewayDecodedChat

/** Decode a first-party gateway embedding request body. */
export declare function decodeGatewayEmbedBody(body: unknown): GatewayDecodedEmbed

/** Format a text response for first-party gateway clients. */
export declare function gatewayTextResponseBody(
  target: string,
  response: SippTextResponse
): GatewayTextResponseBody

/** Format an embedding response for first-party gateway clients. */
export declare function gatewayEmbeddingResponseBody(
  target: string,
  response: SippEmbeddingResponse
): GatewayEmbeddingResponseBody

/** Format a streaming text run as first-party gateway SSE events. */
export declare function gatewayTextStreamResponse(run: SippTextRun): Response

/** Format an error as the first-party gateway JSON error envelope. */
export declare function gatewayErrorResponse(error: unknown): GatewayErrorResponse

/** Return whether an error was raised by the gateway profile helpers. */
export declare function isGatewayProfileError(error: unknown): error is GatewayProfileError
