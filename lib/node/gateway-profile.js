'use strict'

const TEXT_TYPED_FIELDS = new Set([
  'model',
  'prompt',
  'messages',
  'max_tokens',
  'temperature',
  'top_p',
  'stop',
  'stream',
])
const EMBED_TYPED_FIELDS = new Set(['model', 'input'])
const U32_MAX = 0xffffffff

/** Error raised while decoding or formatting the first-party gateway profile. */
class GatewayProfileError extends Error {
  constructor(code, message, options = undefined) {
    super(message)
    this.name = 'GatewayProfileError'
    this.code = code
    this.status = options?.status ?? 400
  }
}

/** Decode a first-party gateway query request body. */
function decodeGatewayQueryBody(body) {
  const value = objectBody(body)
  const target = requiredString(value, 'model')
  const prompt = requiredString(value, 'prompt')
  return {
    target,
    stream: streamFlag(value.stream),
    request: textRequest(
      {
        prompt,
      },
      value,
      TEXT_TYPED_FIELDS,
    ),
  }
}

/** Decode a first-party gateway chat request body. */
function decodeGatewayChatBody(body) {
  const value = objectBody(body)
  const target = requiredString(value, 'model')
  return {
    target,
    stream: streamFlag(value.stream),
    request: textRequest(
      {
        messages: chatMessages(value.messages),
      },
      value,
      TEXT_TYPED_FIELDS,
    ),
  }
}

/** Decode a first-party gateway embedding request body. */
function decodeGatewayEmbedBody(body) {
  const value = objectBody(body)
  const target = requiredString(value, 'model')
  const input = requiredString(value, 'input')
  const endpointOptions = profileOptions(value, EMBED_TYPED_FIELDS)
  return {
    target,
    stream: false,
    request: {
      input,
      ...(Object.keys(endpointOptions).length === 0 ? {} : { endpointOptions }),
    },
  }
}

/** Format a text response for first-party gateway clients. */
function gatewayTextResponseBody(target, response) {
  const body = {
    id: responseId(response),
    model: target,
    text: response.text,
    finish_reason: response.finishReason,
  }
  const usage = gatewayUsage(response.usage)
  if (usage != null) {
    body.usage = usage
  }
  return body
}

/** Format an embedding response for first-party gateway clients. */
function gatewayEmbeddingResponseBody(target, response) {
  const body = {
    id: responseId(response),
    model: target,
    embedding: response.values,
  }
  const usage = gatewayUsage(response.usage)
  if (usage != null) {
    body.usage = usage
  }
  return body
}

/** Format a streaming text run as first-party gateway SSE events. */
function gatewayTextStreamResponse(run) {
  const encoder = new TextEncoder()
  const stream = new ReadableStream({
    async start(controller) {
      try {
        for await (const batch of run.tokens) {
          controller.enqueue(
            encoder.encode(
              sseEvent('token', {
                text: batch.text,
                sequence: batch.sequenceStart,
              }),
            ),
          )
        }

        const response = await run.response
        const usage = gatewayUsage(response.usage)
        if (usage != null) {
          controller.enqueue(encoder.encode(sseEvent('usage', usage)))
        }
        controller.enqueue(
          encoder.encode(
            sseEvent('done', {
              finish_reason: response.finishReason,
            }),
          ),
        )
        controller.close()
      } catch (error) {
        controller.enqueue(encoder.encode(sseEvent('error', gatewayErrorResponse(error).body)))
        controller.close()
      }
    },
    cancel() {
      if (typeof run.cancel === 'function') {
        run.cancel('client_disconnected')
      }
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-store',
    },
  })
}

/** Format an error as the first-party gateway JSON error envelope. */
function gatewayErrorResponse(error) {
  if (isGatewayProfileError(error)) {
    return {
      body: {
        error: {
          code: error.code,
          message: error.message,
        },
      },
      init: {
        status: error.status,
      },
    }
  }

  return {
    body: {
      error: {
        code: errorCode(error),
        message: errorMessage(error),
      },
    },
    init: {
      status: errorStatus(error),
    },
  }
}

/** Return whether an error was raised by the gateway profile helpers. */
function isGatewayProfileError(error) {
  return error instanceof GatewayProfileError
}

function textRequest(payload, body, typedFields) {
  const options = textOptions(body)
  const endpointOptions = profileOptions(body, typedFields)
  return {
    ...payload,
    emitTokens: streamFlag(body.stream),
    ...(Object.keys(options).length === 0 ? {} : { options }),
    ...(Object.keys(endpointOptions).length === 0 ? {} : { endpointOptions }),
  }
}

function textOptions(body) {
  const options = {}
  const maxTokens = optionalPositiveU32(body.max_tokens, 'max_tokens')
  const temperature = optionalTemperature(body.temperature)
  const topP = optionalTopP(body.top_p)
  const stop = optionalStringArray(body.stop, 'stop')
  if (maxTokens != null) {
    options.maxTokens = maxTokens
  }
  if (temperature != null) {
    options.temperature = temperature
  }
  if (topP != null) {
    options.topP = topP
  }
  if (stop != null) {
    options.stop = stop
  }
  return options
}

function profileOptions(body, typedFields) {
  const options = {}
  for (const [key, value] of Object.entries(body)) {
    if (!typedFields.has(key)) {
      options[key] = jsonCompatible(value, key)
    }
  }
  return options
}

function objectBody(value) {
  if (typeof value !== 'object' || value == null || Array.isArray(value)) {
    throw new GatewayProfileError(
      'invalid_request',
      'gateway request body must be a JSON object',
    )
  }
  return value
}

function requiredString(body, key) {
  const value = body[key]
  if (typeof value !== 'string' || value.trim() === '') {
    throw new GatewayProfileError('invalid_request', `${key} is required`)
  }
  return value
}

function streamFlag(value) {
  if (value == null) {
    return false
  }
  if (typeof value !== 'boolean') {
    throw new GatewayProfileError('invalid_request', 'stream must be a boolean')
  }
  return value
}

function chatMessages(value) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new GatewayProfileError('invalid_request', 'messages are required')
  }
  return value.map((item) => {
    const message = objectBody(item)
    const role = message.role
    if (role !== 'system' && role !== 'user' && role !== 'assistant') {
      throw new GatewayProfileError(
        'invalid_request',
        'message role must be system, user, or assistant',
      )
    }
    return {
      role,
      content: requiredString(message, 'content'),
    }
  })
}

function optionalPositiveU32(value, key) {
  if (value == null) {
    return undefined
  }
  if (
    typeof value !== 'number' ||
    !Number.isInteger(value) ||
    value <= 0 ||
    value > U32_MAX
  ) {
    throw new GatewayProfileError('invalid_request', `${key} must be a positive u32 integer`)
  }
  return value
}

function optionalFiniteNumber(value, key) {
  if (value == null) {
    return undefined
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new GatewayProfileError('invalid_request', `${key} must be finite`)
  }
  return value
}

function optionalTemperature(value) {
  const temperature = optionalFiniteNumber(value, 'temperature')
  if (temperature != null && temperature < 0) {
    throw new GatewayProfileError(
      'invalid_request',
      'temperature must be greater than or equal to zero',
    )
  }
  return temperature
}

function optionalTopP(value) {
  const topP = optionalFiniteNumber(value, 'top_p')
  if (topP != null && (topP < 0 || topP > 1)) {
    throw new GatewayProfileError('invalid_request', 'top_p must be between 0 and 1')
  }
  return topP
}

function optionalStringArray(value, key) {
  if (value == null) {
    return undefined
  }
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'string')) {
    throw new GatewayProfileError('invalid_request', `${key} must be an array of strings`)
  }
  return [...value]
}

function jsonCompatible(value, key) {
  if (value == null || typeof value === 'string' || typeof value === 'boolean') {
    return value
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new GatewayProfileError('invalid_request', `${key} must be JSON-compatible`)
    }
    return value
  }
  if (Array.isArray(value)) {
    return value.map((item) => jsonCompatible(item, key))
  }
  if (typeof value === 'object') {
    const copy = {}
    for (const [childKey, childValue] of Object.entries(value)) {
      copy[childKey] = jsonCompatible(childValue, childKey)
    }
    return copy
  }
  throw new GatewayProfileError('invalid_request', `${key} must be JSON-compatible`)
}

function responseId(response) {
  return response.metadata?.upstreamResponseId ?? 'response'
}

function gatewayUsage(usage) {
  if (usage == null) {
    return undefined
  }
  return {
    input_tokens: usage.inputTokens,
    output_tokens: usage.outputTokens,
    total_tokens: usage.totalTokens,
  }
}

function sseEvent(name, value) {
  return `event: ${name}\ndata: ${JSON.stringify(value)}\n\n`
}

function errorCode(error) {
  return typeof error?.code === 'string' && error.code !== '' ? error.code : 'internal'
}

function errorMessage(error) {
  return error instanceof Error && error.message !== '' ? error.message : 'gateway request failed'
}

function errorStatus(error) {
  const status = error?.status
  return Number.isInteger(status) && status >= 400 && status <= 599 ? status : 500
}

module.exports = {
  GatewayProfileError,
  decodeGatewayQueryBody,
  decodeGatewayChatBody,
  decodeGatewayEmbedBody,
  gatewayTextResponseBody,
  gatewayEmbeddingResponseBody,
  gatewayTextStreamResponse,
  gatewayErrorResponse,
  isGatewayProfileError,
}
