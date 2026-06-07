'use strict'

const { randomUUID } = require('node:crypto')
const { CogentClient } = require('./router.js')

const DEFAULT_MAX_REQUEST_BYTES = 1 << 20
const MAX_REQUEST_ID_BYTES = 128
const OPERATIONS = new Set(['query', 'chat', 'embed'])

function createNextGateway(options) {
  assertNodeRuntime()
  if (options == null || typeof options !== 'object') {
    throw new TypeError('createNextGateway options are required')
  }
  if (options.auth !== 'none' && typeof options.auth !== 'function') {
    throw new TypeError('auth must be "none" or an async authorization callback')
  }
  const aliases = options.aliases
  if (aliases == null || typeof aliases !== 'object' || Array.isArray(aliases)) {
    throw new TypeError('aliases must be a non-empty object')
  }
  const entries = Object.entries(aliases)
  if (entries.length === 0) {
    throw new TypeError('aliases must be a non-empty object')
  }
  const maxRequestBytes = options.maxRequestBytes ?? DEFAULT_MAX_REQUEST_BYTES
  if (!Number.isSafeInteger(maxRequestBytes) || maxRequestBytes <= 0) {
    throw new TypeError('maxRequestBytes must be a positive safe integer')
  }

  const client = new CogentClient()
  const endpoints = new Map()
  const ready = (async () => {
    for (const [alias, descriptor] of entries) {
      if (!validAlias(alias)) {
        throw new TypeError(`invalid gateway alias: ${alias}`)
      }
      endpoints.set(alias, await client.add(alias, descriptor))
    }
  })()

  return async function nextGatewayHandler(request, routeContext) {
    const requestId = canonicalRequestId(request.headers.get('x-request-id'))
    try {
      await authorize(options.auth, request)
      await ready
      const operation = await routeOperation(request, routeContext)
      const body = await readJsonBody(request, maxRequestBytes)
      const endpoint = endpoints.get(body.model)
      if (endpoint == null) {
        return errorResponse(404, 'model_not_found', 'model alias not found', requestId)
      }
      const run = startRun(client, operation, endpoint, body, requestId)
      if (request.signal.aborted) {
        run.cancel('client_disconnected')
      }
      if (body.stream === true && operation !== 'embed') {
        return streamResponse(run, request.signal, requestId)
      }
      return unaryResponse(run, operation, body.model, request.signal, requestId)
    } catch (error) {
      return mappedErrorResponse(error, requestId)
    }
  }
}

async function authorize(auth, request) {
  if (auth === 'none') {
    return
  }
  const result = await auth(request)
  if (result !== true && (result == null || result.authorized !== true)) {
    const error = new Error('request is not authorized')
    error.gatewayCode = 'authorization'
    error.status = 403
    throw error
  }
}

async function routeOperation(request, routeContext) {
  const params = await routeContext?.params
  const operation = params?.operation ?? new URL(request.url).pathname.split('/').filter(Boolean).at(-1)
  if (!OPERATIONS.has(operation)) {
    const error = new Error('unsupported gateway operation')
    error.gatewayCode = 'unsupported_feature'
    error.status = 404
    throw error
  }
  return operation
}

async function readJsonBody(request, maxRequestBytes) {
  const contentLength = Number(request.headers.get('content-length'))
  if (Number.isFinite(contentLength) && contentLength > maxRequestBytes) {
    throw requestTooLarge()
  }
  const bytes = new Uint8Array(await request.arrayBuffer())
  if (bytes.byteLength > maxRequestBytes) {
    throw requestTooLarge()
  }
  try {
    return JSON.parse(new TextDecoder().decode(bytes))
  } catch {
    const error = new Error('invalid JSON request body')
    error.gatewayCode = 'invalid_request'
    error.status = 400
    throw error
  }
}

function startRun(client, operation, endpoint, body, requestId) {
  const options = {
    maxTokens: body.max_tokens,
    temperature: body.temperature,
    topP: body.top_p,
    stop: body.stop,
  }
  if (operation === 'query') {
    return client.query({
      requestId,
      endpoint,
      prompt: body.prompt,
      options,
      emitTokens: body.stream === true,
    })
  }
  if (operation === 'chat') {
    return client.chat({
      requestId,
      endpoint,
      messages: body.messages,
      options,
      emitTokens: body.stream === true,
    })
  }
  return client.embed({
    requestId,
    endpoint,
    input: body.input,
  })
}

async function unaryResponse(run, operation, model, signal, requestId) {
  const abort = () => run.cancel('client_disconnected')
  signal.addEventListener('abort', abort, { once: true })
  if (signal.aborted) {
    abort()
  }
  try {
    const response = await run.response
    const body =
      operation === 'embed'
        ? {
            id: requestId,
            model,
            embedding: response.values,
            usage: wireUsage(response.usage),
          }
        : {
            id: requestId,
            model,
            text: response.text,
            finish_reason: response.finishReason,
            usage: wireUsage(response.usage),
          }
    return jsonResponse(200, body, requestId)
  } finally {
    signal.removeEventListener('abort', abort)
  }
}

function streamResponse(run, signal, requestId) {
  const encoder = new TextEncoder()
  const abort = () => run.cancel('client_disconnected')
  signal.addEventListener('abort', abort, { once: true })
  if (signal.aborted) {
    abort()
  }
  const iterator = run[Symbol.asyncIterator]()
  let finished = false

  const finish = () => {
    if (!finished) {
      finished = true
      signal.removeEventListener('abort', abort)
    }
  }

  const body = new ReadableStream({
    async pull(controller) {
      if (finished) {
        return
      }
      try {
        const next = await iterator.next()
        if (!next.done) {
          const batch = next.value
          controller.enqueue(sseBytes(encoder, 'token', {
            text: batch.text,
            sequence: batch.sequence_start,
          }))
          return
        }

        const response = await run.response
        if (response.usage != null) {
          controller.enqueue(sseBytes(encoder, 'usage', wireUsage(response.usage)))
        }
        controller.enqueue(
          sseBytes(encoder, 'done', {
            finish_reason: response.finishReason,
          }),
        )
        finish()
        controller.close()
      } catch (error) {
        const mapped = mapError(error)
        controller.enqueue(
          sseBytes(encoder, 'error', {
            error: { code: mapped.code, message: mapped.message },
          }),
        )
        finish()
        controller.close()
      }
    },
    cancel() {
      finish()
      run.cancel('client_disconnected')
    },
  })
  return new Response(body, {
    status: 200,
    headers: {
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      'content-type': 'text/event-stream',
      'x-request-id': requestId,
    },
  })
}

function sseBytes(encoder, event, data) {
  return encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
}

function wireUsage(usage) {
  if (usage == null) {
    return undefined
  }
  return {
    input_tokens: usage.inputTokens,
    output_tokens: usage.outputTokens,
    total_tokens: usage.totalTokens,
  }
}

function mappedErrorResponse(error, requestId) {
  const mapped = mapError(error)
  return errorResponse(mapped.status, mapped.code, mapped.message, requestId)
}

function mapError(error) {
  const message = typeof error?.message === 'string' ? error.message : 'gateway request failed'
  const code = normalizeErrorCode(
    error?.gatewayCode ??
      error?.kind ??
      cancellationCode(message) ??
      error?.code ??
      (error?.name === 'ProviderError' || error?.name === 'RemoteError'
        ? 'transport'
        : 'internal'),
  )
  return {
    status: Number.isInteger(error?.status) ? error.status : statusForCode(code),
    code,
    message,
  }
}

function normalizeErrorCode(code) {
  if (code === 'InvalidArg' || code === 'invalid_argument') {
    return 'invalid_request'
  }
  if (code === 'GenericFailure') {
    return 'internal'
  }
  return typeof code === 'string' ? code : 'internal'
}

function cancellationCode(message) {
  for (const code of [
    'client_disconnected',
    'server_shutdown',
    'caller_cancelled',
    'deadline_exceeded',
  ]) {
    if (message.includes(code)) {
      return code === 'server_shutdown' ? 'server_restarting' : code
    }
  }
  return null
}

function statusForCode(code) {
  if (code === 'authentication') return 401
  if (code === 'authorization') return 403
  if (code === 'model_not_found') return 404
  if (code === 'request_too_large') return 413
  if (code === 'rate_limited') return 429
  if (code === 'server_restarting' || code === 'overloaded') return 503
  if (code === 'deadline_exceeded' || code === 'timeout') return 408
  if (code === 'invalid_request' || code === 'unsupported_feature') return 400
  return 500
}

function jsonResponse(status, body, requestId) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'content-type': 'application/json',
      'x-request-id': requestId,
    },
  })
}

function errorResponse(status, code, message, requestId) {
  return jsonResponse(status, { error: { code, message } }, requestId)
}

function requestTooLarge() {
  const error = new Error('request body exceeds gateway limit')
  error.gatewayCode = 'request_too_large'
  error.status = 413
  return error
}

function canonicalRequestId(value) {
  return validRequestId(value) ? value : `next_${randomUUID()}`
}

function validRequestId(value) {
  return (
    typeof value === 'string' &&
    value.length > 0 &&
    Buffer.byteLength(value, 'ascii') <= MAX_REQUEST_ID_BYTES &&
    /^[\x21-\x7e]+$/.test(value)
  )
}

function validAlias(value) {
  return typeof value === 'string' && value.length > 0 && value.trim() === value
}

function assertNodeRuntime() {
  if (typeof process === 'undefined' || process.versions?.node == null) {
    throw new Error('@noumena-labs/cogentlm-server/next requires the Next.js Node.js runtime')
  }
}

module.exports = { createNextGateway }
