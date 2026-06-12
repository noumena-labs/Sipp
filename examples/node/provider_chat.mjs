import native from '../../lib/node/router.js';
import {
  DEFAULT_MAX_TOKENS,
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_P,
  intEnv,
  numberEnv,
  printText,
} from './_support.mjs';

const GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/';
const GEMINI_DEFAULT_MODEL = 'gemini-3.5-flash';
const OPENAI_DEFAULT_MODEL = 'gpt-5-mini';

const { SippClient } = native;
const input =
  process.argv.slice(2).join(' ') || 'Say hello from a direct provider.';

// Direct providers belong in trusted Node processes. Browser code should call a
// gateway or application route instead of holding provider credentials.
const client = new SippClient();
const endpoint = await client.add('provider', providerDescriptor());
const result = await client.chat({
  endpoint,
  messages: [{ role: 'user', content: input }],
  options: textOptions(),
}).response;
printText(result);

function providerDescriptor() {
  const provider = providerName();
  switch (provider) {
    case 'gemini': {
      return {
        kind: 'provider',
        provider: 'openai_compatible',
        model: envAny(
          ['SIPP_PROVIDER_MODEL', 'GEMINI_MODEL'],
          GEMINI_DEFAULT_MODEL,
        ),
        baseUrl: env('SIPP_PROVIDER_BASE_URL') ?? GEMINI_BASE_URL,
        apiKey: requiredEnvAny(['SIPP_PROVIDER_API_KEY', 'GEMINI_API_KEY']),
        timeoutMs: providerTimeoutMs(),
      };
    }
    case 'openai': {
      const baseUrl = envAny(['SIPP_PROVIDER_BASE_URL', 'OPENAI_BASE_URL']);
      return {
        kind: 'provider',
        provider: 'openai',
        model: envAny(
          ['SIPP_PROVIDER_MODEL', 'OPENAI_MODEL'],
          OPENAI_DEFAULT_MODEL,
        ),
        apiKey: requiredEnvAny(['SIPP_PROVIDER_API_KEY', 'OPENAI_API_KEY']),
        ...(baseUrl == null ? {} : { baseUrl }),
        timeoutMs: providerTimeoutMs(),
      };
    }
    case 'anthropic': {
      const baseUrl = envAny(['SIPP_PROVIDER_BASE_URL', 'ANTHROPIC_BASE_URL']);
      const version = env('ANTHROPIC_VERSION');
      return {
        kind: 'provider',
        provider: 'anthropic',
        model: requiredEnvAny(['SIPP_PROVIDER_MODEL', 'ANTHROPIC_MODEL']),
        apiKey: requiredEnvAny(['SIPP_PROVIDER_API_KEY', 'ANTHROPIC_API_KEY']),
        ...(baseUrl == null ? {} : { baseUrl }),
        ...(version == null ? {} : { version }),
        timeoutMs: providerTimeoutMs(),
      };
    }
    case 'openai_compatible':
      return {
        kind: 'provider',
        provider: 'openai_compatible',
        model: requiredEnvAny(['SIPP_PROVIDER_MODEL']),
        baseUrl: requiredEnvAny(['SIPP_PROVIDER_BASE_URL']),
        ...openAiCompatibleAuth(),
        timeoutMs: providerTimeoutMs(),
      };
    default:
      throw new Error(
        'SIPP_PROVIDER must be gemini, openai, anthropic, or openai_compatible',
      );
  }
}

function providerName() {
  return (env('SIPP_PROVIDER') ?? 'gemini')
    .toLowerCase()
    .replaceAll('-', '_');
}

function openAiCompatibleAuth() {
  const headerName = env('SIPP_PROVIDER_AUTH_HEADER_NAME');
  const headerValue = env('SIPP_PROVIDER_AUTH_HEADER_VALUE');
  if (headerName != null || headerValue != null) {
    if (headerName == null || headerValue == null) {
      throw new Error(
        'SIPP_PROVIDER_AUTH_HEADER_NAME and ' +
          'SIPP_PROVIDER_AUTH_HEADER_VALUE must be set together',
      );
    }
    return { authHeaderName: headerName, authHeaderValue: headerValue };
  }
  return { apiKey: requiredEnvAny(['SIPP_PROVIDER_API_KEY']) };
}

function textOptions() {
  return {
    maxTokens: intEnv('SIPP_MAX_TOKENS', DEFAULT_MAX_TOKENS),
    temperature: numberEnv('SIPP_TEMPERATURE', DEFAULT_TEMPERATURE),
    topP: numberEnv('SIPP_TOP_P', DEFAULT_TOP_P),
  };
}

function providerTimeoutMs() {
  return intEnv('SIPP_PROVIDER_TIMEOUT_MS', 30_000);
}

function env(name) {
  const value = process.env[name];
  return value == null || value === '' ? undefined : value;
}

function envAny(names, fallback = undefined) {
  for (const name of names) {
    const value = env(name);
    if (value != null) return value;
  }
  return fallback;
}

function requiredEnvAny(names) {
  const value = envAny(names);
  if (value == null) {
    throw new Error(`${names.join(' or ')} is required`);
  }
  return value;
}
