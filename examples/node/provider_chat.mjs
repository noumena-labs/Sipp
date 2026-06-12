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

const { CogentClient } = native;
const input =
  process.argv.slice(2).join(' ') || 'Say hello from a direct provider.';

// Direct providers belong in trusted Node processes. Browser code should call a
// gateway or application route instead of holding provider credentials.
const client = new CogentClient();
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
          ['COGENTLM_PROVIDER_MODEL', 'GEMINI_MODEL'],
          GEMINI_DEFAULT_MODEL,
        ),
        baseUrl: env('COGENTLM_PROVIDER_BASE_URL') ?? GEMINI_BASE_URL,
        apiKey: requiredEnvAny(['COGENTLM_PROVIDER_API_KEY', 'GEMINI_API_KEY']),
        timeoutMs: providerTimeoutMs(),
      };
    }
    case 'openai': {
      const baseUrl = envAny(['COGENTLM_PROVIDER_BASE_URL', 'OPENAI_BASE_URL']);
      return {
        kind: 'provider',
        provider: 'openai',
        model: envAny(
          ['COGENTLM_PROVIDER_MODEL', 'OPENAI_MODEL'],
          OPENAI_DEFAULT_MODEL,
        ),
        apiKey: requiredEnvAny(['COGENTLM_PROVIDER_API_KEY', 'OPENAI_API_KEY']),
        ...(baseUrl == null ? {} : { baseUrl }),
        timeoutMs: providerTimeoutMs(),
      };
    }
    case 'anthropic': {
      const baseUrl = envAny(['COGENTLM_PROVIDER_BASE_URL', 'ANTHROPIC_BASE_URL']);
      const version = env('ANTHROPIC_VERSION');
      return {
        kind: 'provider',
        provider: 'anthropic',
        model: requiredEnvAny(['COGENTLM_PROVIDER_MODEL', 'ANTHROPIC_MODEL']),
        apiKey: requiredEnvAny(['COGENTLM_PROVIDER_API_KEY', 'ANTHROPIC_API_KEY']),
        ...(baseUrl == null ? {} : { baseUrl }),
        ...(version == null ? {} : { version }),
        timeoutMs: providerTimeoutMs(),
      };
    }
    case 'openai_compatible':
      return {
        kind: 'provider',
        provider: 'openai_compatible',
        model: requiredEnvAny(['COGENTLM_PROVIDER_MODEL']),
        baseUrl: requiredEnvAny(['COGENTLM_PROVIDER_BASE_URL']),
        ...openAiCompatibleAuth(),
        timeoutMs: providerTimeoutMs(),
      };
    default:
      throw new Error(
        'COGENTLM_PROVIDER must be gemini, openai, anthropic, or openai_compatible',
      );
  }
}

function providerName() {
  return (env('COGENTLM_PROVIDER') ?? 'gemini')
    .toLowerCase()
    .replaceAll('-', '_');
}

function openAiCompatibleAuth() {
  const headerName = env('COGENTLM_PROVIDER_AUTH_HEADER_NAME');
  const headerValue = env('COGENTLM_PROVIDER_AUTH_HEADER_VALUE');
  if (headerName != null || headerValue != null) {
    if (headerName == null || headerValue == null) {
      throw new Error(
        'COGENTLM_PROVIDER_AUTH_HEADER_NAME and ' +
          'COGENTLM_PROVIDER_AUTH_HEADER_VALUE must be set together',
      );
    }
    return { authHeaderName: headerName, authHeaderValue: headerValue };
  }
  return { apiKey: requiredEnvAny(['COGENTLM_PROVIDER_API_KEY']) };
}

function textOptions() {
  return {
    maxTokens: intEnv('COGENTLM_MAX_TOKENS', DEFAULT_MAX_TOKENS),
    temperature: numberEnv('COGENTLM_TEMPERATURE', DEFAULT_TEMPERATURE),
    topP: numberEnv('COGENTLM_TOP_P', DEFAULT_TOP_P),
  };
}

function providerTimeoutMs() {
  return intEnv('COGENTLM_PROVIDER_TIMEOUT_MS', 30_000);
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
