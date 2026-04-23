//////////////////////////////////////////////////////////////////////////////
//
// director-config.ts
//
// - Parser/validator for `director.json`.
// - Mirrors the character-config approach: validate once at load time and
//   keep the runtime/editor surface deterministic and explicit.
//
//////////////////////////////////////////////////////////////////////////////

import type {
  DirectorConfig,
  DirectorProfileConfig,
  DirectorQueryConfig,
  DirectorScenarioConfig,
  ResponseArraySchema,
  ResponseBooleanSchema,
  ResponseNullSchema,
  ResponseNumberSchema,
  ResponseObjectSchema,
  ResponseSchema,
  ResponseStringSchema,
} from './director-types.js';

const IDENTIFIER_RE = /^[A-Za-z0-9_-]+$/;

export class DirectorConfigError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = 'DirectorConfigError';
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function parseOptionalString(value: unknown, path: string): string | undefined {
  if (value == null) {
    return undefined;
  }
  if (typeof value !== 'string') {
    throw new DirectorConfigError(`\`${path}\` must be a string if present.`);
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function parseOptionalStringArray(value: unknown, path: string): readonly string[] | undefined {
  if (value == null) {
    return undefined;
  }
  if (!Array.isArray(value) || value.some((entry) => typeof entry !== 'string')) {
    throw new DirectorConfigError(`\`${path}\` must be an array of strings if present.`);
  }
  const trimmed = value.map((entry) => entry.trim()).filter((entry) => entry.length > 0);
  return trimmed.length > 0 ? trimmed : undefined;
}

function parseOptionalBoolean(value: unknown, path: string): boolean | undefined {
  if (value == null) {
    return undefined;
  }
  if (typeof value !== 'boolean') {
    throw new DirectorConfigError(`\`${path}\` must be a boolean if present.`);
  }
  return value;
}

function parseOptionalNonNegativeInteger(value: unknown, path: string): number | undefined {
  if (value == null) {
    return undefined;
  }
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || Math.floor(value) !== value) {
    throw new DirectorConfigError(`\`${path}\` must be a non-negative integer if present.`);
  }
  return value;
}

function parseScenario(raw: unknown): DirectorScenarioConfig | undefined {
  if (raw == null) {
    return undefined;
  }
  if (!isRecord(raw)) {
    throw new DirectorConfigError('`scenario` must be an object if present.');
  }
  return {
    name: parseOptionalString(raw.name, 'scenario.name'),
    summary: parseOptionalString(raw.summary, 'scenario.summary'),
  } satisfies DirectorScenarioConfig;
}

function parseDirectorProfile(raw: unknown): DirectorProfileConfig {
  if (!isRecord(raw)) {
    throw new DirectorConfigError('`director` must be an object.');
  }
  const role = parseOptionalString(raw.role, 'director.role');
  if (!role) {
    throw new DirectorConfigError('`director.role` is required and must be a non-empty string.');
  }
  return {
    role,
    objective: parseOptionalString(raw.objective, 'director.objective'),
    instructions: parseOptionalStringArray(raw.instructions, 'director.instructions'),
  } satisfies DirectorProfileConfig;
}

function parseHooks(raw: unknown): Readonly<Record<string, string>> | undefined {
  if (raw == null) {
    return undefined;
  }
  if (!isRecord(raw)) {
    throw new DirectorConfigError('`hooks` must be an object mapping hook names to descriptions.');
  }
  const entries: Array<[string, string]> = [];
  for (const [key, value] of Object.entries(raw)) {
    if (!IDENTIFIER_RE.test(key)) {
      throw new DirectorConfigError(`hook name ${JSON.stringify(key)} must match [A-Za-z0-9_-]+.`);
    }
    const description = parseOptionalString(value, `hooks.${key}`);
    if (!description) {
      throw new DirectorConfigError(`\`hooks.${key}\` must be a non-empty string.`);
    }
    entries.push([key, description]);
  }
  return Object.fromEntries(entries);
}

function parseResponseSchema(raw: unknown, path: string): ResponseSchema {
  if (!isRecord(raw)) {
    throw new DirectorConfigError(`\`${path}\` must be an object.`);
  }
  const type = raw.type;
  if (typeof type !== 'string') {
    throw new DirectorConfigError(`\`${path}.type\` is required.`);
  }
  const description = parseOptionalString(raw.description, `${path}.description`);
  const nullable = parseOptionalBoolean(raw.nullable, `${path}.nullable`) ?? false;

  switch (type) {
    case 'string': {
      const maxLength = parseOptionalNonNegativeInteger(raw.maxLength, `${path}.maxLength`);
      const enumValues = parseOptionalStringArray(raw.enum, `${path}.enum`);
      return {
        type: 'string',
        ...(description ? { description } : {}),
        ...(nullable ? { nullable: true } : {}),
        ...(maxLength != null ? { maxLength } : {}),
        ...(enumValues ? { enum: enumValues } : {}),
      } satisfies ResponseStringSchema;
    }
    case 'number': {
      const integer = parseOptionalBoolean(raw.integer, `${path}.integer`);
      return {
        type: 'number',
        ...(description ? { description } : {}),
        ...(nullable ? { nullable: true } : {}),
        ...(integer != null ? { integer } : {}),
      } satisfies ResponseNumberSchema;
    }
    case 'boolean':
      return {
        type: 'boolean',
        ...(description ? { description } : {}),
        ...(nullable ? { nullable: true } : {}),
      } satisfies ResponseBooleanSchema;
    case 'null':
      if (nullable) {
        throw new DirectorConfigError(`\`${path}.nullable\` is invalid for type \`null\`.`);
      }
      return { type: 'null' } satisfies ResponseNullSchema;
    case 'array': {
      const items = parseResponseSchema(raw.items, `${path}.items`);
      const maxItems = parseOptionalNonNegativeInteger(raw.maxItems, `${path}.maxItems`);
      return {
        type: 'array',
        items,
        ...(description ? { description } : {}),
        ...(nullable ? { nullable: true } : {}),
        ...(maxItems != null ? { maxItems } : {}),
      } satisfies ResponseArraySchema;
    }
    case 'object': {
      if (!isRecord(raw.properties)) {
        throw new DirectorConfigError(`\`${path}.properties\` must be an object.`);
      }
      const entries = Object.entries(raw.properties);
      if (entries.length === 0) {
        throw new DirectorConfigError(`\`${path}.properties\` must define at least one field.`);
      }
      const properties = Object.fromEntries(
        entries.map(([key, value]) => {
          if (!IDENTIFIER_RE.test(key)) {
            throw new DirectorConfigError(
              `response property ${JSON.stringify(key)} at \`${path}\` must match [A-Za-z0-9_-]+.`
            );
          }
          return [key, parseResponseSchema(value, `${path}.properties.${key}`)];
        })
      );
      return {
        type: 'object',
        properties,
        ...(description ? { description } : {}),
        ...(nullable ? { nullable: true } : {}),
      } satisfies ResponseObjectSchema;
    }
    default:
      throw new DirectorConfigError(`Unsupported response schema type at \`${path}.type\`: ${JSON.stringify(type)}.`);
  }
}

function parseQuery(
  name: string,
  raw: unknown,
  hooks: Readonly<Record<string, string>> | undefined
): DirectorQueryConfig {
  if (!isRecord(raw)) {
    throw new DirectorConfigError(`\`queries.${name}\` must be an object.`);
  }
  const queryHooks = parseOptionalStringArray(raw.hooks, `queries.${name}.hooks`);
  if (queryHooks) {
    for (const hookName of queryHooks) {
      if (!hooks || !(hookName in hooks)) {
        throw new DirectorConfigError(
          `\`queries.${name}.hooks\` references unknown hook ${JSON.stringify(hookName)}.`
        );
      }
    }
  }
  return {
    description: parseOptionalString(raw.description, `queries.${name}.description`),
    instructions: parseOptionalStringArray(raw.instructions, `queries.${name}.instructions`),
    ...(queryHooks ? { hooks: queryHooks } : {}),
    response: parseResponseSchema(raw.response, `queries.${name}.response`),
  } satisfies DirectorQueryConfig;
}

export function parseDirectorConfig(raw: unknown): DirectorConfig {
  if (!isRecord(raw)) {
    throw new DirectorConfigError('Director config must be a JSON object.');
  }

  if (typeof raw.id !== 'string' || !IDENTIFIER_RE.test(raw.id)) {
    throw new DirectorConfigError('`id` must be a non-empty identifier ([A-Za-z0-9_-]+).');
  }

  const hooks = parseHooks(raw.hooks);

  if (!isRecord(raw.queries)) {
    throw new DirectorConfigError('`queries` must be an object.');
  }
  const queryEntries = Object.entries(raw.queries);
  if (queryEntries.length === 0) {
    throw new DirectorConfigError('`queries` must contain at least one named query.');
  }
  const queries = Object.fromEntries(
    queryEntries.map(([name, query]) => {
      if (!IDENTIFIER_RE.test(name)) {
        throw new DirectorConfigError(`query name ${JSON.stringify(name)} must match [A-Za-z0-9_-]+.`);
      }
      return [name, parseQuery(name, query, hooks)];
    })
  );

  return {
    id: raw.id,
    scenario: parseScenario(raw.scenario),
    director: parseDirectorProfile(raw.director),
    ...(hooks ? { hooks } : {}),
    queries,
  } satisfies DirectorConfig;
}
