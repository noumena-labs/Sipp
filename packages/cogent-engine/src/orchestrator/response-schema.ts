//////////////////////////////////////////////////////////////////////////////
//
// response-schema.ts
//
// - Runtime validator and human-readable summarizer for response schemas.
// - The grammar constrains the syntax; this file validates the parsed JSON
//   value against the declared structural contract.
//
//////////////////////////////////////////////////////////////////////////////

import type { JsonObject, JsonValue, ResponseSchema } from './director-types.js';

function isObject(value: JsonValue): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function validateResponseValue(
  value: JsonValue,
  schema: ResponseSchema,
  path = '$'
): string | null {
  if (schema.type !== 'null' && schema.nullable && value === null) {
    return null;
  }

  switch (schema.type) {
    case 'null':
      return value === null ? null : `${path} must be null.`;
    case 'string': {
      if (typeof value !== 'string') {
        return `${path} must be a string.`;
      }
      if (schema.maxLength != null && value.length > schema.maxLength) {
        return `${path} must be at most ${schema.maxLength} characters.`;
      }
      if (schema.enum && !schema.enum.includes(value)) {
        return `${path} must be one of [${schema.enum.join(', ')}].`;
      }
      return null;
    }
    case 'number':
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        return `${path} must be a finite number.`;
      }
      if (schema.integer && Math.floor(value) !== value) {
        return `${path} must be an integer.`;
      }
      return null;
    case 'boolean':
      return typeof value === 'boolean' ? null : `${path} must be a boolean.`;
    case 'array': {
      if (!Array.isArray(value)) {
        return `${path} must be an array.`;
      }
      if (schema.maxItems != null && value.length > schema.maxItems) {
        return `${path} must contain at most ${schema.maxItems} items.`;
      }
      for (let i = 0; i < value.length; i += 1) {
        const error = validateResponseValue(value[i]!, schema.items, `${path}[${i}]`);
        if (error) {
          return error;
        }
      }
      return null;
    }
    case 'object': {
      if (!isObject(value)) {
        return `${path} must be an object.`;
      }
      const allowedKeys = Object.keys(schema.properties);
      for (const key of Object.keys(value)) {
        if (!(key in schema.properties)) {
          return `${path}.${key} is not allowed.`;
        }
      }
      for (const key of allowedKeys) {
        if (!(key in value)) {
          return `${path}.${key} is required.`;
        }
        const error = validateResponseValue(value[key]!, schema.properties[key]!, `${path}.${key}`);
        if (error) {
          return error;
        }
      }
      return null;
    }
  }
}

export function renderResponseSchemaSummary(schema: ResponseSchema): string {
  return renderSummary(schema, 0);
}

function renderSummary(schema: ResponseSchema, depth: number): string {
  const pad = '  '.repeat(depth);
  const nullable = schema.type !== 'null' && schema.nullable ? ' | null' : '';

  switch (schema.type) {
    case 'null':
      return `${pad}null`;
    case 'string': {
      const details: string[] = [];
      if (schema.enum && schema.enum.length > 0) {
        details.push(`enum: [${schema.enum.join(', ')}]`);
      }
      if (schema.maxLength != null) {
        details.push(`maxLength: ${schema.maxLength}`);
      }
      return `${pad}string${nullable}${details.length > 0 ? ` (${details.join('; ')})` : ''}`;
    }
    case 'number':
      return `${pad}${schema.integer ? 'integer' : 'number'}${nullable}`;
    case 'boolean':
      return `${pad}boolean${nullable}`;
    case 'array':
      return `${pad}array${nullable}\n${pad}items:\n${renderSummary(schema.items, depth + 1)}`;
    case 'object': {
      const lines = [`${pad}object${nullable}`];
      for (const [key, value] of Object.entries(schema.properties)) {
        lines.push(`${pad}${key}:`);
        lines.push(renderSummary(value, depth + 1));
      }
      return lines.join('\n');
    }
  }
}
