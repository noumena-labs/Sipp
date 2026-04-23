//////////////////////////////////////////////////////////////////////////////
//
// director-prompt.ts
//
// - Prompt renderers for the generic director runtime.
// - The system prompt is config-driven and query-agnostic so a single
//   context key can reuse its cached prefix across many app queries.
//
//////////////////////////////////////////////////////////////////////////////

import type {
  DirectorConfig,
  DirectorQueryConfig,
  DirectorQueryPayload,
  JsonValue,
} from './director-types.js';
import { renderResponseSchemaSummary } from './response-schema.js';

export function renderDirectorSystemPrompt(config: DirectorConfig): string {
  const lines: string[] = [];
  const scenarioName = config.scenario?.name?.trim();
  lines.push(
    scenarioName
      ? `You are the director brain for the scenario \"${scenarioName}\".`
      : 'You are the director brain for a host application scenario.'
  );
  lines.push('Reason only from the supplied query payload. Do not invent unseen facts.');
  lines.push('Return exactly one JSON value that matches the requested response contract.');

  lines.push(`Role: ${config.director.role}`);
  if (config.director.objective) {
    lines.push(`Objective: ${config.director.objective}`);
  }
  if (config.scenario?.summary) {
    lines.push(`Scenario: ${config.scenario.summary}`);
  }
  if (config.director.instructions && config.director.instructions.length > 0) {
    lines.push('Instructions:');
    for (const instruction of config.director.instructions) {
      lines.push(`- ${instruction}`);
    }
  }
  if (config.hooks && Object.keys(config.hooks).length > 0) {
    lines.push('Hook glossary:');
    for (const [name, description] of Object.entries(config.hooks)) {
      lines.push(`- ${name}: ${description}`);
    }
  }

  return lines.join('\n');
}

export function renderDirectorUserMessage(
  config: DirectorConfig,
  queryName: string,
  query: DirectorQueryConfig,
  payload: DirectorQueryPayload
): string {
  const lines: string[] = [];
  lines.push(`Query: ${queryName}`);
  if (query.description) {
    lines.push(`Description: ${query.description}`);
  }
  if (query.instructions && query.instructions.length > 0) {
    lines.push('Query instructions:');
    for (const instruction of query.instructions) {
      lines.push(`- ${instruction}`);
    }
  }
  if (query.hooks && query.hooks.length > 0 && config.hooks) {
    lines.push('Relevant hooks:');
    for (const hookName of query.hooks) {
      const description = config.hooks[hookName];
      if (description) {
        lines.push(`- ${hookName}: ${description}`);
      }
    }
  }
  lines.push('Response contract:');
  lines.push(renderResponseSchemaSummary(query.response));

  const sections = Object.entries(payload).filter(([, value]) => value !== undefined);
  if (sections.length > 0) {
    lines.push('Payload:');
    for (const [sectionName, value] of sections) {
      lines.push(`${sectionName}:`);
      lines.push(renderJson(value!));
    }
  } else {
    lines.push('Payload: {}');
  }

  lines.push('Respond with JSON only. No prose, no markdown fences, no extra keys.');
  return lines.join('\n\n');
}

function renderJson(value: JsonValue): string {
  return JSON.stringify(value, null, 2);
}
