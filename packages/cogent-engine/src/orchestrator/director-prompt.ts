//////////////////////////////////////////////////////////////////////////////
//
// director-prompt.ts
//
// - Prompt renderers for the shape-driven director runtime.
//
//////////////////////////////////////////////////////////////////////////////

import type {
  DirectorChoice,
  DirectorConfig,
  DirectorInputKind,
  DirectorInputValue,
  DirectorOutputConfig,
  DirectorRunRequest,
  DirectorTaskConfig,
  JsonValue,
} from './director-types.js';
import type { ResolvedDirectorChoices } from './director-output.js';

export interface RenderedDirectorUserMessage {
  readonly text: string;
  readonly media: readonly Uint8Array[];
}

interface RenderInputContext {
  readonly inputName: string;
  readonly configuredKind?: DirectorInputKind;
  readonly value: NonNullable<DirectorRunRequest['inputs']>[string];
}

export function renderDirectorSystemPrompt(config: DirectorConfig): string {
  const lines: string[] = [];
  const scenarioName = config.scenario?.name?.trim();
  lines.push(scenarioName ? `Director for ${scenarioName}.` : 'Director for host scenario.');
  lines.push('Use only supplied inputs; do not invent facts.');
  lines.push('Selection tasks: output only a choice id. Text tasks: concise plain text, no JSON.');

  lines.push(`Role: ${config.director.role}`);
  if (config.director.objective) {
    lines.push(`Goal: ${config.director.objective}`);
  }
  if (config.scenario?.summary) {
    lines.push(`Scenario: ${config.scenario.summary}`);
  }
  if (config.director.instructions && config.director.instructions.length > 0) {
    lines.push(`Rules: ${config.director.instructions.join(' ')}`);
  }

  return lines.join('\n');
}

export function renderDirectorUserMessage<TPayload>(
  config: DirectorConfig,
  taskName: string,
  task: DirectorTaskConfig,
  request: DirectorRunRequest<TPayload>,
  resolved: ResolvedDirectorChoices<TPayload>,
  mediaMarker: string | null
): RenderedDirectorUserMessage {
  const lines: string[] = [];
  const media: Uint8Array[] = [];
  lines.push(`Task: ${taskName}`);
  if (task.purpose) {
    lines.push(`Purpose: ${task.purpose}`);
  }
  if (task.instructions && task.instructions.length > 0) {
    lines.push(`Instructions: ${task.instructions.join(' ')}`);
  }
  lines.push(renderOutputInstructions(task.output, resolved));

  const inputSections = collectInputSections(config, task, request);
  if (inputSections.length > 0) {
    lines.push('Inputs:');
    for (const input of inputSections) {
      lines.push(`${input.inputName}:`);
      lines.push(renderInput(input, media, mediaMarker));
    }
  } else {
    lines.push('Inputs: none');
  }

  return { text: lines.join('\n'), media };
}

function renderOutputInstructions<TPayload>(
  output: DirectorOutputConfig,
  resolved: ResolvedDirectorChoices<TPayload>
): string {
  switch (output.shape) {
    case 'select_one':
      return [
        'Output one choice id only.',
        renderChoiceList(resolved.choices ?? []),
      ].join('\n');
    case 'select_many':
      return [
        `Output ${output.min ?? 0} to ${output.max ?? 'all'} choice ids, one per line.`,
        renderChoiceList(resolved.choices ?? []),
      ].join('\n');
    case 'select_slots':
      return renderSlotInstructions(output, resolved);
    case 'text':
      return `Plain text only.${output.maxLength ? ` Max ${output.maxLength} chars.` : ''}`;
    case 'text_with_directives':
      return [
        `Plain text.${output.maxLength ? ` Max ${output.maxLength} chars.` : ''}`,
        `Use directive ids in brackets only when useful.${output.maxDirectives ? ` Max ${output.maxDirectives}.` : ''}`,
        renderChoiceList(resolved.directives ?? [], 'Available directives'),
      ].join('\n');
  }
}

function renderSlotInstructions<TPayload>(
  output: Extract<DirectorOutputConfig, { shape: 'select_slots' }>,
  resolved: ResolvedDirectorChoices<TPayload>
): string {
  const lines = ['Output one line per slot as slot=choice. No prose.'];
  for (const slot of output.slots) {
    lines.push(`Slot ${slot.name}${slot.description ? `: ${slot.description}` : ''}`);
    lines.push(renderChoiceList(resolved.slotChoices?.[slot.name] ?? []));
  }
  return lines.join('\n');
}

function renderChoiceList(
  choices: readonly DirectorChoice[],
  title = 'Available choices'
): string {
  const lines = [`${title}:`];
  for (const choice of choices) {
    const label = choice.label ? `=${choice.label}` : '';
    lines.push(`- ${choice.id}${label}`);
  }
  return lines.join('\n');
}

function collectInputSections<TPayload>(
  config: DirectorConfig,
  task: DirectorTaskConfig,
  request: DirectorRunRequest<TPayload>
): RenderInputContext[] {
  const supplied = request.inputs ?? {};
  const orderedNames = task.inputs ?? Object.keys(supplied);
  const sections: RenderInputContext[] = [];

  for (const inputName of orderedNames) {
    const value = supplied[inputName];
    if (value === undefined) {
      continue;
    }
    const slot = config.inputs?.[inputName];
    sections.push({
      inputName,
      configuredKind: slot?.kind,
      value,
    });
  }

  return sections;
}

function renderInput(
  input: RenderInputContext,
  media: Uint8Array[],
  mediaMarker: string | null
): string {
  const envelope = normalizeInputEnvelope(input.value);
  if (envelope?.kind === 'text') {
    return envelope.text.trim();
  }
  if (envelope?.kind === 'data') {
    return renderJson(envelope.value);
  }
  if (envelope?.kind === 'image') {
    if (!mediaMarker) {
      throw new Error(`input ${JSON.stringify(input.inputName)} is an image, but the loaded runtime has no media marker.`);
    }
    media.push(envelope.media);
    return `${envelope.description ? `${envelope.description}\n` : ''}${mediaMarker}`;
  }

  if (input.configuredKind === 'text' && typeof input.value === 'string') {
    return input.value.trim();
  }
  return renderJson(input.value as JsonValue);
}

function normalizeInputEnvelope(
  value: DirectorInputValue | undefined
): { kind: 'text'; text: string } | { kind: 'data'; value: JsonValue } | { kind: 'image'; media: Uint8Array; description?: string } | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  const record = value as Record<string, unknown>;
  if (record.kind === 'text' && typeof record.text === 'string') {
    return { kind: 'text', text: record.text };
  }
  if (record.kind === 'data' && 'value' in record) {
    return { kind: 'data', value: record.value as JsonValue };
  }
  if (record.kind === 'image' && record.media instanceof Uint8Array) {
    return {
      kind: 'image',
      media: record.media,
      ...(typeof record.description === 'string' ? { description: record.description } : {}),
    };
  }
  return null;
}

function renderJson(value: JsonValue): string {
  return JSON.stringify(value);
}
