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
  readonly description?: string;
  readonly value: NonNullable<DirectorRunRequest['inputs']>[string];
}

export function renderDirectorSystemPrompt(config: DirectorConfig): string {
  const lines: string[] = [];
  const scenarioName = config.scenario?.name?.trim();
  lines.push(
    scenarioName
      ? `You are the director brain for the scenario "${scenarioName}".`
      : 'You are the director brain for a host application scenario.'
  );
  lines.push('Reason only from the supplied task inputs. Do not invent unseen facts.');
  lines.push('For selection tasks, output only the requested choice id format.');
  lines.push('For text tasks, write concise plain text. Never output JSON.');

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
  if (config.inputs && Object.keys(config.inputs).length > 0) {
    lines.push('Input glossary:');
    for (const [name, input] of Object.entries(config.inputs)) {
      lines.push(`- ${name} (${input.kind}): ${input.description}`);
    }
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
    lines.push('Task instructions:');
    for (const instruction of task.instructions) {
      lines.push(`- ${instruction}`);
    }
  }
  lines.push('Output shape:');
  lines.push(renderOutputInstructions(task.output, resolved));

  const inputSections = collectInputSections(config, task, request);
  if (inputSections.length > 0) {
    lines.push('Inputs:');
    for (const input of inputSections) {
      lines.push(`${input.inputName}:`);
      if (input.description) {
        lines.push(`Description: ${input.description}`);
      }
      lines.push(renderInput(input, media, mediaMarker));
    }
  } else {
    lines.push('Inputs: none');
  }

  return { text: lines.join('\n\n'), media };
}

function renderOutputInstructions<TPayload>(
  output: DirectorOutputConfig,
  resolved: ResolvedDirectorChoices<TPayload>
): string {
  switch (output.shape) {
    case 'select_one':
      return [
        'Select exactly one choice. Output only the choice id. No prose.',
        renderChoiceList(resolved.choices ?? []),
      ].join('\n');
    case 'select_many':
      return [
        `Select ${output.min ?? 0} to ${output.max ?? 'all'} choices. Output one choice id per line. No prose.`,
        renderChoiceList(resolved.choices ?? []),
      ].join('\n');
    case 'select_slots':
      return renderSlotInstructions(output, resolved);
    case 'text':
      return `Write plain text only.${output.maxLength ? ` Keep it at most ${output.maxLength} characters.` : ''}`;
    case 'text_with_directives':
      return [
        `Write plain text.${output.maxLength ? ` Keep it at most ${output.maxLength} characters.` : ''}`,
        'When a grounded app action is useful, include the directive id in square brackets.',
        `${output.maxDirectives ? `Use at most ${output.maxDirectives} directives.` : 'Use only needed directives.'}`,
        renderChoiceList(resolved.directives ?? [], 'Available directives'),
      ].join('\n');
  }
}

function renderSlotInstructions<TPayload>(
  output: Extract<DirectorOutputConfig, { shape: 'select_slots' }>,
  resolved: ResolvedDirectorChoices<TPayload>
): string {
  const lines = ['Select exactly one choice for each slot. Output one line per slot as slot=choice. No prose.'];
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
    const label = choice.label ? ` - ${choice.label}` : '';
    const description = choice.description ? `: ${choice.description}` : '';
    lines.push(`- ${choice.id}${label}${description}`);
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
  const seen = new Set<string>();
  const sections: RenderInputContext[] = [];

  for (const inputName of orderedNames) {
    const value = supplied[inputName];
    if (value === undefined) {
      continue;
    }
    seen.add(inputName);
    const slot = config.inputs?.[inputName];
    sections.push({
      inputName,
      configuredKind: slot?.kind,
      description: slot?.description,
      value,
    });
  }

  for (const [inputName, value] of Object.entries(supplied)) {
    if (seen.has(inputName) || value === undefined) {
      continue;
    }
    const slot = config.inputs?.[inputName];
    sections.push({
      inputName,
      configuredKind: slot?.kind,
      description: slot?.description,
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
  return JSON.stringify(value, null, 2);
}
